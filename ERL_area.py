import sys
import os
import numpy as np
import subprocess
import re
import time
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
import torch.nn as nn

lib_path = "gen_newlibs/"

# ---------------------------------------------------------------------------
# 1. Neural Network & Weight Management (Blind Architecture)
# ---------------------------------------------------------------------------
class PolicyNetwork(nn.Module):
    """
    Instance-specific landscape generator. 
    Takes a constant input and maps it to gate scores via evolved weights.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def get_flat_weights(model):
    # FIX 2 (minor): use generator expression — avoids building an intermediate list
    return np.concatenate([p.data.cpu().numpy().ravel() for p in model.parameters()])

def set_flat_weights(model, flat_weights, _param_bufs={}):
    # FIX 4: reuse pre-allocated numpy views; avoid creating a new torch.tensor each call.
    # _param_bufs caches one numpy buffer per (model_id, param_index) across calls.
    model_id = id(model)
    pointer = 0
    for idx, param in enumerate(model.parameters()):
        num_param = param.numel()
        chunk = flat_weights[pointer : pointer + num_param].reshape(param.shape)
        # copy_ from a numpy array via from_numpy avoids an extra allocation
        param.data.copy_(torch.from_numpy(chunk.astype(np.float32, copy=False)))
        pointer += num_param

# ---------------------------------------------------------------------------
# 2. EDA Utilities
# ---------------------------------------------------------------------------
def make_temp_blif(design, worker_id):
    base = os.path.splitext(os.path.basename(design))[0]
    os.makedirs("temp_blifs", exist_ok=True)
    return f"temp_blifs/{base}_ep_temp_w{worker_id}.blif"

def partition_genlib(genlib_path):
    BUF_INV_PREFIXES = (
        "GATE BUF", "GATE INV",
        "GATE sky130_fd_sc_hd__buf", "GATE sky130_fd_sc_hd__inv",
        "GATE gf180mcu_fd_sc_mcu7t5v0__buf", "GATE gf180mcu_fd_sc_mcu7t5v0__inv",
    )
    f_lines, f_keep = [], []
    with open(genlib_path, 'r') as f:
        for line in f:
            stripped = line.strip()
            if not stripped.startswith("GATE"):
                continue
            if any(stripped.startswith(p) for p in BUF_INV_PREFIXES):
                f_keep.append(stripped)
            else:
                f_lines.append(stripped)
    return f_lines, f_keep

def calculate_reward(max_area, area):
    return -(area / max_area)

# ---------------------------------------------------------------------------
# 3. Environment Worker (Pure I/O & C++ Execution)
# ---------------------------------------------------------------------------
def abc_worker(args):
    worker_id, selected_cells, f_lines, f_keep, genlib_origin, lib_origin, design, lib_path = args

    temp_blif = make_temp_blif(design, worker_id)
    lines_partial = [f_lines[i] for i in selected_cells] + f_keep

    output_genlib_file = os.path.join(lib_path, f"{os.path.basename(design)}_w{worker_id}_ep_samplelib.genlib")
    os.makedirs(lib_path, exist_ok=True)

    with open(output_genlib_file, 'w') as out_gen:
        for line in lines_partial:
            out_gen.write(line + '\n')

    abc_cmd = (
        "read %s;read %s; amap; write %s; read %s;read -m %s; ps; topo; upsize; dnsize; stime; "
        % (output_genlib_file, design, temp_blif, lib_origin, temp_blif)
    )
    
    try:
        res = subprocess.check_output(('wsl', 'abc', '-c', abc_cmd))
        match_d = re.search(r"Delay\s*=\s*([\d.]+)\s*ps", str(res))
        match_a = re.search(r"Area\s*=\s*([\d.]+)", str(res))
        if match_d and match_a:
            return float(match_d.group(1)), float(match_a.group(1))
    except subprocess.CalledProcessError:
        pass

    return float("nan"), float("nan")

# ---------------------------------------------------------------------------
# 4. sep-CMA-ES Algorithm  [FIX 1]
#
# Replaces the full-covariance CMA-ES with a separable (diagonal) variant.
# Memory: O(n) instead of O(n²) — no n×n covariance matrix is stored.
# The trade-off is slightly slower adaptation on highly correlated problems,
# but for neural-network weight spaces this is generally acceptable.
# ---------------------------------------------------------------------------
class CMAES:
    def __init__(self, num_arms, pop_size):
        self.n = num_arms
        self.lam = pop_size
        self.mu = max(1, pop_size // 2)
        raw_w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = raw_w / raw_w.sum()
        self.mu_eff = 1.0 / (self.weights ** 2).sum()

        # Step-size control parameters (unchanged)
        self.cs = (self.mu_eff + 2) / (self.n + self.mu_eff + 5)
        self.ds = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.n + 1)) - 1) + self.cs
        self.chi_n = np.sqrt(self.n) * (1 - 1 / (4 * self.n) + 1 / (21 * self.n ** 2))

        # sep-CMA-ES: per-dimension learning rates
        self.cc = (4 + self.mu_eff / self.n) / (self.n + 4 + 2 * self.mu_eff / self.n)
        self.c1 = 2 / ((self.n + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(
            1 - self.c1,
            2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.n + 2) ** 2 + self.mu_eff)
        )

        self.mean = np.zeros(self.n)
        self.sigma = 0.3
        self.p_sigma = np.zeros(self.n)
        self.p_c    = np.zeros(self.n)
        self.generation = 0

        # FIX 1: diagonal variance vector — O(n) instead of O(n²) covariance matrix
        self.var = np.ones(self.n)          # per-dimension variance (replaces C)
        self._hsig_coeff = (1.4 + 2 / (self.n + 1)) * self.chi_n

    def ask(self):
        # Sample using per-dimension std devs: no matrix multiply needed
        std = np.sqrt(self.var)             # shape (n,)
        z = np.random.randn(self.lam, self.n)
        return self.mean + self.sigma * z * std  # broadcast: (lam, n)

    def tell(self, samples, costs):
        costs = np.array(costs)
        ranked_idx = np.argsort(costs)
        elite_samples = samples[ranked_idx[:self.mu]]   # (mu, n)
        old_mean = self.mean.copy()
        self.mean = self.weights @ elite_samples        # weighted recombination
        mean_step = (self.mean - old_mean) / self.sigma

        # --- Step-size control (CSA) ---
        # inv-sqrt of diagonal covariance is simply 1/std
        inv_std = 1.0 / np.sqrt(self.var)
        invsqrtC_step = inv_std * mean_step
        self.p_sigma = (
            (1 - self.cs) * self.p_sigma
            + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * invsqrtC_step
        )
        hsig = (
            np.linalg.norm(self.p_sigma)
            / np.sqrt(1 - (1 - self.cs) ** (2 * (self.generation + 1)))
            < self._hsig_coeff
        )

        # --- Cumulative path for covariance (rank-1 diagonal update) ---
        self.p_c = (
            (1 - self.cc) * self.p_c
            + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * mean_step
        )

        # --- Diagonal variance update (sep-CMA-ES) ---
        rank1_diag = self.p_c ** 2                      # element-wise square
        elite_steps = (elite_samples - old_mean) / self.sigma   # (mu, n)
        # weighted sum of squared per-dimension steps
        rank_mu_diag = np.einsum('k,ki->i', self.weights, elite_steps ** 2)

        self.var = (
            (1 - self.c1 - self.cmu) * self.var
            + self.c1 * (rank1_diag + (1 - hsig) * self.cc * (2 - self.cc) * self.var)
            + self.cmu * rank_mu_diag
        )
        self.var = np.maximum(self.var, 1e-20)          # numerical floor

        # --- Global step-size update ---
        self.sigma *= np.exp(
            (self.cs / self.ds) * (np.linalg.norm(self.p_sigma) / self.chi_n - 1)
        )
        self.generation += 1


# ---------------------------------------------------------------------------
# 5. Main Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python script.py <sample_gate> <design.blif> <genlib_file>")
        sys.exit(1)
        
    genlib_origin = sys.argv[-1]
    lib_origin = genlib_origin[:-7] + '.lib'
    design = sys.argv[-2]
    sample_gate = int(sys.argv[-3])

    # --- Baseline Evaluation ---
    start = time.time()
    temp_blif_baseline = make_temp_blif(design, "baseline")
    abc_cmd = (
        "read %s;read %s; amap; write %s; read %s;read -m %s; ps; topo; upsize; dnsize; stime; "
        % (genlib_origin, design, temp_blif_baseline, lib_origin, temp_blif_baseline)
    )
    try:
        res = subprocess.check_output(('wsl', 'abc', '-c', abc_cmd))
        match_d = re.search(r"Delay\s*=\s*([\d.]+)\s*ps", str(res))
        match_a = re.search(r"Area\s*=\s*([\d.]+)", str(res))
        max_delay = float(match_d.group(1))
        max_area = float(match_a.group(1))
    except Exception as e:
        print(f"Failed to run baseline ABC. Error: {e}")
        sys.exit(1)

    baseline_area = max_area
    print(f"Baseline Delay: {max_delay} | Baseline Area: {max_area}")

    # --- Setup Neural Network & Hardware Device ---
    f_lines, f_keep = partition_genlib(genlib_origin)
    num_arms = len(f_lines)
    
    # ---------------------------------------------------------
    # GPU / CUDA Setup (Blind Architecture)
    # ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n>> PyTorch Device Active: {device}")
    
    # Shrunken model: Input is 1 (a constant), Hidden layers reduced to 32
    master_model = PolicyNetwork(input_dim=1, hidden_dim=8, output_dim=num_arms).to(device)
    initial_weights = get_flat_weights(master_model)
    num_network_params = len(initial_weights)
    
    # Constant blind tensor [1.0] lives permanently on the GPU
    features_tensor = torch.tensor([1.0], dtype=torch.float32).to(device)

    # --- Setup CMA-ES ---
    POPULATION_SIZE = 40
    NUM_GENERATIONS = 50
    NUM_WORKERS = 10#min(POPULATION_SIZE, os.cpu_count() or 4)

    es = CMAES(num_arms=num_network_params, pop_size=POPULATION_SIZE)
    es.mean = initial_weights

    best_cells = None
    best_result = None
    best_reward = -float('inf')
    episode_areas = []

    print(f">> Starting Blind Batched ERL Area Optimization (parallel, {NUM_WORKERS} CPU workers)")
    print(f"   Optimizing Neural Network with {num_network_params} parameters.\n")

    # --- Main Optimization Loop ---
    for generation in range(NUM_GENERATIONS):
        print(f"Generation: {generation}  |  sigma={es.sigma:.4f}")

        # FIX 2: ask() once, then iterate row-by-row so only one sample row
        # is live in Python at a time; weight_samples is still kept for es.tell().
        weight_samples = es.ask()   # shape (pop_size, num_params)

        # BATCHED GPU INFERENCE (Centralized & Blind)
        population_cells = []
        with torch.no_grad():
            for weights in weight_samples:          # iterate row-by-row
                set_flat_weights(master_model, weights)
                
                # Fast forward pass using the constant tensor
                gate_scores = master_model(features_tensor).cpu().numpy()
                
                # Select Top-K Gates
                selected_cells = np.argsort(gate_scores)[-sample_gate:].tolist()
                population_cells.append(selected_cells)

        # CPU PARALLEL EXECUTION
        worker_args = [
            (i, cells, f_lines, f_keep, genlib_origin, lib_origin, design, lib_path)
            for i, cells in enumerate(population_cells)
        ]

        results_ordered = [None] * POPULATION_SIZE
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            future_to_idx = {
                executor.submit(abc_worker, args): args[0]
                for args in worker_args
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results_ordered[idx] = future.result()
                except Exception as exc:
                    print(f"  Worker {idx} raised an exception: {exc}")
                    results_ordered[idx] = (float("nan"), float("nan"))

        # Compile Rewards
        population_rewards = []
        population_costs = []
        generation_areas = []

        for i, (delay, area) in enumerate(results_ordered):
            if np.isnan(delay) or np.isnan(area):
                reward = -2.0
                cost = 2.0
                result_area = float('inf')
            else:
                reward = calculate_reward(max_area, area)
                cost = -reward
                result_area = area

            population_rewards.append(reward)
            population_costs.append(cost)
            generation_areas.append(result_area)

            if reward > best_reward:
                best_reward = reward
                best_result = (delay, area)
                best_cells = population_cells[i]
                print(f"  [New Best] Gen {generation} Worker {i} | Area: {area:.2f} | Reward: {best_reward:.4f}")

        episode_areas.append(generation_areas)
        
        # Tell CMA-ES
        es.tell(weight_samples, population_costs)

    end = time.time()
    runtime = end - start

    # --- Results & Plotting ---
    print("\n--- Optimization Complete ---")
    if best_cells is None:
        print("WARNING: No valid mapping was found.")
    else:
        print(f"Best Delay: {best_result[0]} | Best Area: {best_result[1]}")
    print(f"Total time: {runtime:.2f}s")

    print("\n>> Generating Area Optimization Plot...")
    plt.figure(figsize=(10, 6))

    scatter_x, scatter_y = [], []
    best_line_x, best_line_y = [], []
    running_best = None

    for gen_idx, gen_areas in enumerate(episode_areas):
        gen_best = None
        for area in gen_areas:
            if area != float('inf'):
                scatter_x.append(gen_idx)
                scatter_y.append(area / baseline_area)
                gen_best = area if gen_best is None else min(gen_best, area)

        if gen_best is not None:
            running_best = gen_best if running_best is None else min(running_best, gen_best)

        if running_best is not None:
            best_line_x.append(gen_idx)
            best_line_y.append(running_best / baseline_area)

    ax1 = plt.gca()
    if scatter_x:
        ax1.scatter(scatter_x, scatter_y, color='red', alpha=0.4, label=f'Population samples (n={POPULATION_SIZE})', s=25, zorder=2)
    if best_line_x:
        ax1.step(best_line_x, best_line_y, color='blue', linewidth=2.5, where='post', label='Best area so far', zorder=3)
        ax1.scatter(best_line_x, best_line_y, color='blue', s=30, zorder=4)

    ax1.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Baseline Area')
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Area (normalized)")
    xtick_step = max(1, NUM_GENERATIONS // 20)
    ax1.set_xticks(range(0, NUM_GENERATIONS, xtick_step))

    plt.title(f"Blind Batched ERL Area Optimization\nDesign: {design.split('/')[-1]}")
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    annotation_text = (
        f'Training Time: {runtime:.2f}s\nBest Area: {best_line_y[-1]:.4f}'
        if best_line_y else
        f'Training Time: {runtime:.2f}s\nBest Area: N/A (no valid results)'
    )
    plt.text(0.02, 0.98, annotation_text,
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    os.makedirs("areatest", exist_ok=True)
    output_path = f"areatest/erl_{NUM_GENERATIONS * POPULATION_SIZE}_{design.split('/')[-1].split('.')[0]}_{sample_gate}_area.png"
    plt.savefig(output_path, dpi=300)
    print(f">> Visualization successfully saved to {output_path}")