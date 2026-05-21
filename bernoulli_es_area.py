import sys
import os
import numpy as np
import subprocess
import re
import time
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

lib_path = "gen_newlibs/"


def make_temp_blif(design, worker_id):
    base = os.path.splitext(os.path.basename(design))[0]
    os.makedirs("temp_blifs", exist_ok=True)
    return f"temp_blifs/{base}_ep_temp_w{worker_id}.blif"


# --- Single-pass library partitioner ---
def partition_genlib(genlib_path):
    BUF_INV_PREFIXES = (
        "GATE BUF",
        "GATE INV",
        "GATE sky130_fd_sc_hd__buf",
        "GATE sky130_fd_sc_hd__inv",
        "GATE gf180mcu_fd_sc_mcu7t5v0__buf",
        "GATE gf180mcu_fd_sc_mcu7t5v0__inv",
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


# --- Technology mapper (runs inside each worker process) ---
def technology_mapper(args):
    worker_id, genlib_origin, lib_origin, design, partial_cell_library, lib_path = args

    temp_blif = make_temp_blif(design, worker_id)
    f_lines, f_keep = partition_genlib(genlib_origin)

    lines_partial = [f_lines[i] for i in partial_cell_library] + f_keep

    output_genlib_file = os.path.join(
        lib_path,
        f"{os.path.basename(design)}_w{worker_id}_ep_samplelib.genlib"
    )
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


# --- Reward ---
def calculate_reward(max_area, area):
    return -(area / max_area)


# ---------------------------------------------------------------------------
# Adam Optimizer
# ---------------------------------------------------------------------------
class AdamOptimizer:
    """Adam Optimizer used to update the logits in Bernoulli-ES."""
    def __init__(self, n, step_size, beta1=0.9, beta2=0.999):
        self.n = n
        self.step_size = step_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = np.zeros(n)
        self.v = np.zeros(n)
        self.t = 0

    def update(self, theta, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        # We ADD the gradient because we are MAXIMIZING the reward
        return theta + self.step_size * m_hat / (np.sqrt(v_hat) + 1e-8)


# ---------------------------------------------------------------------------
# Bernoulli-ES (Categorical ES)
# ---------------------------------------------------------------------------
class BernoulliES:
    """
    Bernoulli Evolution Strategy for dynamic subset selection.
    Maintains a logit for each gate, translating to an independent inclusion 
    probability via the sigmoid function. 
    """

    def __init__(self, num_arms, pop_size, learning_rate=0.1):
        self.n = num_arms
        self.pop_size = pop_size
        
        # Initialize logits to 0 (sigmoid(0) = 0.5 -> 50% chance of inclusion)
        self.logits = np.zeros(self.n)
        self.optimizer = AdamOptimizer(self.n, step_size=learning_rate)
        
        # Placeholders for state tracking
        self.p = None
        self.masks = None
        self.generation = 0

    def ask(self):
        """
        Sample lambda candidate subsets using independent Bernoulli trials.
        Returns:
          population_selections : list of dynamically sized gate-index lists
        """
        # 1. Convert logits to probabilities (clamped to avoid overflow)
        clipped_logits = np.clip(self.logits, -10.0, 10.0)
        self.p = 1.0 / (1.0 + np.exp(-clipped_logits))
        
        # 2. Sample binary masks (1 = include gate, 0 = exclude)
        self.masks = np.random.binomial(1, self.p, size=(self.pop_size, self.n))
        
        # 3. Convert binary masks to variable-length lists of selected indices
        population_selections = [
            np.where(mask == 1)[0].tolist()
            for mask in self.masks
        ]
        return population_selections

    def tell(self, rewards):
        """
        Update the logits via Natural Gradient Estimation + Adam.
        Applies a rank transformation to raw rewards for stability.
        """
        rewards = np.array(rewards)
        
        # -- Fitness Shaping (Rank Transformation) --
        ranks = np.zeros_like(rewards)
        ranks[np.argsort(rewards)] = np.arange(len(rewards))
        ranks = ranks / (len(rewards) - 1) - 0.5
        
        # -- REINFORCE Gradient Estimator for Bernoulli --
        # grad = 1/N * sum( R_i * (M_i - p) )
        grad = np.dot(ranks, (self.masks - self.p)) / self.pop_size
        
        # -- Update Logits via Adam --
        self.logits = self.optimizer.update(self.logits, grad)
        self.generation += 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Removed sample_gate constraint from arguments
    genlib_origin = sys.argv[-1]
    lib_origin = genlib_origin[:-7] + '.lib'
    design = sys.argv[-2]

    # -------------------------------------------------------------------------
    # Baseline
    # -------------------------------------------------------------------------
    start = time.time()
    temp_blif_baseline = make_temp_blif(design, "baseline")
    abc_cmd = (
        "read %s;read %s; amap; write %s; read %s;read -m %s; ps; topo; upsize; dnsize; stime; "
        % (genlib_origin, design, temp_blif_baseline, lib_origin, temp_blif_baseline)
    )
    print(abc_cmd)
    res = subprocess.check_output(('wsl', 'abc', '-c', abc_cmd))
    print(res)

    match_d = re.search(r"Delay\s*=\s*([\d.]+)\s*ps", str(res))
    match_a = re.search(r"Area\s*=\s*([\d.]+)", str(res))
    max_delay = float(match_d.group(1))
    max_area = float(match_a.group(1))
    baseline_area = max_area

    print("Baseline Delay:", max_delay)
    print("Baseline Area:", max_area)

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    f_lines, _ = partition_genlib(genlib_origin)
    num_arms = len(f_lines)

    POPULATION_SIZE = 40
    NUM_GENERATIONS = 75
    NUM_WORKERS = min(POPULATION_SIZE, os.cpu_count() or 4)

    # Instantiate Bernoulli ES (No K-gate constraint)
    es = BernoulliES(num_arms, pop_size=POPULATION_SIZE, learning_rate=0.1)

    best_cells = None
    best_result = None
    best_reward = -float('inf')
    episode_areas = []

    print(f"\n>> Starting Bernoulli-ES Area Optimization (parallel, {NUM_WORKERS} workers)")
    print(f"   Dynamic Subset Selection (No cardinality constraint)")
    print(f"   {NUM_GENERATIONS} generations × {POPULATION_SIZE} population = "
          f"{NUM_GENERATIONS * POPULATION_SIZE} total evaluations\n")
    print(f"   Bernoulli-ES: n={num_arms}, lambda={es.pop_size}")

    # -------------------------------------------------------------------------
    # Main loop (Optimized Parallelism)
    # -------------------------------------------------------------------------
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for generation in range(NUM_GENERATIONS):
            print(f"Generation: {generation}")

            population_cells = es.ask()

            worker_args = [
                (i, genlib_origin, lib_origin, design, selected_cells, lib_path)
                for i, selected_cells in enumerate(population_cells)
            ]

            results_ordered = [None] * POPULATION_SIZE
            
            future_to_idx = {
                executor.submit(technology_mapper, args): args[0]
                for args in worker_args
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results_ordered[idx] = future.result()
                except Exception as exc:
                    print(f"  Worker {idx} raised an exception: {exc}")
                    results_ordered[idx] = (float("nan"), float("nan"))

            # Collect rewards; handle NaN results (failed ABC mapping)
            population_rewards = []
            generation_areas = []

            for i, (delay, area) in enumerate(results_ordered):
                # An empty or functionally incomplete subset will return NaNs
                if results_ordered[i] is None or np.isnan(delay) or np.isnan(area):
                    reward = -2.0  # Heavy penalty for invalid libraries
                    result_area = float('inf')
                else:
                    reward = calculate_reward(max_area, area)
                    result_area = area

                population_rewards.append(reward)
                generation_areas.append(result_area)

                if reward > best_reward:
                    best_reward = reward
                    best_result = (delay, area)
                    best_cells = population_cells[i]
                    # Also print how many gates it chose this time
                    print(f"  [New Best] Gen {generation} Worker {i} | Size: {len(best_cells)} gates | Reward: {best_reward:.4f} | Area: {area:.4f}")

            episode_areas.append(generation_areas)

            # Update logits
            es.tell(population_rewards)

    end = time.time()
    runtime = end - start

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------
    print("\n--- Optimization Complete ---")
    if best_cells is None:
        print("WARNING: No valid mapping was found across all evaluations.")
    else:
        print(f"Best Subset Size: {len(best_cells)} / {num_arms} arms")
        print("Best Cells:", best_cells)
        print("Best Delay:", best_result[0])
        print("Best Area:", best_result[1])
        print("Best Reward:", best_reward)
    print("Total time:", runtime)

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    print("\n>> Generating Area Optimization Plot...")
    plt.figure(figsize=(10, 6))

    total_evaluations = NUM_GENERATIONS * POPULATION_SIZE

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
        ax1.scatter(scatter_x, scatter_y, color='red', alpha=0.4,
                    label=f'Population samples (n={POPULATION_SIZE})', s=25, zorder=2)

    if best_line_x:
        ax1.step(best_line_x, best_line_y, color='blue', linewidth=2.5,
                 where='post', label='Best area so far', zorder=3)
        ax1.scatter(best_line_x, best_line_y, color='blue', s=30, zorder=4)

    ax1.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Baseline Area')
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Area (normalized)")

    xtick_step = max(1, NUM_GENERATIONS // 20)
    ax1.set_xticks(range(0, NUM_GENERATIONS, xtick_step))

    plt.title(
        f"Bernoulli-ES Area Optimization (Dynamic Subset)\n"
        f"Parallel ({NUM_WORKERS} workers) | Design: {design.split('/')[-1]}"
    )
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
    # Removed sample_gate from the output file name
    output_path = (
        f"areatest/bernoullies_{total_evaluations}_"
        f"{design.split('/')[-1].split('.')[0]}_"
        f"{lib_origin.split('/')[-1][:-4]}.png"
    )
    plt.savefig(output_path, dpi=300)
    print(f">> Visualization successfully saved to {output_path}")