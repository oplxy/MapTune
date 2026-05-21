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
    """Adam Optimizer used to update the mean in OpenAI-ES."""
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
        # Note: We ADD the gradient because we are MAXIMIZING the reward
        return theta + self.step_size * m_hat / (np.sqrt(v_hat) + 1e-8)


# ---------------------------------------------------------------------------
# OpenAI-ES
# ---------------------------------------------------------------------------
class OpenAIES:
    """
    OpenAI Evolution Strategies (OpenAI-ES) for top-K discrete gate selection.
    Uses Antithetic (Mirrored) Sampling, Fitness Shaping (Rank Transformation),
    and the Adam optimizer to robustly navigate the search space.
    """

    def __init__(self, num_arms, sample_gate, pop_size, sigma=0.1, learning_rate=0.05):
        self.n = num_arms
        self.sample_gate = sample_gate
        self.pop_size = pop_size
        self.sigma = sigma
        
        self.mean = np.zeros(self.n)            # distribution mean
        self.optimizer = AdamOptimizer(self.n, step_size=learning_rate)
        
        # Placeholders for state tracking
        self.full_noise = None
        self.generation = 0

    def ask(self):
        """
        Sample lambda candidate solutions using mirrored sampling.
        Returns:
          population_selections : list of lambda gate-index lists (discrete)
          samples               : (lambda, n) array of continuous samples
        """
        # Mirrored sampling: Generate half the noise vectors, use both +noise and -noise
        half_pop = self.pop_size // 2
        noise = np.random.randn(half_pop, self.n)
        
        if self.pop_size % 2 == 0:
            self.full_noise = np.concatenate([noise, -noise], axis=0)
        else:
            extra = np.random.randn(1, self.n)
            self.full_noise = np.concatenate([noise, -noise, extra], axis=0)
            
        samples = self.mean + self.sigma * self.full_noise

        # Discrete gate selection: top-K by continuous score
        population_selections = [
            np.argsort(samples[i])[-self.sample_gate:].tolist()
            for i in range(self.pop_size)
        ]
        return population_selections, samples

    def tell(self, samples, rewards):
        """
        Update the mean parameter via Natural Gradient Estimation + Adam.
        Applies a rank transformation to raw rewards for stability.
        """
        rewards = np.array(rewards)
        
        # -- Fitness Shaping (Rank Transformation) --
        # Ranks the rewards, then scales to [-0.5, 0.5]
        # Higher reward -> higher rank -> pushes mean in the direction of the noise
        ranks = np.zeros_like(rewards)
        ranks[np.argsort(rewards)] = np.arange(len(rewards))
        ranks = ranks / (len(rewards) - 1) - 0.5
        
        # -- Gradient Estimator --
        # g = 1 / (N * sigma) * sum(F_i * epsilon_i)
        grad = (1.0 / (self.pop_size * self.sigma)) * np.dot(self.full_noise.T, ranks)
        
        # -- Update Mean via Adam --
        self.mean = self.optimizer.update(self.mean, grad)
        self.generation += 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    genlib_origin = sys.argv[-1]
    lib_origin = genlib_origin[:-7] + '.lib'
    design = sys.argv[-2]
    sample_gate = int(sys.argv[-3])

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

    # Note: You can tune sigma and learning_rate for your specific landscape
    es = OpenAIES(num_arms, sample_gate, pop_size=POPULATION_SIZE, sigma=0.1, learning_rate=0.05)

    best_cells = None
    best_result = None
    best_reward = -float('inf')
    episode_areas = []

    print(f"\n>> Starting OpenAI-ES Area Optimization (parallel, {NUM_WORKERS} workers)")
    print(f"   {NUM_GENERATIONS} generations × {POPULATION_SIZE} population = "
          f"{NUM_GENERATIONS * POPULATION_SIZE} total evaluations\n")
    print(f"   OpenAI-ES: n={num_arms}, lambda={es.pop_size}, sigma={es.sigma}")

    # -------------------------------------------------------------------------
    # Main loop (Optimized Parallelism)
    # -------------------------------------------------------------------------
    
    # We open the ProcessPoolExecutor OUTSIDE the generation loop!
    # This prevents spinning up and tearing down OS processes every generation.
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for generation in range(NUM_GENERATIONS):
            print(f"Generation: {generation}")

            population_cells, samples = es.ask()

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

            # Collect rewards; handle NaN results
            population_rewards = []
            generation_areas = []

            for i, (delay, area) in enumerate(results_ordered):
                if results_ordered[i] is None or np.isnan(delay) or np.isnan(area):
                    reward = -2.0  # Penalise failed runs
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
                    print(f"  [New Best] Gen {generation} Worker {i} | Reward: {best_reward:.4f} | Area: {area:.4f}")

            episode_areas.append(generation_areas)

            # OpenAI-ES update: tell the optimizer the actual raw rewards
            # (it handles its own Rank Transformation/Fitness Shaping internally)
            es.tell(samples, population_rewards)

    end = time.time()
    runtime = end - start

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------
    print("\n--- Optimization Complete ---")
    if best_cells is None:
        print("WARNING: No valid mapping was found across all evaluations.")
        print("  All ABC calls either failed or returned NaN delay/area.")
        print("  Check your .genlib / .blif files and ABC installation.")
    else:
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
        f"OpenAI-ES Area Optimization (Parallel)\n"
        f"Top-K Selection | {NUM_WORKERS} workers | Design: {design.split('/')[-1]}"
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
    output_path = (
        f"areatest/openaies_{total_evaluations}_"
        f"{design.split('/')[-1].split('.')[0]}_{sample_gate}_"
        f"{lib_origin.split('/')[-1][:-4]}.png"
    )
    plt.savefig(output_path, dpi=300)
    print(f">> Visualization successfully saved to {output_path}")