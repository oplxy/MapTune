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
# CMA-ES
# ---------------------------------------------------------------------------
class CMAES:
    """
    (mu/mu_w, lambda)-CMA-ES for top-K discrete gate selection.

    Notation follows the Hansen tutorial (arxiv 1604.00772):
      n          : number of dimensions (= num_arms, one score per gate)
      lambda_    : population size (number of samples per generation)
      mu         : number of elite samples used to update the distribution
      mean       : current distribution mean (replaces 'theta' in OpenAI-ES)
      sigma      : global step size (scalar), adapted via CSA
      C          : covariance matrix (n×n), adapted via rank-1 + rank-mu
      p_sigma    : evolution path for sigma (CSA)
      p_c        : evolution path for C (rank-1 update)

    Gate selection: top-K indices of the sampled continuous vector.
    CMA-ES minimises cost, so rewards are negated costs passed in from outside.
    Internally we work with costs = -rewards so the algorithm minimises them.
    """

    def __init__(self, num_arms, sample_gate, pop_size):
        self.n = num_arms
        self.sample_gate = sample_gate
        self.lam = pop_size                     # lambda: total samples

        # --- Elite set size and recombination weights ---
        self.mu = max(1, pop_size // 2)         # use top half as elites
        raw_w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = raw_w / raw_w.sum()      # positive, sum to 1
        self.mu_eff = 1.0 / (self.weights ** 2).sum()   # variance-effective mu

        # --- Step-size control (CSA) ---
        self.cs = (self.mu_eff + 2) / (self.n + self.mu_eff + 5)
        self.ds = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.n + 1)) - 1) + self.cs
        # E[||N(0,I)||] — expected length of a standard Gaussian vector
        self.chi_n = np.sqrt(self.n) * (1 - 1 / (4 * self.n) + 1 / (21 * self.n ** 2))

        # --- Covariance matrix control ---
        self.cc = (4 + self.mu_eff / self.n) / (self.n + 4 + 2 * self.mu_eff / self.n)
        self.c1 = 2 / ((self.n + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(
            1 - self.c1,
            2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.n + 2) ** 2 + self.mu_eff)
        )

        # --- State ---
        self.mean = np.zeros(self.n)            # distribution mean
        self.sigma = 0.3                        # initial global step size
        self.C = np.eye(self.n)                 # covariance matrix
        self.p_sigma = np.zeros(self.n)         # CSA evolution path
        self.p_c = np.zeros(self.n)             # rank-1 evolution path
        self.eigeneval = 0                      # tracks when to re-decompose C
        self.generation = 0

        # Cached decomposition: C = B * diag(D^2) * B^T
        self.B = np.eye(self.n)
        self.D = np.ones(self.n)

        # hsig threshold constant
        self._hsig_coeff = (1.4 + 2 / (self.n + 1)) * self.chi_n

    def _update_eigdecomp(self):
        """Recompute B and D from C. Called lazily every n/(c1+cmu)/10 evals."""
        self.C = np.triu(self.C) + np.triu(self.C, 1).T   # enforce symmetry
        eigvals, self.B = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-20)               # clamp negatives
        self.D = np.sqrt(eigvals)

    def ask(self):
        """
        Sample lambda candidate solutions.
        Returns:
          population_selections : list of lambda gate-index lists (discrete)
          samples               : (lambda, n) array of continuous samples
                                  (needed by tell())
        """
        # Lazy eigendecomposition — only recompute every n/(c1+cmu)/10 steps
        if self.generation - self.eigeneval > self.n / ((self.c1 + self.cmu) * self.n * 10):
            self._update_eigdecomp()
            self.eigeneval = self.generation

        # Sample: x = mean + sigma * B * D * z,  z ~ N(0, I)
        z = np.random.randn(self.lam, self.n)               # (lam, n)
        # BD maps unit Gaussian to the current covariance ellipse
        BD = self.B * self.D[np.newaxis, :]                 # (n, n) broadcast
        samples = self.mean + self.sigma * (z @ BD.T)       # (lam, n)

        # Discrete gate selection: top-K by continuous score
        population_selections = [
            np.argsort(samples[i])[-self.sample_gate:].tolist()
            for i in range(self.lam)
        ]
        return population_selections, samples

    def tell(self, samples, costs):
        """
        Update mean, sigma, and C given the sampled continuous vectors and
        their associated costs (lower = better; pass in -reward).

        Args:
          samples : (lambda, n) array returned by ask()
          costs   : length-lambda array of scalar costs
        """
        costs = np.array(costs)

        # --- Rank samples by cost (ascending = best first) ---
        ranked_idx = np.argsort(costs)
        elite_samples = samples[ranked_idx[:self.mu]]       # (mu, n)

        # --- Update mean ---
        old_mean = self.mean.copy()
        self.mean = self.weights @ elite_samples            # weighted centroid

        # Step in mean space, normalised by sigma (used for path updates)
        mean_step = (self.mean - old_mean) / self.sigma     # (n,)

        # --- CSA: update evolution path p_sigma ---
        # C^{-1/2} * mean_step = B * (1/D) * B^T * mean_step
        invsqrtC_step = self.B @ ((1.0 / self.D) * (self.B.T @ mean_step))
        self.p_sigma = (
            (1 - self.cs) * self.p_sigma
            + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * invsqrtC_step
        )

        # --- Heaviside indicator (hsig): suppress rank-1 update if sigma is
        #     increasing too fast, to avoid premature covariance growth ---
        hsig = (
            np.linalg.norm(self.p_sigma) / np.sqrt(1 - (1 - self.cs) ** (2 * (self.generation + 1)))
            < self._hsig_coeff
        )

        # --- Rank-1 evolution path p_c ---
        self.p_c = (
            (1 - self.cc) * self.p_c
            + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * mean_step
        )

        # --- Covariance matrix update ---
        # Rank-1 term
        rank1 = np.outer(self.p_c, self.p_c)

        # Rank-mu term: weighted sum of outer products of elite steps
        elite_steps = (elite_samples - old_mean) / self.sigma  # (mu, n)
        rank_mu = sum(
            self.weights[k] * np.outer(elite_steps[k], elite_steps[k])
            for k in range(self.mu)
        )

        self.C = (
            (1 - self.c1 - self.cmu) * self.C
            + self.c1 * (rank1 + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
            + self.cmu * rank_mu
        )

        # --- Step-size adaptation (CSA) ---
        self.sigma *= np.exp(
            (self.cs / self.ds) * (np.linalg.norm(self.p_sigma) / self.chi_n - 1)
        )

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

    es = CMAES(num_arms, sample_gate, pop_size=POPULATION_SIZE)

    best_cells = None
    best_result = None
    best_reward = -float('inf')
    episode_areas = []   # episode_areas[g] = list of area values for generation g

    print(f"\n>> Starting CMA-ES Area Optimization (parallel, {NUM_WORKERS} workers)")
    print(f"   {NUM_GENERATIONS} generations × {POPULATION_SIZE} population = "
          f"{NUM_GENERATIONS * POPULATION_SIZE} total evaluations\n")
    print(f"   CMA-ES: n={num_arms}, mu={es.mu}, lambda={es.lam}")
    print(f"   cs={es.cs:.4f}, ds={es.ds:.4f}, cc={es.cc:.4f}, c1={es.c1:.6f}, cmu={es.cmu:.6f}\n")

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------
    for generation in range(NUM_GENERATIONS):
        print(f"Generation: {generation}  |  sigma={es.sigma:.4f}")

        population_cells, samples = es.ask()

        worker_args = [
            (i, genlib_origin, lib_origin, design, selected_cells, lib_path)
            for i, selected_cells in enumerate(population_cells)
        ]

        results_ordered = [None] * POPULATION_SIZE
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
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

        # Collect rewards and costs; handle NaN results
        population_rewards = []
        population_costs = []
        generation_areas = []

        for i, (delay, area) in enumerate(results_ordered):
            if results_ordered[i] is None or np.isnan(delay) or np.isnan(area):
                reward = -2.0
                # Penalise failed runs with a cost worse than anything real.
                # We use 2.0 because rewards are in [-1, 0] for valid results,
                # so -(-2.0) = 2.0 is safely worse than any normalised area.
                cost = 2.0
                result_area = float('inf')
            else:
                reward = calculate_reward(max_area, area)
                cost = -reward                  # CMA-ES minimises cost
                result_area = area

            population_rewards.append(reward)
            population_costs.append(cost)
            generation_areas.append(result_area)

            if reward > best_reward:
                best_reward = reward
                best_result = (delay, area)
                best_cells = population_cells[i]
                print(f"  [New Best] Gen {generation} Worker {i} | Reward: {best_reward:.4f}")

        episode_areas.append(generation_areas)

        # CMA-ES update: pass the continuous samples and their costs
        es.tell(samples, population_costs)

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
    sigma_x, sigma_y = [], []       # track sigma evolution on a twin axis
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
        f"CMA-ES Area Optimization (Parallel)\n"
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
        f"areatest/cmaes_{total_evaluations}_"
        f"{design.split('/')[-1].split('.')[0]}_{sample_gate}_"
        f"{lib_origin.split('/')[-1][:-4]}.png"
    )
    plt.savefig(output_path, dpi=300)
    print(f">> Visualization successfully saved to {output_path}")