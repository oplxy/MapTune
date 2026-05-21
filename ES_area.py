import random
import sys
import os 
import numpy as np
import subprocess
from subprocess import PIPE
import re
import time
import matplotlib.pyplot as plt
import concurrent.futures

# --- 1. SETUP AND PARSING ---
genlib_origin = sys.argv[-1]
lib_origin = genlib_origin[:-7] + '.lib'
design = sys.argv[-2]
sample_gate = int(sys.argv[-3])
lib_path = "gen_newlibs/"

# Seed for reproducibility
np.random.seed(int(time.time()))
random.seed(time.time())
start = time.time()

os.makedirs("temp_blifs", exist_ok=True)
os.makedirs("gen_newlibs", exist_ok=True)
os.makedirs("newtest", exist_ok=True)

# Baseline Evaluation
baseline_temp_blif = f"temp_blifs/{design.split('/')[-1][:-5]}_ep_temp_baseline.blif"
abc_cmd = "read %s;read %s; amap; write %s; read %s;read -m %s; ps; topo; upsize; dnsize; stime; " % (genlib_origin, design, baseline_temp_blif, lib_origin, baseline_temp_blif)
print("Evaluating Baseline...")
try:
    res = subprocess.check_output(('wsl', 'abc', '-c', abc_cmd))
    match_d = re.search(r"Delay\s*=\s*([\d.]+)\s*ps", str(res))
    match_a = re.search(r"Area\s*=\s*([\d.]+)", str(res))
    max_delay = float(match_d.group(1))
    max_area = float(match_a.group(1))
except Exception as e:
    print(f"Error getting baseline: {e}")
    sys.exit(1)

print(f"Baseline Delay: {max_delay:.2f} | Baseline Area: {max_area:.2f}")

# --- 2. THREAD-SAFE TECHNOLOGY MAPPER ---
def technology_mapper(genlib_origin, partial_cell_library, worker_id):
    safe_design_name = design.replace('/', '_').replace('.', '_')
    unique_temp_blif = f"temp_blifs/{safe_design_name}_ep_temp_{worker_id}.blif"
    unique_genlib_file = f"{lib_path}{safe_design_name}_{len(partial_cell_library)}_ep_samplelib_{worker_id}.genlib"

    with open(genlib_origin, 'r') as f:
        f_lines = [line.strip() for line in f if line.startswith("GATE") and not line.startswith("GATE BUF") and not line.startswith("GATE INV") and not line.startswith("GATE sky130_fd_sc_hd__buf") and not line.startswith("GATE sky130_fd_sc_hd__inv") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv")]
    with open(genlib_origin, 'r') as f:
        f_keep = [line.strip() for line in f if line.startswith("GATE BUF") or line.startswith("GATE INV") or line.startswith("GATE sky130_fd_sc_hd__buf") or line.startswith("GATE sky130_fd_sc_hd__inv") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv")]
    
    lines_partial = [f_lines[i] for i in partial_cell_library]
    lines_partial = lines_partial + f_keep

    with open(unique_genlib_file, 'w') as out_gen:
        for line in lines_partial:
            out_gen.write(line + '\n')

    abc_cmd = "read %s;read %s; amap; write %s; read %s;read -m %s; ps; topo; upsize; dnsize; stime; " % (unique_genlib_file, design, unique_temp_blif, lib_origin, unique_temp_blif)
    
    try:
        res = subprocess.check_output(('wsl', 'abc', '-c', abc_cmd))
        match_d = re.search(r"Delay\s*=\s*([\d.]+)\s*ps", str(res))
        match_a = re.search(r"Area\s*=\s*([\d.]+)", str(res))
        if match_d and match_a:
            delay = float(match_d.group(1))
            area = float(match_a.group(1))
        else:
            delay, area = float("inf"), float("inf")
    except subprocess.CalledProcessError:
        delay, area = float("inf"), float("inf")
        
    return delay, area

# --- 3. EVOLUTION STRATEGIES (ES) SETUP ---
with open(genlib_origin, 'r') as f:
    f_lines = [line.strip() for line in f if line.startswith("GATE") and not line.startswith("GATE BUF") and not line.startswith("GATE INV") and not line.startswith("GATE sky130_fd_sc_hd__buf") and not line.startswith("GATE sky130_fd_sc_hd__inv") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv")]
num_arms = len(f_lines)

# ES Hyperparameters
POPULATION_SIZE = 20       # Number of parallel workers per generation
ITERATIONS = 150           # Total update steps
LEARNING_RATE = 0.1        # Step size for the weight update (Alpha)
SIGMA = 0.5                # Standard deviation of the Gaussian noise

print(f"\n--- Starting Evolution Strategies (ES) Optimization ---")
print(f"Workers: {POPULATION_SIZE} | Iterations: {ITERATIONS} | Alpha: {LEARNING_RATE} | Sigma: {SIGMA}")

# Initialize the base weights (Logits) to zeros
weights = np.zeros(num_arms)

best_global_area = float('inf')
best_global_delay = float('inf')
best_global_state = None

# Tracking for plots
best_area_history = []
iteration_x = []
scatter_x = []
scatter_y = []

# --- 4. MAIN ES LOOP ---
for iteration in range(ITERATIONS):
    print(f"\n[Iteration {iteration+1}/{ITERATIONS}] Sampling and evaluating...")
    
    # Generate Gaussian Noise for the entire population
    # Shape: (POPULATION_SIZE, num_arms)
    noise = np.random.randn(POPULATION_SIZE, num_arms)
    
    # Apply noise to base weights to create population policies
    noisy_weights = weights + (SIGMA * noise)
    
    # Convert continuous weights to discrete actions (Pick top K indices)
    # np.argsort(-x) sorts descending, so we grab the indices with the highest noisy weight
    population_actions = [np.argsort(-noisy_weights[i])[:sample_gate].tolist() for i in range(POPULATION_SIZE)]
    
    areas = np.zeros(POPULATION_SIZE)
    delays = np.zeros(POPULATION_SIZE)
    
    # PARALLEL EVALUATION
    with concurrent.futures.ThreadPoolExecutor(max_workers=POPULATION_SIZE) as executor:
        futures = {
            executor.submit(technology_mapper, genlib_origin, action, idx): (idx, action) 
            for idx, action in enumerate(population_actions)
        }
        
        for future in concurrent.futures.as_completed(futures):
            idx, action = futures[future]
            try:
                delay, area = future.result() 
                
                # Handle failed synthesis
                if area == float('inf') or np.isnan(area):
                    areas[idx] = max_area * 5 
                    delays[idx] = float('inf')
                else:
                    areas[idx] = area
                    delays[idx] = delay
                    
                if areas[idx] < max_area * 2:
                    scatter_x.append(iteration)
                    scatter_y.append(areas[idx])
                    
            except Exception as exc:
                print(f"Worker {idx} generated an exception: {exc}")
                areas[idx] = max_area * 5
                delays[idx] = float('inf')
                
    # Track Best Found
    min_idx = np.argmin(areas)
    iter_best_area = areas[min_idx]
    
    if iter_best_area < best_global_area:
        best_global_area = iter_best_area
        best_global_delay = delays[min_idx]
        best_global_state = list(population_actions[min_idx])
        print(f"   >>> NEW GLOBAL BEST: Area = {best_global_area:.2f}")
        
    best_area_history.append(best_global_area)
    iteration_x.append(iteration)
    
    print(f"Iteration {iteration+1} Best: {iter_best_area:.2f} | Global Best: {best_global_area:.2f}")

    # --- 5. GRADIENT-FREE WEIGHT UPDATE ---
    # We want to MINIMIZE area, so smaller area = higher reward
    rewards = -areas 
    
    # Standardize rewards (Mean=0, Std=1) to stabilize the gradient step
    # This prevents wild area fluctuations from blowing up the weight vector
    rewards_std = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
    
    # The ES Update Rule:
    # weights = weights + (learning_rate / (N * sigma)) * dot(noise^T, rewards)
    gradient_estimator = np.dot(noise.T, rewards_std)
    weights += LEARNING_RATE / (POPULATION_SIZE * SIGMA) * gradient_estimator

# --- 6. RESULTS & PLOTTING ---
end = time.time()
runtime = end - start

print("\n--- OPTIMIZATION COMPLETE ---")
print("Best Cells (Indices):", best_global_state)
print("Best Delay:", best_global_delay)
print("Best Area:", best_global_area)
print("Total time:", runtime)

print("\n>> Generating ES Plot...")
plt.figure(figsize=(10, 6))

clean_best_area = [val/max_area for val in best_area_history]
norm_scatter_y = [val/max_area for val in scatter_y]

plt.plot(iteration_x, clean_best_area, label='Global Best Area', color='blue', linewidth=2.5, marker='o')
plt.scatter(scatter_x, norm_scatter_y, color='red', alpha=0.3, label='Population Samples', s=15)
plt.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Baseline Area (1.0)')

plt.title(f"Evolution Strategies Area Optimization\nDesign: {design.split('/')[-1] if '/' in design else design} | Workers: {POPULATION_SIZE}")
plt.xlabel("Iterations")
plt.ylabel("Normalized Area (vs Baseline)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

plt.text(0.75, 0.98, f'Training Time: {runtime:.2f}s\nBest Area: {clean_best_area[-1]:.3f}\nEvals: {POPULATION_SIZE * ITERATIONS}', 
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

safe_design_name = design.split('/')[1].split('.')[0]
safe_lib_name = lib_origin[:-4]
output_path = f"areatest/es_{ITERATIONS}_{safe_design_name}_{sample_gate}_{safe_lib_name}_area.png"
plt.savefig(output_path, dpi=300)
print(f">> Visualization successfully saved to {output_path}")