import random
import sys
import os 
import numpy as np
import subprocess
from subprocess import PIPE
import re
import time
import math
import matplotlib.pyplot as plt
import concurrent.futures

# --- 1. SETUP AND PARSING ---
genlib_origin = sys.argv[-1]
lib_origin = genlib_origin[:-7] + '.lib'
design = sys.argv[-2]
sample_gate = int(sys.argv[-3])
lib_path = "gen_newlibs/"

random.seed(time.time())
start = time.time()

# Ensure directories exist
os.makedirs("temp_blifs", exist_ok=True)
os.makedirs("gen_newlibs", exist_ok=True)
os.makedirs("newtest", exist_ok=True)

# Baseline Evaluation (Runs synchronously once)
baseline_temp_blif = f"temp_blifs/{design.split('/')[-1][:-5]}_ep_temp_baseline.blif"
abc_cmd = "read %s;read %s; amap; write %s; read %s;read -m %s; ps; topo; upsize; dnsize; stime; " % (genlib_origin, design, baseline_temp_blif, lib_origin, baseline_temp_blif)
print(abc_cmd)
try:
    res = subprocess.check_output(('wsl', 'abc', '-c', abc_cmd))
    print(res)
    match_d = re.search(r"Delay\s*=\s*([\d.]+)\s*ps", str(res))
    match_a = re.search(r"Area\s*=\s*([\d.]+)", str(res))
    max_delay = float(match_d.group(1))
    max_area = float(match_a.group(1))
except Exception as e:
    print(f"Error getting baseline: {e}")
    sys.exit(1)

print("\nBaseline Delay:", max_delay)
print("Baseline Area:", max_area)

# --- 2. THREAD-SAFE TECHNOLOGY MAPPER ---
def technology_mapper(genlib_origin, partial_cell_library, worker_id):
    """
    Evaluates a specific subset of gates. 
    Uses worker_id to ensure thread-safe file I/O during parallel execution.
    """
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

# --- 3. GENETIC ALGORITHM SETUP ---
with open(genlib_origin, 'r') as f:
    f_lines = [line.strip() for line in f if line.startswith("GATE") and not line.startswith("GATE BUF") and not line.startswith("GATE INV") and not line.startswith("GATE sky130_fd_sc_hd__buf") and not line.startswith("GATE sky130_fd_sc_hd__inv") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv")]
num_arms = len(f_lines)

# Hyperparameters
POPULATION_SIZE = 10    
GENERATIONS = 100      
MUTATION_RATE = 0.25    
ELITISM = 2             

print(f"\n--- Starting Parallel Genetic Algorithm Optimization ---")
print(f"Population: {POPULATION_SIZE} | Generations: {GENERATIONS} | Library Size: {sample_gate} gates")

population = [random.sample(range(num_arms), sample_gate) for _ in range(POPULATION_SIZE)]

best_global_area = float('inf')
best_global_delay = float('inf')
best_global_state = None

best_area_history = []
generation_x = []
scatter_x = []
scatter_y = []

# --- 4. MAIN EVOLUTION LOOP ---
for gen in range(GENERATIONS):
    print(f"\n[Generation {gen+1}/{GENERATIONS}] Evaluating population in parallel...")
    
    fitness_scores = []
    
    # PARALLEL EVALUATION
    with concurrent.futures.ThreadPoolExecutor(max_workers=POPULATION_SIZE) as executor:
        futures = {
            executor.submit(technology_mapper, genlib_origin, individual, idx): individual 
            for idx, individual in enumerate(population)
        }
        
        for future in concurrent.futures.as_completed(futures):
            individual = futures[future]
            try:
                delay, area = future.result() 
                
                # Penalize failed synthesis
                if area == float('inf') or np.isnan(area):
                    eval_area = max_area * 5 
                else:
                    eval_area = area
                    
                fitness_scores.append((eval_area, individual, delay))
                
                if eval_area < max_area * 2:
                    scatter_x.append(gen)
                    scatter_y.append(eval_area)
                    
            except Exception as exc:
                print(f"Individual evaluation generated an exception: {exc}")
                fitness_scores.append((max_area * 5, individual, float('inf')))
                
    # Sort population from best (lowest area) to worst
    fitness_scores.sort(key=lambda x: x[0])
    
    gen_best_area = fitness_scores[0][0]
    gen_best_state = fitness_scores[0][1]
    gen_best_delay = fitness_scores[0][2]
    
    if gen_best_area < best_global_area:
        best_global_area = gen_best_area
        best_global_delay = gen_best_delay
        best_global_state = list(gen_best_state)
        print(f"   >>> NEW GLOBAL BEST: Area = {best_global_area:.2f} (Delay = {best_global_delay:.2f})")
        
    best_area_history.append(best_global_area)
    generation_x.append(gen)
    
    print(f"Generation {gen+1} Best: {gen_best_area:.2f} | Global Best: {best_global_area:.2f}")

    if gen == GENERATIONS - 1:
        break

    # --- 5. BREEDING & SELECTION ---
    next_generation = []
    
    for i in range(ELITISM):
        next_generation.append(list(fitness_scores[i][1]))
        
    while len(next_generation) < POPULATION_SIZE:
        tournament1 = random.sample(fitness_scores, 3)
        parent1 = min(tournament1, key=lambda x: x[0])[1]
        
        tournament2 = random.sample(fitness_scores, 3)
        parent2 = min(tournament2, key=lambda x: x[0])[1]
        
        split_point = sample_gate // 2
        child = set(parent1[:split_point] + parent2[split_point:])
        
        while len(child) < sample_gate:
            new_gate = random.randint(0, num_arms - 1)
            child.add(new_gate)
            
        child = list(child)
        
        if random.random() < MUTATION_RATE:
            gate_to_remove = random.choice(child)
            child.remove(gate_to_remove)
            while len(child) < sample_gate:
                new_gate = random.randint(0, num_arms - 1)
                if new_gate not in child:
                    child.append(new_gate)
                    
        next_generation.append(child)
        
    population = next_generation

# --- 6. RESULTS & PLOTTING ---
end = time.time()
runtime = end - start

print("\n--- OPTIMIZATION COMPLETE ---")
print("Best Cells (Indices):", best_global_state)
print("Best Delay:", best_global_delay)
print("Best Area:", best_global_area)
print("Total time:", runtime)

print("\n>> Generating Evolutionary Plot...")
plt.figure(figsize=(10, 6))

clean_best_area = [val/max_area for val in best_area_history]
norm_scatter_y = [val/max_area for val in scatter_y]

plt.plot(generation_x, clean_best_area, label='Global Best Area', color='blue', linewidth=2.5, marker='o')
plt.scatter(scatter_x, norm_scatter_y, color='red', alpha=0.3, label='Population Samples', s=15)
plt.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Baseline Area (1.0)')

plt.title(f"Parallel Genetic Algorithm Area Optimization\nDesign: {design.split('/')[-1] if '/' in design else design} | Pop: {POPULATION_SIZE}")
plt.xlabel("Generations")
plt.ylabel("Normalized Area (vs Baseline)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

plt.text(0.75, 0.98, f'Training Time: {runtime:.2f}s\nBest Area: {clean_best_area[-1]:.3f}\nEvals: {POPULATION_SIZE * GENERATIONS}', 
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

safe_design_name = design.replace('/', '_').replace('.', '_')
safe_lib_name = lib_origin.replace('/', '_').replace('.', '_')
output_path = f"areatest/ga_{GENERATIONS}gen_{safe_design_name}_{sample_gate}_{safe_lib_name}_area.png"
plt.savefig(output_path, dpi=300)
print(f">> Visualization successfully saved to {output_path}")