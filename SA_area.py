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

genlib_origin = sys.argv[-1]
lib_origin = genlib_origin[:-7] + '.lib'
design = sys.argv[-2]
sample_gate = int(sys.argv[-3])
temp_blif = "temp_blifs/" + design[:-5] + "_ep_temp.blif"
lib_path = "gen_newlibs/"

random.seed(time.time())
start = time.time()

# Ensure directories exist
os.makedirs("temp_blifs", exist_ok=True)
os.makedirs("gen_newlibs", exist_ok=True)
os.makedirs("newtest", exist_ok=True)

abc_cmd = "read %s;read %s; amap; write %s; read %s;read -m %s; ps; topo; upsize; dnsize; stime; " % (genlib_origin, design, temp_blif, lib_origin, temp_blif)
print(abc_cmd)
res = subprocess.check_output(('wsl', 'abc', '-c', abc_cmd))
print(res)
match_d = re.search(r"Delay\s*=\s*([\d.]+)\s*ps", str(res))
match_a = re.search(r"Area\s*=\s*([\d.]+)", str(res))

# Baseline
max_delay = float(match_d.group(1))
max_area = float(match_a.group(1))

print("Baseline Delay:", max_delay)
print("Baseline Area:", max_area)

# Mapper call
def technology_mapper(genlib_origin, partial_cell_library):
    with open(genlib_origin, 'r') as f:
        f_lines = [line.strip() for line in f if line.startswith("GATE") and not line.startswith("GATE BUF") and not line.startswith("GATE INV") and not line.startswith("GATE sky130_fd_sc_hd__buf") and not line.startswith("GATE sky130_fd_sc_hd__inv") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv")]
    with open(genlib_origin, 'r') as f:
        f_keep = [line.strip() for line in f if line.startswith("GATE BUF") or line.startswith("GATE INV") or line.startswith("GATE sky130_fd_sc_hd__buf") or line.startswith("GATE sky130_fd_sc_hd__inv") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv")]
    
    lines_partial = [f_lines[i] for i in partial_cell_library]
    lines_partial = lines_partial + f_keep

    output_genlib_file = lib_path + design.replace('/', '_') + "_" + str(len(lines_partial)) + "_ep_samplelib.genlib"
    with open(output_genlib_file, 'w') as out_gen:
        for line in lines_partial:
            out_gen.write(line + '\n')

    abc_cmd = "read %s;read %s; amap; write %s; read %s;read -m %s; ps; topo; upsize; dnsize; stime; " % (output_genlib_file, design, temp_blif, lib_origin, temp_blif)
    
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

# Initialization
num_cells_select = sample_gate
with open(genlib_origin, 'r') as f:
    f_lines = [line.strip() for line in f if line.startswith("GATE") and not line.startswith("GATE BUF") and not line.startswith("GATE INV") and not line.startswith("GATE sky130_fd_sc_hd__buf") and not line.startswith("GATE sky130_fd_sc_hd__inv") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv")]

num_arms = len(f_lines)

# --- SIMULATED ANNEALING SETUP ---
num_iterations = 1000
# Initial state: Randomly pick `sample_gate` number of unique gates
current_state = random.sample(range(num_arms), sample_gate)
current_delay, current_area = technology_mapper(genlib_origin, current_state)

# If the random start is invalid, penalize it so it forces a change
if current_area == float('inf'):
    current_area = max_area * 2 

best_state = list(current_state)
best_area = current_area
best_delay = current_delay

# Temperature bounds 
# Start temperature at ~10% of the baseline area to allow initial exploration
initial_temp = max_area * 0.1 
final_temp = 0.01
# Calculate exponential cooling rate based on the number of iterations
cooling_rate = (final_temp / initial_temp) ** (1.0 / num_iterations)
current_temp = initial_temp

# Lists to store Area values for plotting
episode_area = []
best_area_over_time = []

def generate_neighbor(state, num_arms):
    """ Swaps exactly ONE gate from the current subset to create a neighbor. """
    neighbor = set(state)
    gate_to_remove = random.choice(list(neighbor))
    neighbor.remove(gate_to_remove)
    
    while True:
        new_gate = random.randint(0, num_arms - 1)
        if new_gate not in neighbor:
            neighbor.add(new_gate)
            break
            
    return list(neighbor)

print("\nStarting Simulated Annealing Optimization...")

for i in range(num_iterations):
    print(f"Iteration: {i:03d} | Temp: {current_temp:.2f} | Current Best Area: {best_area:.2f}")
    
    # 1. Generate a neighbor state (swap 1 gate)
    neighbor_state = generate_neighbor(current_state, num_arms)
    
    # 2. Evaluate the neighbor
    neighbor_delay, neighbor_area = technology_mapper(genlib_origin, neighbor_state)
    
    if neighbor_area == float('inf'):
        neighbor_area = max_area * 2 # Penalty for failing synthesis

    # 3. Calculate Area difference
    delta_area = neighbor_area - current_area
    
    # 4. Acceptance Check
    # If the new area is smaller (better), OR if it's worse but passes the probability check
    if delta_area < 0 or random.random() < math.exp(-delta_area / current_temp):
        current_state = neighbor_state
        current_area = neighbor_area
        current_delay = neighbor_delay
        
        # 5. Update global best
        if current_area < best_area:
            best_area = current_area
            best_state = list(current_state)
            best_delay = current_delay
            print(f"   >>> NEW GLOBAL BEST FOUND! Area: {best_area:.2f}")

    # Track metrics for the graph
    episode_area.append(current_area if current_area < max_area*2 else float('nan'))
    best_area_over_time.append(best_area)
    
    # 6. Cool down the temperature
    current_temp *= cooling_rate

end = time.time()
runtime = end - start

print("\n--- OPTIMIZATION COMPLETE ---")
print("Best Cells (Indices):", best_state)
print("Best Delay:", best_delay)
print("Best Area:", best_area)
print("Total time:", runtime)

print("\n>> Generating Area Optimization Plot...")
plt.figure(figsize=(10, 6))

episodes_x = list(range(num_iterations))
clean_best_area = [val/max_area for val in best_area_over_time]
plt.plot(episodes_x, clean_best_area, label='Best Area Over Time', color='blue', linewidth=2.5)

# Plot valid sampled areas (Current State areas)
valid_areas = [(idx, val/max_area) for idx, val in enumerate(episode_area) if not math.isnan(val)]
if valid_areas:
    x_vals, y_vals = zip(*valid_areas)
    plt.scatter(x_vals, y_vals, color='red', alpha=0.3, label='Current State Area', s=15)

plt.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Baseline Area (1.0)')

plt.title(f"Simulated Annealing Area Optimization\nDesign: {design.split('/')[-1] if '/' in design else design}")
plt.xlabel("Search Iterations")
plt.ylabel("Normalized Area (vs Baseline)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Add training time and best Area annotation
plt.text(0.8, 0.98, f'Training Time: {runtime:.2f}s\nBest Area: {clean_best_area[-1]:.3f}', 
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

safe_design_name = design.replace('/', '_').replace('.', '_')
safe_lib_name = lib_origin.replace('/', '_').replace('.', '_')
output_path = f"areatest/sa_{num_iterations}_{safe_design_name}_{sample_gate}_{safe_lib_name}_area.png"
plt.savefig(output_path, dpi=300)
print(f">> Visualization successfully saved to {output_path}")