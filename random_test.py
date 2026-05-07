import random
import sys
import os
import numpy as np
import subprocess
from subprocess import PIPE
import re
import time
import matplotlib.pyplot as plt

random.seed(time.time())

# Input arguments: sample_gate design genlib_origin
sample_gate = int(sys.argv[-3])
design = sys.argv[-2]
genlib_origin = sys.argv[-1]
lib_origin = genlib_origin[:-7] + '.lib'
temp_blif = "temp_blifs/" + design[:-5] + "_ep_temp.blif"
lib_path = "gen_newlibs/"

start = time.time()
abc_cmd = "read %s;read %s; map; write %s; read %s;read -m %s; ps; topo; upsize; dnsize; stime; " % (
    genlib_origin, design, temp_blif, lib_origin, temp_blif)
print(abc_cmd)
res = subprocess.check_output(('wsl', 'abc', '-c', abc_cmd))
print(res)
match_d = re.search(r"Delay\s*=\s*([\d.]+)\s*ps", str(res))
match_a = re.search(r"Area\s*=\s*([\d.]+)", str(res))
max_delay = float(match_d.group(1))
max_area = float(match_a.group(1))

print("Baseline Delay:", max_delay)
print("Baseline Area:", max_area)

# Mapper call

def technology_mapper(genlib_origin, partial_cell_library):
    with open(genlib_origin, 'r') as f:
        f_lines = [line.strip() for line in f if line.startswith("GATE")
                   and not line.startswith("GATE BUF")
                   and not line.startswith("GATE INV")
                   and not line.startswith("GATE sky130_fd_sc_hd__buf")
                   and not line.startswith("GATE sky130_fd_sc_hd__inv")
                   and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf")
                   and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv")]
    with open(genlib_origin, 'r') as f:
        f_keep = [line.strip() for line in f if line.startswith("GATE BUF")
                  or line.startswith("GATE INV")
                  or line.startswith("GATE sky130_fd_sc_hd__buf")
                  or line.startswith("GATE sky130_fd_sc_hd__inv")
                  or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf")
                  or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv")]

    lines_partial = [f_lines[i] for i in partial_cell_library]
    lines_partial.extend(f_keep)

    output_genlib_file = lib_path + design + "_" + str(len(lines_partial)) + "_ep_samplelib.genlib"
    with open(output_genlib_file, 'w') as out_gen:
        for line in lines_partial:
            out_gen.write(line + '\n')

    abc_cmd = "read %s;read %s; map; write %s; read %s;read -m %s; ps; topo; upsize; dnsize; stime; " % (
        output_genlib_file, design, temp_blif, lib_origin, temp_blif)
    res = subprocess.check_output(('wsl', 'abc', '-c', abc_cmd))
    match_d = re.search(r"Delay\s*=\s*([\d.]+)\s*ps", str(res))
    match_a = re.search(r"Area\s*=\s*([\d.]+)", str(res))
    if match_d and match_a:
        delay = float(match_d.group(1))
        area = float(match_a.group(1))
    else:
        delay, area = float('nan'), float('nan')
    return delay, area

# Reward calculation
def calculate_reward(max_delay, max_area, delay, area):
    normalized_delay = delay / max_delay
    normalized_area = area / max_area
    return -np.sqrt(normalized_delay * normalized_area)


def select_random_cells(num_arms, sample_gate):
    return random.sample(range(num_arms), sample_gate)

# Initialization
with open(genlib_origin, 'r') as f:
    f_lines = [line.strip() for line in f if line.startswith("GATE")
               and not line.startswith("GATE BUF")
               and not line.startswith("GATE INV")
               and not line.startswith("GATE sky130_fd_sc_hd__buf")
               and not line.startswith("GATE sky130_fd_sc_hd__inv")
               and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf")
               and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv")]
num_arms = len(f_lines)

best_cells = None
best_result = (float('inf'), float('inf'))
best_reward = -float('inf')

episode_adp = []
best_adp_over_time = []

num_iterations = 3000

for i in range(num_iterations):
    print("Iteration:", i)
    selected_cells = select_random_cells(num_arms, sample_gate)
    delay, area = technology_mapper(genlib_origin, selected_cells)
    if np.isnan(delay) or np.isnan(area):
        reward = -float('inf')
        adp = float('inf')
    else:
        reward = calculate_reward(max_delay, max_area, delay, area)
        adp = delay * area

    episode_adp.append(adp)
    if best_adp_over_time:
        best_adp_over_time.append(min(best_adp_over_time[-1], adp))
    else:
        best_adp_over_time.append(adp)

    if reward > best_reward:
        best_reward = reward
        best_result = (delay, area)
        best_cells = selected_cells
        print("Current best reward:", best_reward)
        print("Current best result:", best_result)
        print("Current best cells:", best_cells)

end = time.time()
runtime = end - start

print("Best Cells:", best_cells)
print("Best Delay:", best_result[0])
print("Best Area:", best_result[1])
print("Best Reward:", best_reward)
print("Total time:", runtime)

print("\n>> Generating ADP Optimization Plot...")
plt.figure(figsize=(10, 6))
episodes_x = list(range(num_iterations))
baseline_adp = max_delay * max_area
clean_best_adp = [val/baseline_adp if val != float('inf') else 1.0 for val in best_adp_over_time]
plt.plot(episodes_x, clean_best_adp, label='Best ADP Over Time', color='blue', linewidth=2.5)

valid_adps = [(idx, val/baseline_adp) for idx, val in enumerate(episode_adp) if val != float('inf')]
if valid_adps:
    x_vals, y_vals = zip(*valid_adps)
    plt.scatter(x_vals, y_vals, color='red', alpha=0.3, label='Episode Sampled ADP', s=15)

plt.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Baseline ADP')
plt.title(f"Random Selection ADP Optimization\nState: Pure Random Selection (Design: {design})")
plt.xlabel("Training Episodes")
plt.ylabel("Area-Delay Product (ADP)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.text(0.8, 0.98, f'Training Time: {runtime:.2f}s\nBest ADP: {clean_best_adp[-1]:.2f}',
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

output_path = f"random_test/random_{num_iterations}_{design.split('/')[1].split('.')[0]}_{sample_gate}_{lib_origin[:-4]}.png"
plt.savefig(output_path, dpi=300)
print(f">> Visualization successfully saved to {output_path}")
