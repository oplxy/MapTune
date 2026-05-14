import random
import sys
import os 
import numpy as np
import subprocess
from subprocess import PIPE
import re
import time
import matplotlib.pyplot as plt

genlib_origin = sys.argv[-1]
lib_origin = genlib_origin[:-7] + '.lib'
design = sys.argv[-2]
sample_gate = int(sys.argv[-3])
temp_blif = "temp_blifs/" + design[:-5] + "_ep_temp.blif"
lib_path = "gen_newlibs/"

random.seed(time.time())
start=time.time()
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
        #f_lines = [line.strip() for line in f if line.startswith("GATE") and not any(substr in line for substr in ["BUF", "INV", "inv", "buf"])]
        f_lines = [line.strip() for line in f if line.startswith("GATE") and not line.startswith("GATE BUF") and not line.startswith("GATE INV") and not line.startswith("GATE sky130_fd_sc_hd__buf") and not line.startswith("GATE sky130_fd_sc_hd__inv") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv")]
    f.close()
    with open(genlib_origin, 'r') as f:
        #f_keep = [line.strip() for line in f if any(substr in line for substr in ["BUF", "INV", "inv", "buf"])]
        f_keep = [line.strip() for line in f if line.startswith("GATE BUF") or line.startswith("GATE INV") or line.startswith("GATE sky130_fd_sc_hd__buf") or line.startswith("GATE sky130_fd_sc_hd__inv") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv")]
    f.close()
    lines_partial = [f_lines[i] for i in partial_cell_library]
    lines_partial = lines_partial + f_keep

    output_genlib_file = lib_path + design + "_" + str(len(lines_partial)) + "_ep_samplelib.genlib"
    with open(output_genlib_file, 'w') as out_gen:
        for line in lines_partial:
            out_gen.write(line + '\n')
    out_gen.close() 

    abc_cmd = "read %s;read %s; amap; write %s; read %s;read -m %s; ps; topo; upsize; dnsize; stime; " % (output_genlib_file, design, temp_blif, lib_origin, temp_blif)
    res = subprocess.check_output(('wsl', 'abc', '-c', abc_cmd))
    match_d = re.search(r"Delay\s*=\s*([\d.]+)\s*ps", str(res))
    match_a = re.search(r"Area\s*=\s*([\d.]+)", str(res))
    if match_d and match_a:
        delay = float(match_d.group(1))
        area = float(match_a.group(1))
    else:
        delay, area = float("NaN"),float("NaN")
    return delay, area

# Reward calculation
def calculate_reward(max_delay, max_area, delay, area):

    normalized_delay = delay / max_delay
    normalized_area = area / max_area

    return -normalized_area


  # Epsilon-Greedy MAB Class
class EpsilonGreedyMAB:
  def __init__(self, num_arms, epsilon, sample_gate):
    self.num_arms = num_arms
    self.epsilon = epsilon  # Exploration vs Exploitation factor (0 to 1)
    self.q_values = [0.0] * num_arms  # Estimated average reward for each arm
    self.counts = [0] * num_arms  # Number of times each arm was selected
    self.sample_gate = sample_gate
  def select_action(self):
    selected_cells = set()  
    
    while len(selected_cells) < self.sample_gate:
        if random.random() > self.epsilon:
            select = (np.argmax(self.q_values))
        # If prob falls in epsilon range, do exploration
        else:
            select = (random.randint(0, self.num_arms - 1))
    
        if select not in selected_cells:
            selected_cells.add(select)

    return list(selected_cells)

  def update(self, selected_arm, reward):
      for arm in selected_arm:
            self.counts[arm] += 1
            self.q_values[arm] = (self.q_values[arm] * self.counts[arm] + reward) / self.counts[arm]


# Initialization
num_cells_select = sample_gate
with open(genlib_origin, 'r') as f:
        # Modify constraints for extra kept gates
        #f_lines = [line.strip() for line in f if line.startswith("GATE") and not any(substr in line for substr in ["BUF", "INV", "inv", "buf"])]
        f_lines = [line.strip() for line in f if line.startswith("GATE") and not line.startswith("GATE BUF") and not line.startswith("GATE INV") and not line.startswith("GATE sky130_fd_sc_hd__buf") and not line.startswith("GATE sky130_fd_sc_hd__inv") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv")]
f.close()
num_arms=len(f_lines)
#print(num_arms)
mab = EpsilonGreedyMAB(num_arms, epsilon=0.1, sample_gate=num_cells_select)  
best_cells = None
best_result = (float('inf'), float('inf'))  
best_reward = -float('inf')  # Track best reward

# Lists to store Area values
episode_area = []
best_area_over_time = []

# Main Loop
num_iterations = 1001

for i in range(num_iterations):
  print("Iteration: ", i)
  selected_cells = mab.select_action()
  #try:
  delay, area = technology_mapper(genlib_origin, selected_cells)
  if np.isnan(delay) or np.isnan(area):
      reward = -float('inf') 
      adp = float('inf')
  else:
      reward = calculate_reward(max_delay, max_area, delay, area)
      adp = delay * area
  episode_area.append(area)
  if best_area_over_time:
      best_area_over_time.append(min(best_area_over_time[-1], area))
  else:
      best_area_over_time.append(area)
  if reward > best_reward:
      best_reward = reward
      print("Current best reward: ", best_reward)
      best_result = (delay, area)
      print("Current best result: ", best_result)
      best_cells = selected_cells
      print("Current best cells: ", best_cells)
  mab.update(selected_cells, reward)
end=time.time()
runtime=end-start

print("Best Cells:", best_cells)
print("Best Delay:", best_result[0])
print("Best Area:", best_result[1])
print("Best Reward:", best_reward)
print("Total time:", runtime)

print("\n>> Generating Area Optimization Plot...")
plt.figure(figsize=(10, 6))

episodes_x = list(range(num_iterations))
baseline_area = max_area
clean_best_area = [val/baseline_area if val != float('inf') else baseline_area/baseline_area for val in best_area_over_time]
plt.plot(episodes_x, clean_best_area, label='Best Area Over Time', color='blue', linewidth=2.5)

valid_areas = [(idx, val/baseline_area) for idx, val in enumerate(episode_area) if val != float('inf')]
if valid_areas:
    x_vals, y_vals = zip(*valid_areas)
    plt.scatter(x_vals, y_vals, color='red', alpha=0.3, label='Episode Sampled Area', s=15)

plt.axhline(y=baseline_area/baseline_area, color='green', linestyle='--', linewidth=2, label='Baseline Area')

plt.title(f"MAB_EP Area Optimization\nDesign: {design}")
plt.xlabel("Training Episodes")
plt.ylabel("Area")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Add training time and best Area annotation
plt.text(0.8, 0.98, f'Training Time: {runtime:.2f}s\nBest Area: {clean_best_area[-1]:.2f}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

output_path = f"random_test/mab_ep_{num_iterations}_{design.split('/')[1].split('.')[0]}_{sample_gate}_{lib_origin[:-4]}_area.png"
plt.savefig(output_path, dpi=300)
print(f">> Visualization successfully saved to {output_path}")

