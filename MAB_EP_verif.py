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

# --- Added target_ratio to command line arguments ---
target_ratio = 0.8
sample_gate = int(sys.argv[-3])
design = sys.argv[-2]
genlib_origin = sys.argv[-1]

lib_origin = genlib_origin[:-7] + '.lib'
temp_blif = "temp_blifs/" + design[:-5] + "_ep_temp.blif"
lib_path = "gen_newlibs/"

start=time.time()
abc_cmd = "read %s;read %s; map; write %s; read %s;read -m %s; ps; topo; upsize; dnsize; stime; " % (genlib_origin, design, temp_blif, lib_origin, temp_blif)
print(abc_cmd)
res = subprocess.check_output(('wsl', 'abc', '-c', abc_cmd))
print(res)
match_d = re.search(r"Delay\s*=\s*([\d.]+)\s*ps", str(res))
match_a = re.search(r"Area\s*=\s*([\d.]+)", str(res))

# Baseline
max_delay = float(match_d.group(1))
max_area = float(match_a.group(1))
baseline_adp = max_delay * max_area
target_adp = baseline_adp * target_ratio

print("Baseline Delay:", max_delay)
print("Baseline Area:", max_area)
print(f"Baseline ADP: {baseline_adp}")
print(f"Target ADP Ratio: {target_ratio} (Target ADP: {target_adp})")

# Mapper call
def technology_mapper(genlib_origin, partial_cell_library):
    with open(genlib_origin, 'r') as f:
        f_lines = [line.strip() for line in f if line.startswith("GATE") and not line.startswith("GATE BUF") and not line.startswith("GATE INV") and not line.startswith("GATE sky130_fd_sc_hd__buf") and not line.startswith("GATE sky130_fd_sc_hd__inv") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv")]
    f.close()
    with open(genlib_origin, 'r') as f:
        f_keep = [line.strip() for line in f if line.startswith("GATE BUF") or line.startswith("GATE INV") or line.startswith("GATE sky130_fd_sc_hd__buf") or line.startswith("GATE sky130_fd_sc_hd__inv") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv")]
    f.close()
    
    lines_partial = [f_lines[i] for i in partial_cell_library]
    lines_partial = lines_partial + f_keep

    output_genlib_file = lib_path + design + "_" + str(len(lines_partial)) + "_ep_samplelib.genlib"
    with open(output_genlib_file, 'w') as out_gen:
        for line in lines_partial:
            out_gen.write(line + '\n')
    out_gen.close() 

    abc_cmd = "read %s;read %s; map; write %s; read %s;read -m %s; ps; topo; upsize; dnsize; stime; " % (output_genlib_file, design, temp_blif, lib_origin, temp_blif)
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
    return -np.sqrt(normalized_delay * normalized_area) 

# Epsilon-Greedy MAB Class
class EpsilonGreedyMAB:
  def __init__(self, num_arms, epsilon, sample_gate):
    self.num_arms = num_arms
    self.epsilon = epsilon  
    self.q_values = [0.0] * num_arms  
    self.counts = [0] * num_arms  
    self.sample_gate = sample_gate
    
  def select_action(self):
    selected_cells = set()  
    while len(selected_cells) < self.sample_gate:
        if random.random() > self.epsilon:
            select = (np.argmax(self.q_values))
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
        f_lines = [line.strip() for line in f if line.startswith("GATE") and not line.startswith("GATE BUF") and not line.startswith("GATE INV") and not line.startswith("GATE sky130_fd_sc_hd__buf") and not line.startswith("GATE sky130_fd_sc_hd__inv") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv")]
f.close()
num_arms=len(f_lines)

mab = EpsilonGreedyMAB(num_arms, epsilon=0.2, sample_gate=num_cells_select)  
best_cells = None
best_result = (float('inf'), float('inf'))  
best_reward = -float('inf')  

episode_adp = []
best_adp_over_time = []

# --- Replaced For-loop with While-loop based on target ratio ---
i = 0
current_best_adp = float('inf')

print("\n>> Starting Optimization Loop...")
while current_best_adp > target_adp:
  print(f"Iteration: {i}") #| Current Best ADP: {current_best_adp if current_best_adp != float('inf') else 'N/A'} | Target ADP: {target_adp}")
  selected_cells = mab.select_action()
  
  delay, area = technology_mapper(genlib_origin, selected_cells)
  
  # Properly check for NaN values in Python
  if math.isnan(delay) or math.isnan(area): 
      reward = -float('inf') 
      adp = float('inf')
  else:
      reward = calculate_reward(max_delay, max_area, delay, area)
      adp = delay * area
      
  episode_adp.append(adp)
  
  if best_adp_over_time:
      current_best_adp = min(best_adp_over_time[-1], adp)
      best_adp_over_time.append(current_best_adp)
  else:
      current_best_adp = adp
      best_adp_over_time.append(current_best_adp)
      
  if reward > best_reward:
      best_reward = reward
      print("-> Current best reward: ", best_reward)
      best_result = (delay, area)
      print("-> Current best result: ", best_result)
      best_cells = selected_cells
      
  mab.update(selected_cells, reward)
  i += 1

# Capture total iterations run
num_iterations = i
end=time.time()
runtime=end-start

print("\n=== OPTIMIZATION COMPLETE ===")
print("Target ADP reached!")
print("Total Iterations:", num_iterations)
print("Best Cells:", best_cells)
print("Best Delay:", best_result[0])
print("Best Area:", best_result[1])
print("Best ADP:", current_best_adp)
print("Best Reward:", best_reward)
print("Total time:", runtime)

print("\n>> Generating ADP Optimization Plot...")
plt.figure(figsize=(10, 6))

episodes_x = list(range(num_iterations))
clean_best_adp = [val/baseline_adp if val != float('inf') else 1.0 for val in best_adp_over_time]
plt.plot(episodes_x, clean_best_adp, label='Best ADP Over Time', color='blue', linewidth=2.5)

valid_adps = [(idx, val/baseline_adp) for idx, val in enumerate(episode_adp) if val != float('inf')]
if valid_adps:
    x_vals, y_vals = zip(*valid_adps)
    plt.scatter(x_vals, y_vals, color='red', alpha=0.3, label='Episode Sampled ADP', s=15)

plt.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Baseline ADP')
plt.axhline(y=target_ratio, color='purple', linestyle='-.', linewidth=2, label=f'Target ADP Ratio ({target_ratio})')

plt.title(f"MAB_EP ADP Optimization\nState: No Feature + Selection Mask (Design: {design})")
plt.xlabel("Training Episodes")
plt.ylabel("Area-Delay Product (ADP) Ratio")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

plt.text(0.8, 0.98, f'Training Time: {runtime:.2f}s\nBest ADP Ratio: {clean_best_adp[-1]:.2f}\nIterations: {num_iterations}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Create directory if it doesn't exist
os.makedirs("random_test", exist_ok=True)
plt.savefig(f"random_test/mab_ep_target_{target_ratio}.png", dpi=300)
print(f">> Visualization successfully saved to random_test/mab_ep_target_{target_ratio}.png")