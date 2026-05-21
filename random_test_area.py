import random
import sys
import os
import numpy as np
import subprocess
import re
import time
import matplotlib.pyplot as plt
import concurrent.futures
import shutil

# Make sure temporary directories exist
os.makedirs("temp_blifs", exist_ok=True)
os.makedirs("gen_newlibs", exist_ok=True)
os.makedirs("random_test", exist_ok=True)

def parse_abc_output(res_bytes):
    """Helper function to parse delay and area from ABC output"""
    match_d = re.search(r"Delay\s*=\s*([\d.]+)\s*ps", str(res_bytes))
    match_a = re.search(r"Area\s*=\s*([\d.]+)", str(res_bytes))
    if match_d and match_a:
        return float(match_d.group(1)), float(match_a.group(1))
    return float('nan'), float('nan')

def technology_mapper(genlib_origin, partial_cell_library, task_id, design, lib_origin):
    """
    Modified technology mapper that uses a unique task_id to avoid 
    race conditions when running parallel threads.
    """
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

    # Unique temp files for parallel safety
    lib_path = "gen_newlibs/"
    output_genlib_file = f"{lib_path}{design}_{len(lines_partial)}_ep_samplelib_{task_id}.genlib"
    temp_blif = f"temp_blifs/{design[:-5]}_ep_temp_{task_id}.blif"

    with open(output_genlib_file, 'w') as out_gen:
        for line in lines_partial:
            out_gen.write(line + '\n')

    abc_cmd = "read %s;read %s; amap; write %s; read %s;read -m %s; ps; topo; upsize; dnsize; stime; " % (
        output_genlib_file, design, temp_blif, lib_origin, temp_blif)
    
    try:
        res = subprocess.check_output(('wsl', 'abc', '-c', abc_cmd), stderr=subprocess.STDOUT)
        delay, area = parse_abc_output(res)
    except subprocess.CalledProcessError:
        delay, area = float('nan'), float('nan')
    
    # Cleanup temporary files to prevent disk bloating
    if os.path.exists(output_genlib_file):
        os.remove(output_genlib_file)
    if os.path.exists(temp_blif):
        os.remove(temp_blif)

    return delay, area

def calculate_reward(max_delay, max_area, delay, area):
    normalized_delay = delay / max_delay
    normalized_area = area / max_area
    return -normalized_area

def select_random_cells(num_arms, sample_gate):
    return random.sample(range(num_arms), sample_gate)

def evaluate_episode(task_id, num_arms, sample_gate, genlib_origin, design, lib_origin):
    """Wrapper function to be run by the multiprocessing pool"""
    # Seed locally so threads don't produce the exact same "random" sequences
    random.seed((os.getpid() * int(time.time())) % 123456789)
    
    selected_cells = select_random_cells(num_arms, sample_gate)
    delay, area = technology_mapper(genlib_origin, selected_cells, task_id, design, lib_origin)
    return task_id, selected_cells, delay, area


if __name__ == '__main__':
    random.seed(time.time())

    # Input arguments: sample_gate design genlib_origin
    if len(sys.argv) < 4:
        print("Usage: python script.py <sample_gate> <design> <genlib_origin>")
        sys.exit(1)

    sample_gate = int(sys.argv[-3])
    design = sys.argv[-2]
    genlib_origin = sys.argv[-1]
    lib_origin = genlib_origin[:-7] + '.lib'
    
    # Run Baseline
    start = time.time()
    print(">> Running Baseline evaluation...")
    baseline_temp_blif = f"temp_blifs/{design[:-5]}_ep_temp_baseline.blif"
    abc_cmd = "read %s;read %s; amap; write %s; read %s;read -m %s; ps; topo; upsize; dnsize; stime; " % (
        genlib_origin, design, baseline_temp_blif, lib_origin, baseline_temp_blif)
    
    res = subprocess.check_output(('wsl', 'abc', '-c', abc_cmd))
    max_delay, max_area = parse_abc_output(res)

    print("Baseline Delay:", max_delay)
    print("Baseline Area:", max_area)

    # Initialization / Count valid arms
    with open(genlib_origin, 'r') as f:
        f_lines = [line.strip() for line in f if line.startswith("GATE")
                   and not line.startswith("GATE BUF")
                   and not line.startswith("GATE INV")
                   and not line.startswith("GATE sky130_fd_sc_hd__buf")
                   and not line.startswith("GATE sky130_fd_sc_hd__inv")
                   and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf")
                   and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv")]
    num_arms = len(f_lines)

    num_iterations = 3000
    
    # ---------------------------------------------
    # Parallel Execution Block
    # ---------------------------------------------
    print(f"\n>> Starting {num_iterations} parallel iterations using ProcessPoolExecutor...")
    results = []
    
    # Use max workers based on CPU count. 
    # You can restrict this (e.g., max_workers=8) if your system slows down too much.
    num_cores = os.cpu_count() or 4
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Submit all tasks to the process pool
        futures = {
            executor.submit(
                evaluate_episode, i, num_arms, sample_gate, genlib_origin, design, lib_origin
            ): i for i in range(num_iterations)
        }
        
        # As tasks complete, collect them and show progress
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            completed += 1
            if completed % 50 == 0:
                print(f"Completed {completed}/{num_iterations} iterations...")

    # Sort results by task_id to preserve the chronological "over time" aspect for plotting
    results.sort(key=lambda x: x[0])

    # Process metrics after collecting parallel results
    best_cells = None
    best_result = (float('inf'), float('inf'))
    best_reward = -float('inf')

    episode_area = []
    best_area_over_time = []

    for task_id, selected_cells, delay, area in results:
        if np.isnan(delay) or np.isnan(area):
            reward = -float('inf')
        else:
            reward = calculate_reward(max_delay, max_area, delay, area)

        episode_area.append(area)
        
        # Track running best
        if best_area_over_time:
            best_area_over_time.append(min(best_area_over_time[-1], area))
        else:
            best_area_over_time.append(area)

        # Update absolute best
        if reward > best_reward:
            best_reward = reward
            best_result = (delay, area)
            best_cells = selected_cells

    end = time.time()
    runtime = end - start

    print("\n==============================")
    print("Parallel Execution Summary:")
    print("==============================")
    print("Best Cells:", best_cells)
    print("Best Delay:", best_result[0])
    print("Best Area:", best_result[1])
    print("Best Reward:", best_reward)
    print("Total time:", runtime)

    # ---------------------------------------------
    # Plotting Output
    # ---------------------------------------------
    print("\n>> Generating Area Optimization Plot...")
    plt.figure(figsize=(10, 6))
    episodes_x = list(range(num_iterations))
    baseline_area = max_area
    clean_best_area = [val/baseline_area if val != float('inf') else 1.0 for val in best_area_over_time]
    plt.plot(episodes_x, clean_best_area, label='Best Area Over Time', color='blue', linewidth=2.5)

    valid_areas = [(idx, val/baseline_area) for idx, val in enumerate(episode_area) if val != float('inf')]
    if valid_areas:
        x_vals, y_vals = zip(*valid_areas)
        plt.scatter(x_vals, y_vals, color='red', alpha=0.3, label='Episode Sampled Area', s=15)

    plt.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Baseline Area')
    plt.title(f"Random Selection Area Optimization (Parallel)\nDesign: {design}")
    plt.xlabel("Training Episodes")
    plt.ylabel("Area")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.text(0.03, 0.98, f'Training Time: {runtime:.2f}s\nBest Area: {clean_best_area[-1]:.4f}',
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Safely extract design name for filename
    try:
        design_name = design.split('/')[1].split('.')[0]
    except IndexError:
        design_name = design.split('.')[0]

    output_path = f"random_test/random_{num_iterations}_{design_name}_{sample_gate}_{lib_origin[:-4].replace('/', '_')}_area.png"
    plt.savefig(output_path, dpi=300)
    print(f">> Visualization successfully saved to {output_path}")