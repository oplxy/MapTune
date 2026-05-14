import gymnasium as gym
from gymnasium import spaces
import numpy as np
import subprocess
import re
import sys
import os
import random
from collections import deque
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ==========================================
# 1. Feature Extraction Helper
# ==========================================
def extract_gate_features(f_lines):
    """Parses .genlib lines to extract numerical features, treating every cell individually."""
    
    def get_inputs_from_name(gate_name):
        """Heuristically determines the number of inputs based on the gate's naming convention."""
        # Strip standard library prefixes (like sky130_fd_sc_hd__)
        if '__' in gate_name:
            gate_name = gate_name.split('__')[-1]
            
        # Isolate the functional part before drive strength (e.g., 'x' or '_')
        base_part = re.split(r'x|_', gate_name)[0]
        
        # Find all digits in this functional base part
        digits = re.findall(r'\d', base_part)
        
        if not digits:
            return 1.0 # Default for INV, BUF, etc.
            
        # Sum the digits (e.g., 'AOI221' -> 2 + 2 + 1 = 5)
        return float(sum(int(d) for d in digits))

    # ==========================================
    # Pass 1: Collect every exact, individual cell name
    # ==========================================
    exact_gate_names = []
    for line in f_lines:
        name = line.split()[1]
        exact_gate_names.append(name)
        
    # Sorting ensures the one-hot encoding array is consistent
    exact_gate_names = sorted(list(set(exact_gate_names)))
    print(exact_gate_names)

    # ==========================================
    # Pass 2: Build the Feature Matrix
    # ==========================================
    features = []
    for line in f_lines:
        parts = line.split()
        name = parts[1]
        area = float(parts[2])
        
        # 1. Exact One-hot encode the specific cell name
        type_vec = [1.0 if name == g else 0.0 for g in exact_gate_names]
        
        # 2. Extract inputs from name string
        num_inputs = get_inputs_from_name(name)
        
        # 3. Create the feature vector for this gate
        gate_feature = [area, num_inputs] + type_vec
        features.append(gate_feature)
        
    return np.array(features, dtype=np.float32)

# ==========================================
# 2. Replay Buffer
# ==========================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# ==========================================
# 3. Gymnasium Environment (Feature-Aware)
# ==========================================
class GateSelectionEnv(gym.Env):
    """Gate selection environment for reinforcement learning"""
    metadata = {'render.modes': ['human']}

    def __init__(self, genlib_origin, lib_path, design, total_gates, sample_gate, max_delay, max_area, gate_features):
        super().__init__()
        self.genlib_origin = genlib_origin
        self.lib_path = lib_path
        self.design = design
        self.total_gates = total_gates
        self.sample_gate = sample_gate
        self.max_delay = max_delay
        self.max_area = max_area
        
        # New Feature tracking
        self.gate_features = gate_features
        self.feature_size = gate_features.shape[1]
        
        # State now consists of both the mask and the cumulative features
        self.state_mask = np.zeros(self.total_gates, dtype=int)
        self.state_features = np.zeros(self.feature_size, dtype=np.float32)
        self.selection_count = 0

    def step(self, action):
        if self.state_mask[action] == 0 and self.selection_count < self.sample_gate:
            self.state_mask[action] = 1
            # Add the features of the selected gate to the environment state
            self.state_features += self.gate_features[action] 
            self.selection_count += 1

        done = self.selection_count == self.sample_gate
        reward = 0
        delay = 1
        area = 1
        
        next_state_feat = self.state_features.copy()
        next_state_mask = self.state_mask.copy()

        if done:
            # Evaluate the selected gates only once all required selections are made
            delay, area = self.technology_mapper(list(np.where(self.state_mask == 1)[0]))
            reward = self.calculate_reward(delay, area)
            # Do NOT reset here; the training loop calls env.reset() at the start of each episode
            
        return (next_state_feat, next_state_mask), reward, done, delay, area

    def technology_mapper(self, partial_cell_library):
        # Read the original library file and filter gates
        with open(self.genlib_origin, 'r') as f:
            f_lines = [line.strip() for line in f if line.startswith("GATE") and not line.startswith("GATE BUF") and not line.startswith("GATE INV") and not line.startswith("GATE sky130_fd_sc_hd__buf") and not line.startswith("GATE sky130_fd_sc_hd__inv") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv")]
        with open(self.genlib_origin, 'r') as f:
            f_keep = [line.strip() for line in f if line.startswith("GATE BUF") or line.startswith("GATE INV") or line.startswith("GATE sky130_fd_sc_hd__buf") or line.startswith("GATE sky130_fd_sc_hd__inv") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv")]

        lines_partial = [f_lines[i] for i in partial_cell_library] + f_keep
        output_genlib_file = self.lib_path + self.design + "_" + str(len(lines_partial)) + "_dqn_samplelib.genlib"
        lib_origin = self.genlib_origin[:-7] + '.lib'
        temp_blif = "temp_blifs/" + self.design[:-5] + "_dqn_temp.blif"
        
        os.makedirs(self.lib_path, exist_ok=True)
        os.makedirs("temp_blifs", exist_ok=True)
        
        with open(output_genlib_file, 'w') as out_gen:
            for line in lines_partial:
                out_gen.write(line + '\n')

        # Execute the mapping command using ABC
        abc_cmd = f"wsl abc -c 'read {output_genlib_file}; read {self.design}; map -a; write {temp_blif}; read {lib_origin}; read -m {temp_blif}; ps; topo; upsize; dnsize; stime;'"
        try:
            res = subprocess.check_output(abc_cmd, shell=True, text=True)
            match_d = re.search(r"Delay\s*=\s*([\d.]+)\s*ps", res)
            match_a = re.search(r"Area\s*=\s*([\d.]+)", res)
            delay = float(match_d.group(1)) if match_d else float('inf')
            area = float(match_a.group(1)) if match_a else float('inf')
        except subprocess.CalledProcessError as e:
            print("Failed to execute ABC:", e)
            delay, area = float('inf'), float('inf')

        return delay, area

    def calculate_reward(self, delay, area):
        if delay == float('inf') or area == float('inf'):
            return float('-inf')
        normalized_delay = delay / self.max_delay
        normalized_area = area / self.max_area
        return -np.sqrt(normalized_delay * normalized_area)

    def reset(self):
        self.state_mask = np.zeros(self.total_gates, dtype=int)
        self.state_features = np.zeros(self.feature_size, dtype=np.float32)
        self.selection_count = 0
        return (self.state_features.copy(), self.state_mask.copy())

    def render(self, mode='human'):
        print(f"Selected Gates: {np.where(self.state_mask == 1)[0]}")

    def close(self):
        pass

# ==========================================
# 4. Neural Network
# ==========================================
class DQNNetwork(nn.Module):
    def __init__(self, feature_size):
        super(DQNNetwork, self).__init__()
        # Input is now: (Current State Features) + (Candidate Gate Features)
        self.fc1 = nn.Linear(feature_size * 2, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1) # Outputs a SINGLE predicted Q-Value

    def forward(self, state_features, action_features):
        x = torch.cat([state_features, action_features], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ==========================================
# 5. RL Agent
# ==========================================
class DQNAgent:
    def __init__(self, feature_size, gate_features, learning_rate=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = DQNNetwork(feature_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.gate_features = torch.FloatTensor(gate_features).to(self.device)
        self.num_actions = len(gate_features)
        self.gamma = 0.99

    def select_action(self, state, epsilon=0.2):
        state_features, state_mask = state
        valid_actions = np.where(state_mask == 0)[0] # Only look at unselected gates
        
        if np.random.rand() < epsilon:
            return np.random.choice(valid_actions)
        else:
            with torch.no_grad():
                # Test the current state against the features of ALL valid candidate gates
                s_tensor = torch.FloatTensor(state_features).unsqueeze(0).repeat(len(valid_actions), 1).to(self.device)
                a_tensor = self.gate_features[valid_actions]
                
                q_values = self.model(s_tensor, a_tensor).squeeze()
                
                # Pick the valid gate with the highest predicted Q-Value
                best_idx = q_values.argmax().item()
                return valid_actions[best_idx]
                
    def update_batch(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Extract features from the tuple states
        s_feat = torch.FloatTensor(np.array([s[0] for s in states])).to(self.device)
        ns_feat = torch.FloatTensor(np.array([ns[0] for ns in next_states])).to(self.device)
        
        actions_idx = list(actions)
        action_features = self.gate_features[actions_idx]
        
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)

        # 1. Current Q values for the actions we actually took
        current_qs = self.model(s_feat, action_features)
        
        # 2. Next Q values: Evaluate ALL possible actions for the next states to find max
        batch_size = len(states)
        ns_expanded = ns_feat.unsqueeze(1).repeat(1, self.num_actions, 1)
        act_expanded = self.gate_features.unsqueeze(0).repeat(batch_size, 1, 1)
        
        next_qs = self.model(ns_expanded, act_expanded).squeeze(-1) # Shape: [Batch, Num_Actions]
        
        # Mask out already-selected gates (mask==1) so they cannot be chosen as the max next action
        ns_masks = torch.FloatTensor(np.array([ns[1] for ns in next_states])).to(self.device)  # Shape: [Batch, Num_Actions]
        next_qs = next_qs.masked_fill(ns_masks.bool(), float('-inf'))
        
        max_next_qs = next_qs.max(dim=1, keepdim=True)[0]
        
        expected_qs = rewards + self.gamma * (1 - dones) * max_next_qs

        loss = F.mse_loss(current_qs, expected_qs.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

# ==========================================
# 6. Training Loop
# ==========================================
def train_agent(num_episodes, agent, env, batch_size, buffer_size):
    replay_buffer = ReplayBuffer(buffer_size)
    highest_reward = float('-inf')
    
    # Lists to store ADP values 
    episode_adp = []
    best_adp_over_time = []

    for episode in range(num_episodes):
        state = env.reset() # state is now a tuple: (features, mask)
        done = False
        episode_reward = 0.0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, delay, area = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            if done:
                episode_reward = reward
                
                if delay == float('inf') or area == float('inf'):
                    adp = float('inf')
                else:
                    adp = delay * area
                    
                episode_adp.append(adp)
                if best_adp_over_time:
                    best_adp_over_time.append(min(best_adp_over_time[-1], adp))
                else:
                    best_adp_over_time.append(adp)
                
                if reward > highest_reward:
                    highest_reward = reward
                    best_result = (delay, area)
                    print('Current Best Result: ', best_result)

            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                loss = agent.update_batch(batch)  # Process batch update

        print(f"Episode {episode + 1}, Episode Reward = {episode_reward:.4f}, Highest Reward = {highest_reward:.4f}")

    return episode_adp, best_adp_over_time

# ==========================================
# 7. Main Execution
# ==========================================
if __name__ == "__main__":
    genlib_origin = sys.argv[-1]
    lib_origin = genlib_origin[:-7] + '.lib'
    design = sys.argv[-2]
    sample_gate = int(sys.argv[-3])
    temp_blif = "temp_blifs/" + design[:-5] + "_dqn_temp.blif"
    lib_path = "gen_newlibs/"

    os.makedirs("temp_blifs", exist_ok=True)
    os.makedirs("gen_newlibs", exist_ok=True)

    # Calculate Baseline
    abc_cmd = "read %s;read %s; map -a; write %s; read %s;read -m %s; ps; topo; upsize; dnsize; stime; " % (genlib_origin, design, temp_blif, lib_origin, temp_blif)
    res = subprocess.check_output(('wsl', 'abc', '-c', abc_cmd))
    match_d = re.search(r"Delay\s*=\s*([\d.]+)\s*ps", str(res))
    match_a = re.search(r"Area\s*=\s*([\d.]+)", str(res))

    max_delay = float(match_d.group(1))
    max_area = float(match_a.group(1))
    print('Baseline Delay: ', max_delay)
    print('Baseline Area: ', max_area)

    # Read Gates and Extract Features
    with open(genlib_origin, 'r') as f:
            f_lines = [line.strip() for line in f if line.startswith("GATE") and not line.startswith("GATE BUF") and not line.startswith("GATE INV") and not line.startswith("GATE sky130_fd_sc_hd__buf") and not line.startswith("GATE sky130_fd_sc_hd__inv") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv")]
    
    gate_features_matrix = extract_gate_features(f_lines)
    feature_size = gate_features_matrix.shape[1]
    total_gates = len(f_lines)

    num_episodes = 2000
    batch_size = 10
    buffer_size = 10000

    # Initialize Environment and Agent
    env = GateSelectionEnv(genlib_origin, lib_path, design, total_gates, sample_gate, max_delay, max_area, gate_features_matrix)
    agent = DQNAgent(feature_size, gate_features_matrix)

    # Start Training
    start = time.time()
    episode_adp, best_adp_over_time = train_agent(num_episodes, agent, env, batch_size, buffer_size)
    end = time.time()

    runtime = end - start
    print('Total time: ', runtime)

    # ==========================================
    # 8. Visualization Output
    # ==========================================
    print("\n>> Generating ADP Optimization Plot...")
    os.makedirs("random_test", exist_ok=True)
    plt.figure(figsize=(10, 6))

    episodes_x = list(range(num_episodes))
    baseline_adp = max_delay * max_area

    clean_best_adp = [val/baseline_adp if val != float('inf') else baseline_adp/baseline_adp for val in best_adp_over_time]
    plt.plot(episodes_x, clean_best_adp, label='Best ADP Over Time', color='blue', linewidth=2.5)

    valid_adps = [(idx, val/baseline_adp) for idx, val in enumerate(episode_adp) if val != float('inf')]
    if valid_adps:
        x_vals, y_vals = zip(*valid_adps)
        plt.scatter(x_vals, y_vals, color='red', alpha=0.3, label='Episode Sampled ADP', s=15)

    plt.axhline(y=baseline_adp/baseline_adp, color='green', linestyle='--', linewidth=2, label='Baseline ADP')

    plt.title(f"Contextual DQN ADP Optimization\nState: Feature Aware (Design: {design})")
    plt.xlabel("Training Episodes")
    plt.ylabel("Area-Delay Product (ADP)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.text(0.8, 0.98, f'Training Time: {runtime:.2f}s\nBest ADP: {clean_best_adp[-1]:.2f}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    try:
        design_name = design.split('/')[1].split('.')[0]
    except IndexError:
        design_name = os.path.basename(design).split('.')[0]
        
    try:
        lib_name = lib_origin[:-4].split('/')[-1]
    except Exception:
        lib_name = lib_origin[:-4]

    output_path = f"random_test/dqn_{num_episodes}_{design_name}_{sample_gate}_{lib_name}_name.png"
    plt.savefig(output_path, dpi=300)
    print(f">> Visualization successfully saved to {output_path}")