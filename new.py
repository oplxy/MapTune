import gymnasium as gym
from gymnasium import spaces
import numpy as np
import subprocess
import re
import sys
import random
import os
from collections import deque
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# PyTorch Geometric for the Graph Attention Network (GAT)
try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GATConv, global_mean_pool
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    print("Warning: torch_geometric not found. GNN will use mock signatures.")

# ==========================================
# PHASE 1 & 2: Input, Curriculum & Cell Features
# ==========================================

class CurriculumScheduler:
    """Organizes designs from simple logic cones to full flattened circuits."""
    @staticmethod
    def get_curriculum(full_design_path):
        # In a production flow, logic cones are extracted via ABC 'cone'
        return [
            {"design": full_design_path, "complexity_lvl": 1, "desc": "Sub-Graph Cluster"},
            {"design": full_design_path, "complexity_lvl": 10, "desc": "Full Circuit Design"}
        ]

class CellFeatureExtractor:
    """Parses .lib to extract Physical (Area), Electrical (Drive), and Functional features."""
    def __init__(self, lib_path):
        self.lib_path = lib_path
        self.cell_data = self._parse()

    def _parse(self):
        content = open(self.lib_path).read()
        cell_blocks = re.findall(r'cell\s*\((.*?)\)\s*\{(.*?)\n\s*\}', content, re.DOTALL)
        parsed_cells = []
        for name, body in cell_blocks:
            # 1. Physical: Area
            area = float(re.search(r'area\s*:\s*([\d.]+);', body).group(1))
            
            # 2. Electrical: Drive Strength (max_capacitance of output pin Y)
            max_cap = re.search(r'pin\s*\(Y\).*?max_capacitance\s*:\s*([\d.]+);', body, re.DOTALL)
            drive = float(max_cap.group(1)) if max_cap else 0.1
            
            # 3. Functional: Complexity (Literal count)
            func_match = re.search(r'function\s*:\s*"(.*?)";', body)
            func_str = func_match.group(1) if func_match else "A"
            complexity = len(re.findall(r'[A-D]|[\*\+\!\&\|]', func_str))
            
            parsed_cells.append({
                "name": name, 
                "feats": [area, drive, float(complexity)]
            })
        return parsed_cells

    def get_matrix(self):
        matrix = torch.tensor([c["feats"] for c in self.cell_data], dtype=torch.float32)
        # Standardize for neural network stability
        mean = matrix.mean(dim=0, keepdim=True)
        std = matrix.std(dim=0, keepdim=True) + 1e-6
        return (matrix - mean) / std

class CircuitGNN(nn.Module):
    """GNN that compresses AIG topology into a 1D Circuit Signature."""
    def __init__(self, node_in_dim=4, hidden_dim=64, signature_dim=16):
        super(CircuitGNN, self).__init__()
        if not PYG_AVAILABLE: return
        self.conv1 = GATConv(node_in_dim, hidden_dim, heads=2, concat=True)
        self.conv2 = GATConv(hidden_dim * 2, hidden_dim, heads=1, concat=False)
        self.projection = nn.Linear(hidden_dim, signature_dim)

    def forward(self, data=None):
        if not PYG_AVAILABLE or data is None:
            return torch.zeros(16) 
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.projection(x).squeeze()

# ==========================================
# PHASE 3: RL Brain (Contextual Agent)
# ==========================================

class ContextualDDQNNetwork(nn.Module):
    """Fuses Circuit Signature, Cell Knowledge, and Selection State."""
    def __init__(self, sig_dim, cell_dim, num_gates):
        super(ContextualDDQNNetwork, self).__init__()
        # Total input = Signature + Total Library Matrix + Binary Selection Mask
        total_in = sig_dim + cell_dim + num_gates
        self.model = nn.Sequential(
            nn.Linear(total_in, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, num_gates)
        )

    def forward(self, sig, lib_ctx, mask):
        # Concatenate all context into one flat vector
        x = torch.cat([sig, lib_ctx, mask], dim=-1)
        return self.model(x)

class MapTuneAgent:
    def __init__(self, sig_dim, cell_dim, num_gates, lr=0.001, tau=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_gates = num_gates
        self.online = ContextualDDQNNetwork(sig_dim, cell_dim, num_gates).to(self.device)
        self.target = ContextualDDQNNetwork(sig_dim, cell_dim, num_gates).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        self.tau = tau
        self.gamma = 0.99
        self.db_path = "maptune_knowledge.pt"

    def retrieve_memory(self, signature):
        """Phase 3: Checks persistent DB for similar designs to warm-start weights."""
        if not os.path.exists(self.db_path): return False
        db = torch.load(self.db_path, weights_only=False)
        for cluster in db:
            if F.cosine_similarity(signature.unsqueeze(0), cluster["sig"].unsqueeze(0)) > 0.96:
                self.online.load_state_dict(cluster["weights"])
                self.target.load_state_dict(self.online.state_dict())
                print(">> [Memory] Warm-starting from similar design match.")
                return True
        return False

    def select_action(self, sig, lib_ctx, mask, epsilon):
        if random.random() < epsilon:
            return random.choice(np.where(mask == 0)[0])
        with torch.no_grad():
            mask_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(self.device)
            q = self.online(sig.unsqueeze(0), lib_ctx.unsqueeze(0), mask_t).squeeze()
            q[mask == 1] = float('-inf') # Action Masking
            return q.argmax().item()

    def update_batch(self, batch, sig, lib_ctx):
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Prepare Tensors
        s_masks = torch.FloatTensor(np.array(states)).to(self.device)
        n_masks = torch.FloatTensor(np.array(next_states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        
        # Batch context
        b_sig = sig.unsqueeze(0).repeat(len(batch), 1).to(self.device)
        b_lib = lib_ctx.unsqueeze(0).repeat(len(batch), 1).to(self.device)

        # DDQN Update Logic
        curr_q = self.online(b_sig, b_lib, s_masks).gather(1, actions).squeeze(1)
        next_actions = self.online(b_sig, b_lib, n_masks).argmax(1).unsqueeze(1)
        next_q = self.target(b_sig, b_lib, n_masks).gather(1, next_actions).squeeze(1)
        target_q = rewards + (self.gamma * next_q * (1 - dones))

        loss = F.mse_loss(curr_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft Update Target
        for tp, op in zip(self.target.parameters(), self.online.parameters()):
            tp.data.copy_(self.tau * op.data + (1.0 - self.tau) * tp.data)

# ==========================================
# PHASE 4: Environment (ABC Evaluator)
# ==========================================

class MapTuneEnv(gym.Env):
    def __init__(self, genlib, lib_path, design, n_samples):
        super().__init__()
        self.genlib = genlib
        self.lib_path = lib_path
        self.design = design
        self.n_samples = n_samples
        self.state = np.zeros(0) 
        self.counter = 0

    def get_reward(self, delay, area, b_delay, b_area):
        """Implements the MapTune ADP reward: R = -ADP."""
        if delay == float('inf'): return -2.0
        norm_adp = (delay / b_delay) * (area / b_area)
        return -norm_adp

    def technology_mapper(self, selected_indices, all_lines):
        # Filter logic to preserve BUF/INV as required by ABC
        f_lines = [l for l in all_lines if l.startswith("GATE") and not any(x in l for x in ["BUF", "INV", "inv", "buf"])]
        f_keep = [l for l in all_lines if any(x in l for x in ["BUF", "INV", "inv", "buf"])]
        
        subset = [f_lines[i] for i in selected_indices] + f_keep
        out_lib = f"{self.lib_path}/maptune_partial.genlib"
        os.makedirs(self.lib_path, exist_ok=True)
        
        with open(out_lib, 'w') as f:
            for l in subset: f.write(l + '\n')
            
        lib_dot_lib = self.genlib[:-7] + ".lib"
        temp_blif = "temp_blifs/maptune_run.blif"
        os.makedirs("temp_blifs", exist_ok=True)

        # Official MapTune ABC Command Sequence
        abc_cmd = f"abc -c 'read {out_lib}; read {self.design}; map -a; write {temp_blif}; read {lib_dot_lib}; read -m {temp_blif}; ps; topo; upsize; dnsize; stime;'"
        try:
            res = subprocess.check_output(abc_cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
            d = float(re.search(r"Delay\s*=\s*([\d.]+)", res).group(1))
            a = float(re.search(r"Area\s*=\s*([\d.]+)", res).group(1))
            return d, a
        except:
            return float('inf'), float('inf')

# ==========================================
# PHASE 5: Execution Loop
# ==========================================

def run_maptune():
    if len(sys.argv) < 4:
        print("Usage: python maptune.py <sample_size> <design.blif> <library.genlib>")
        sys.exit(1)

    k_subset = int(sys.argv[-3])
    design_path = sys.argv[-2]
    genlib_path = sys.argv[-1]
    lib_path = genlib_path[:-7] + ".lib"

    # 1. Initialize Contextual Components
    extractor = CellFeatureExtractor(lib_path)
    cell_names = [c["name"] for c in extractor.cell_data]
    lib_ctx = extractor.get_matrix().view(-1).to("cuda" if torch.cuda.is_available() else "cpu")
    
    gnn = CircuitGNN(signature_dim=16).to(lib_ctx.device)
    agent = MapTuneAgent(16, lib_ctx.numel(), len(cell_names))
    env = MapTuneEnv(genlib_path, "gen_newlibs/", design_path, k_subset)
    
    # Baseline for Reward Normalization
    with open(genlib_path, 'r') as f: all_lines = [l.strip() for l in f]
    print("Evaluating Baseline (Full Library)...")
    b_delay, b_area = env.technology_mapper(range(len([l for l in all_lines if "GATE" in l and "BUF" not in l])), all_lines)

    # 2. Start Curriculum Steps
    curriculum = CurriculumScheduler.get_curriculum(design_path)
    
    for step in curriculum:
        print(f"\n--- Curriculum Phase: {step['desc']} ---")
        env.design = step["design"]
        
        # Phase 2: Topology Sensing
        sig = gnn() 
        
        # Phase 3: Memory Matching
        agent.retrieve_memory(sig)
        
        replay = deque(maxlen=2000)
        best_reward = float('-inf')

        for eps in range(100):
            mask = np.zeros(len(cell_names))
            env.counter = 0
            
            while env.counter < k_subset:
                state_mask = mask.copy()
                action = agent.select_action(sig, lib_ctx, mask, epsilon=0.2)
                
                mask[action] = 1
                env.counter += 1
                
                done = (env.counter == k_subset)
                reward = 0
                
                if done:
                    delay, area = env.technology_mapper(np.where(mask==1)[0], all_lines)
                    reward = env.get_reward(delay, area, b_delay, b_area)
                    
                    if reward > best_reward:
                        best_reward = reward
                        print(f"Eps {eps+1} | New Best ADP: {abs(reward):.4f} | D: {delay} A: {area}")

                replay.append((state_mask, action, reward, mask.copy(), done))

                if len(replay) >= 32:
                    batch = random.sample(replay, 32)
                    agent.update_batch(batch, sig, lib_ctx)
        
        # Phase 5: Knowledge Serialization
        print(f"Step Complete. Saving learned weights for signature {sig[:4]}...")
        knowledge = [{"sig": sig.cpu(), "weights": agent.online.state_dict()}]
        torch.save(knowledge, agent.db_path)

if __name__ == "__main__":
    run_maptune()