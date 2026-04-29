import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
import os
import random
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# Importing the CircuitGNN class from your local circuit_gnn.py file
from circuit_gnn import CircuitGNN

# ==========================================
# 1. MULTI-FORMAT CONTRASTIVE DATASET (RANDOMIZED)
# ==========================================
class ContrastiveCircuitDataset(torch.utils.data.Dataset):
    """
    Handles .bench, .blif, and .v files.
    Creates Positive Pairs by generating TWO completely random, 
    structurally different (but logically identical) variations.
    The original structural design is NOT used during training.
    """
    def __init__(self, data_dir):
        self.supported_exts = {'.bench', '.blif', '.v', '.verilog', '.edif'}
        self.files = [
            os.path.join(data_dir, f) for f in os.listdir(data_dir) 
            if Path(f).suffix.lower() in self.supported_exts
        ]
        
        # Pool of logic-preserving transformations
        self.abc_pool = [
            "rewrite", "rewrite -z", 
            "refactor", "refactor -z", 
            "resub", "resub -z", 
            "balance", "fraig"
        ]

        if not self.files:
            print(f">> Warning: No supported files found in {data_dir}")

    def __len__(self):
        return len(self.files)

    def _get_abc_read_cmd(self, file_path):
        ext = Path(file_path).suffix.lower()
        if ext == '.bench': return "read_bench"
        if ext in {'.v', '.verilog'}: return "read_verilog"
        if ext == '.blif': return "read_blif"
        return "read"

    def _generate_random_script(self, min_commands=3, max_commands=20):
        """Generates a random sequence of ABC optimization commands."""
        # Pick 3 to 10 commands WITH replacement (allows repeating commands like 'rewrite; rewrite')
        num_ops = random.randint(min_commands, max_commands)
        selected_ops = random.choices(self.abc_pool, k=num_ops)
        return "; ".join(selected_ops)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        read_cmd = self._get_abc_read_cmd(file_path)
        
        # We now generate two completely different variations
        aag_pos1 = str(Path(file_path).with_suffix(".pos1.aig"))
        aag_pos2 = str(Path(file_path).with_suffix(".pos2.aig"))

        file_path_wsl = CircuitGNN._to_wsl_path(file_path)
        aag_pos1_wsl = CircuitGNN._to_wsl_path(aag_pos1)
        aag_pos2_wsl = CircuitGNN._to_wsl_path(aag_pos2)
        
        # Create two unique random scripts
        script1 = self._generate_random_script()
        script2 = self._generate_random_script()
        
        # 1. Generate Variation A
        os.system(f"wsl abc -c \"{read_cmd} '{file_path_wsl}'; strash; {script1}; write_aiger '{aag_pos1_wsl}'\"")
        
        # 2. Generate Variation B
        os.system(f"wsl abc -c \"{read_cmd} '{file_path_wsl}'; strash; {script2}; write_aiger '{aag_pos2_wsl}'\"")
        
        # 3. Parse into Graph Data
        data_pos1 = CircuitGNN._parse_aag(aag_pos1)
        data_pos2 = CircuitGNN._parse_aag(aag_pos2)
        
        # Cleanup temporary files
        for f in [aag_pos1, aag_pos2]:
            if os.path.exists(f): os.remove(f)
        
        # Return the two random variations. The original structure is gone.
        return data_pos1, data_pos2

# ==========================================
# 2. TRAINING LOGIC
# ==========================================
import random # Make sure this is imported at the top

def train_gnn(data_dir, epochs=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f">> Training initiated on: {device}")
    
    model = CircuitGNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    dataset = ContrastiveCircuitDataset(data_dir)
    
    # The margin dictates how far apart negative samples should be pushed.
    # 1.0 is a standard starting point.
    margin = 1.0 
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_items = len(dataset)
        
        for i in range(num_items):
            # 1. Get the Positive Pair (Variations of the same circuit)
            pos1, pos2 = dataset[i]
            pos1, pos2 = pos1.to(device), pos2.to(device)
            
            # 2. Get a Negative Sample (A completely different circuit)
            # Pick a random index that is NOT the current circuit 'i'
            neg_idx = random.choice([x for x in range(num_items) if x != i])
            neg_pos1, _ = dataset[neg_idx] # We only need one variation of the negative
            neg = neg_pos1.to(device)
            
            optimizer.zero_grad()
            
            # Extract signatures
            z_1 = model(pos1)
            z_2 = model(pos2)
            z_neg = model(neg)
            
            # 3. Calculate Distances
            # Distance between identical logic (should be small)
            distance_positive = F.mse_loss(z_1, z_2) 
            
            # Distance between different logic (should be large)
            distance_negative = F.mse_loss(z_1, z_neg) 
            
            # 4. Triplet Loss Calculation
            # Loss = max(0, distance_positive - distance_negative + margin)
            loss = F.relu(distance_positive - distance_negative + margin)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / num_items if num_items > 0 else 0
        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {avg_loss:.6f}")
    
    model.save_weights("gnn_trained.pth")
    return model

# ==========================================
# 3. VISUALIZATION (t-SNE)
# ==========================================
def visualize_results(model, data_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    signatures = []
    names = []
    
    # File discovery logic
    supported_exts = {'.bench', '.blif', '.v', '.verilog', '.edif'}
    valid_files = [f for f in os.listdir(data_dir) if Path(f).suffix.lower() in supported_exts]
    helper = ContrastiveCircuitDataset(data_dir)

    print(f">> Generating signatures for {len(valid_files)} designs...")
    for f in valid_files:
        path = os.path.join(data_dir, f)
        read_cmd = helper._get_abc_read_cmd(path)
        temp_aag = str(Path(path).with_suffix(".viz.aig"))
        
        path_wsl = CircuitGNN._to_wsl_path(path)
        temp_aag_wsl = CircuitGNN._to_wsl_path(temp_aag)
        
        # Generate standard AIG for the benchmark
        os.system(f"wsl abc -c \"{read_cmd} '{path_wsl}'; strash; write_aiger '{temp_aag_wsl}'\"")
        
        if os.path.exists(temp_aag):
            data = CircuitGNN._parse_aag(temp_aag)
            with torch.no_grad():
                sig = model(data.to(device))
            
            # Store the raw, high-dimensional vector
            signatures.append(sig.cpu().detach().numpy().flatten())
            names.append(f)
            os.remove(temp_aag)
        
    if not signatures:
        print(">> Error: No signatures generated.")
        return

    # Convert to numpy array
    sigs_np = np.array(signatures)
    
    # Calculate pairwise cosine similarity between all high-dimensional signatures
    print(">> Calculating Cosine Similarity Matrix...")
    sim_matrix = cosine_similarity(sigs_np)
    
    # Plotting the Heatmap
    plt.figure(figsize=(12, 10))
    
    # Using seaborn for a clean heatmap layout
    sns.heatmap(
        sim_matrix, 
        xticklabels=names, 
        yticklabels=names, 
        cmap="viridis",          
        annot=True,              
        fmt=".3f",               # Show 4 decimal places
        vmin=-1.0,               # Force the minimum color scale to -1.0
        vmax=1.0,                # Force the maximum color scale to 1.0
        linewidths=0.5,          
        cbar_kws={'label': 'Cosine Similarity (1.0 = Identical)'}
    )
    
    plt.title("MapTune: High-Dimensional Logic Fingerprint Similarity", fontsize=15, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.show()

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    DATA_PATH = base_dir / "benchmarks"

    if DATA_PATH.exists():
        trained_model = train_gnn(str(DATA_PATH), epochs=20)
        visualize_results(trained_model, str(DATA_PATH))
    else:
        print(f">> Error: Directory {DATA_PATH} not found.")