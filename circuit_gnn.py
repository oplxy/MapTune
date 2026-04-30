import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
import subprocess
import os
from pathlib import Path
import re

class CircuitGNN(nn.Module):
    """
    Graph Convolutional Network (GCN) for AIG-based Circuit Analysis.
    Supports multi-format conversion (.v, .bench, .blif), AIGER parsing, 
    and unified weight persistence.
    """
    def __init__(self, node_in_dim=3, hidden_dim=64, signature_dim=24):
        super(CircuitGNN, self).__init__()
        
        # 3-Layer GCN to capture logic topology
        self.conv1 = GCNConv(node_in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Projection layer to generate the 16D Circuit Signature
        self.fc = nn.Linear(hidden_dim, signature_dim)

    def forward(self, data):
        """Processes the graph data and returns a normalized 1D signature."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        
        # Global Readout: Average features across all gates
        sig = global_mean_pool(x, batch)
        
        # Tanh ensures the signature stays within the [-1, 1] range for RL stability
        return torch.tanh(self.fc(sig))

    # ==========================================
    # WEIGHT PERSISTENCE
    # ==========================================

    def save_weights(self, file_path):
        """Saves the model state dictionary."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), file_path)
        print(f">> GNN weights saved to: {file_path}")

    def load_weights(self, file_path, device='cpu'):
        """Loads weights and maps to the appropriate device."""
        if not os.path.exists(file_path):
            print(f">> Warning: {file_path} not found.")
            return False
        
        state_dict = torch.load(file_path, map_location=device, weights_only=True)
        self.load_state_dict(state_dict)
        self.to(device)
        print(f">> GNN weights loaded from: {file_path}")
        return True

    # ==========================================
    # EDA UTILITIES (Multi-Format Support)
    # ==========================================

    @staticmethod
    def file_to_signature(file_path, model, device='cpu'):
        """High-level API: Netlist File -> Signature Vector."""
        aag_path = CircuitGNN._convert_to_aag(file_path)
        if not aag_path: return None
            
        data = CircuitGNN._parse_aag(aag_path).to(device)
        
        model.eval()
        with torch.no_grad():
            signature = model(data)
            
        if os.path.exists(aag_path): os.remove(aag_path)
        return signature

    @staticmethod
    def _get_abc_read_cmd(file_path):
        """Resolves the correct ABC command based on file extension."""
        ext = Path(file_path).suffix.lower()
        if ext == '.bench': return "read_bench"
        if ext in {'.v', '.verilog'}: return "read_verilog"
        if ext == '.blif': return "read_blif"
        return "read"

    @staticmethod
    def _to_wsl_path(file_path):
        """Convert Windows paths like C:\\... into WSL /mnt/c/... paths."""
        path_str = str(file_path).replace("\\", "/")
        if path_str.startswith("/mnt/"):
            return path_str

        match = re.match(r"^([A-Za-z]):/(.*)$", path_str)
        if match:
            drive = match.group(1).lower()
            rest = match.group(2)
            return f"/mnt/{drive}/{rest}"

        return Path(path_str).as_posix()

    @staticmethod
    def _convert_to_aag(file_path):
        """Invokes ABC to convert any netlist to standardized AIGER."""
        if not os.path.exists(file_path): return None

        read_cmd = CircuitGNN._get_abc_read_cmd(file_path)
        aag_path = str(Path(file_path).with_suffix(".temp.aig"))
        file_path_wsl = CircuitGNN._to_wsl_path(file_path)
        aag_path_wsl = CircuitGNN._to_wsl_path(aag_path)
        
        # Standardization: strash ensures we are working with an AIG
        cmd = f"wsl abc -c \"{read_cmd} '{file_path_wsl}'; strash; write_aiger '{aag_path_wsl}'\""
        #print(cmd) # Debug: Show the command being executed
        try:
            subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
            return aag_path
        except subprocess.CalledProcessError:
            print(f">> ABC Error: Failed to process {file_path}")
            return None

    @staticmethod
    def _read_aiger_uint(data, offset):
        value = 0
        shift = 0
        while True:
            byte = data[offset]
            offset += 1
            value |= (byte & 0x7F) << shift
            if not (byte & 0x80):
                break
            shift += 7
        return value, offset

    @staticmethod
    def _parse_aag(file_path):
        """Parses ASCII or binary AIGER files into PyTorch Geometric Data format."""
        with open(file_path, 'rb') as f:
            data = f.read()

        if data.startswith(b'aag '):
            text = data.decode('ascii', errors='replace').splitlines()
            return CircuitGNN._parse_ascii_aag(text)
        if data.startswith(b'aig '):
            return CircuitGNN._parse_binary_aig(data)

        raise ValueError(f"Unsupported AIGER format for file: {file_path}")

    @staticmethod
    def _parse_ascii_aag(lines):
        header = lines[0].strip().split()
        M, I, L, O, A = map(int, header[1:6])
        num_nodes = M + 1
        
        x = torch.zeros((num_nodes, 3), dtype=torch.float)
        edge_index = []

        # Parse Primary Inputs
        for i in range(1, I + 1):
            idx = int(lines[i].strip()) // 2
            x[idx, 0] = 1.0

        # Parse AND Gates (Connectivity: input -> output)
        start_line = 1 + I + L + O
        for i in range(start_line, start_line + A):
            parts = list(map(int, lines[i].strip().split()))
            out_idx, in1_idx, in2_idx = parts[0]//2, parts[1]//2, parts[2]//2
            x[out_idx, 1] = 1.0
            edge_index.append([in1_idx, out_idx])
            edge_index.append([in2_idx, out_idx])

        # Parse Primary Outputs
        out_start = 1 + I + L
        for i in range(out_start, out_start + O):
            idx = int(lines[i].strip()) // 2
            x[idx, 2] = 1.0

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return Data(x=x, edge_index=edge_index, batch=torch.zeros(num_nodes, dtype=torch.long))

    @staticmethod
    def _parse_binary_aig(raw):
        header_end = raw.find(b'\n')
        if header_end == -1:
            raise ValueError("Invalid binary AIGER file: missing header newline")

        header = raw[:header_end].decode('ascii', errors='replace').strip().split()
        M = int(header[1])
        I = int(header[2])
        L = int(header[3])
        O = int(header[4])
        A = int(header[5])
        B = int(header[6]) if len(header) > 6 else 0
        R = int(header[7]) if len(header) > 7 else 0

        num_nodes = M + 1
        x = torch.zeros((num_nodes, 3), dtype=torch.float)
        edge_index = []

        for var in range(1, I + 1):
            x[var, 0] = 1.0

        offset = header_end + 1
        for _ in range(L):
            line_end = raw.find(b'\n', offset)
            if line_end == -1:
                raise ValueError("Invalid binary AIGER file: truncated latch section")
            offset = line_end + 1

        output_literals = []
        for _ in range(O):
            line_end = raw.find(b'\n', offset)
            if line_end == -1:
                raise ValueError("Invalid binary AIGER file: truncated output section")
            output_literals.append(int(raw[offset:line_end].decode('ascii').strip()))
            offset = line_end + 1

        for _ in range(B + R):
            line_end = raw.find(b'\n', offset)
            if line_end == -1:
                raise ValueError("Invalid binary AIGER file: truncated bad/constraint section")
            offset = line_end + 1

        for i in range(A):
            lhs = 2 * (I + L + i + 1)
            delta0, offset = CircuitGNN._read_aiger_uint(raw, offset)
            delta1, offset = CircuitGNN._read_aiger_uint(raw, offset)
            rhs0 = lhs - delta0
            rhs1 = rhs0 - delta1

            out_idx = lhs // 2
            x[out_idx, 1] = 1.0
            edge_index.append([rhs0 // 2, out_idx])
            edge_index.append([rhs1 // 2, out_idx])

        for lit in output_literals:
            idx = lit // 2
            x[idx, 2] = 1.0

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return Data(x=x, edge_index=edge_index, batch=torch.zeros(num_nodes, dtype=torch.long))