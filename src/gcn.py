import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        """
        x: (batch_size, num_nodes, in_features)
        adj: (batch_size, num_nodes, num_nodes)
        """
        # A * X
        out = torch.bmm(adj, x)
        # (A * X) * W
        out = self.linear(out)
        return out

class GCNFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor using GCN for Graph Routing.
    Constructs node features from traffic and state information.
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128, 
                 hidden_dim: int = 128, n_layers: int = 2, out_dim: int = None):
        # We need to calculate the actual feature dim based on N and hidden units
        # but BaseFeaturesExtractor requires features_dim to be set in super().__init__
        
        # Get N from observation space
        n = observation_space["traffic"].shape[0]
        
        # Determine final output dim
        # Flattened dim: N * hidden_dim
        post_concat_dim = n * hidden_dim + 4
        
        super_features_dim = out_dim if out_dim is not None else post_concat_dim
        
        super(GCNFeatureExtractor, self).__init__(observation_space, super_features_dim)
        
        self.n = n
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.out_dim = out_dim
        
        # Node Features Dimension:
        # 1. Is Destination (1)
        # 2. Is Current Node (1)
        # 3. Is Visited (1)
        # 4. Traffic Out (1)
        # 5. Traffic In (1)
        # 6. Capacity Out (1)
        # 7. Capacity In (1)
        # 8. Utilization Out (1)
        # 9. Utilization In (1)
        # 10. BG Util Out (1)
        # 11. BG Util In (1)
        input_dim = 11
        
        self.gcn_layers = nn.ModuleList()
        # First layer
        self.gcn_layers.append(GCNLayer(input_dim, hidden_dim))
        
        # Subsequent layers
        for _ in range(n_layers - 1):
            self.gcn_layers.append(GCNLayer(hidden_dim, hidden_dim))
        
        # Final flattening: N * hidden_dim -> features_dim
        self.flatten = nn.Flatten()
        
        self.post_concat_dim = post_concat_dim
        
        self.projection = None
        if out_dim is not None:
            self.projection = nn.Sequential(
                nn.Linear(self.post_concat_dim, out_dim),
                nn.ReLU()
            )
            self._features_dim = out_dim
        else:
            self._features_dim = self.post_concat_dim
        

    def forward(self, observations):
        # 1. Extract Inputs
        # valid assumptions: batch_first=True in SB3
        
        # topology: (B, N, N)
        adj = observations["topology"]
        # Normalize Adjacency: D^-0.5 * (A + I) * D^-0.5
        # For simplicity in RL, we often use just A + I or D^-1 A
        # Let's do simple A + I and row normalization (Mean aggregation roughly)
        B, N, _ = adj.shape
        device = adj.device
        
        eye = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)
        
        # Normalize adjacency (min-max scaling per batch item) to [0, 1]
        # This prevents large weights (e.g. 10000) from making self-loops (1.0) negligible
        max_val = adj.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        adj = adj / (max_val + 1e-8)
        
        adj_hat = adj + eye
        
        # Degree matrix
        degree = adj_hat.sum(dim=2) # (B, N)
        deg_inv_sqrt = degree.pow(-0.5)
        deg_inv_sqrt[degree == 0] = 0
        
        # D^-0.5 * A_hat * D^-0.5
        # shape ops: (B, N, 1) * (B, N, N) * (B, 1, N)
        deg_inv_sqrt_mat = torch.diag_embed(deg_inv_sqrt)
        norm_adj = torch.bmm(torch.bmm(deg_inv_sqrt_mat, adj_hat), deg_inv_sqrt_mat)
        
        # 2. Construct Node Features (batch-wise)
        traffic = observations["traffic"] # (B, N, N)
        dest = observations["destination"].long() # (B, 1)
        path = observations["path"].long() # (B, MaxSteps)
        
        # Feature: Is Destination
        # Create one-hot (B, N)
        is_dest = F.one_hot(dest.squeeze(1), num_classes=self.n).float().unsqueeze(2) # (B, N, 1)
        
        # Feature: Is Current Node
        # Current node is the last valid node in path. 
        # But path is padded with -1.
        # However, env.path[-1] is the current.
        # In observation, path is [s, n1, n2, ..., current, -1, -1]
        # We need to find the last non-(-1) element or use the fact that we can track it.
        # But wait, 'path' obs is just an array.
        # Let's use argmax checks? No.
        # We can find the length. 
        # Actually, let's look at env.py: "path" contains the sequence.
        # We don't have "current_node" explicit in observation space, but we have it in logic.
        # We need to deduce it.
        # The last element != -1 is the current node.
        
        # Helper to get current node from path (B, T)
        # Argmax of (path != -1) * index?
        mask = (path != -1)
        lens = mask.sum(dim=1) # (B,)
        # current node indices
        batch_indices = torch.arange(B, device=device)
        current_node_indices = path[batch_indices, lens - 1] # (B,)
        
        is_current = F.one_hot(current_node_indices, num_classes=self.n).float().unsqueeze(2) # (B, N, 1)

        # Feature: Is Visited
        # path contains node indices.
        # We want a mask (B, N) where 1 if node in path.
        # scatter add?
        is_visited = torch.zeros(B, N, device=device)
        # We need to scatter ones.
        # path is (B, T). flattened: (B*T)
        # We can mask out -1s
        valid_path = path.clone()
        valid_path[path == -1] = 0 # Dummy index, but we won't scatter 1 there if we handle it
        
        # Using scatter:
        # src: (B, T) -> values 1
        # index: path
        # We need to be careful with -1.
        
        # Loop implementation for safety (batch size is usually small or vectorized logic complex)
        # Vectorized:
        # Create one_hot for each step and sum?
        # path shape (B, T). One hot (B, T, N). Sum over T -> (B, N). Clamp to 1.
        # Handle -1: F.one_hot requires non-negative.
        path_safe = path.clone()
        path_safe[path < 0] = 0 # temporarily map -1 to 0
        path_one_hot = F.one_hot(path_safe, num_classes=self.n).float() # (B, T, N)
        # Zero out the -1 positions
        mask_t = (path != -1).float().unsqueeze(2) # (B, T, 1)
        path_one_hot = path_one_hot * mask_t
        is_visited = path_one_hot.sum(dim=1).clamp(0, 1).unsqueeze(2) # (B, N, 1)
        
        # Feature: Traffic & Capacity (Raw)
        # traffic (B, N, N)
        # raw_adj (B, N, N) - Capacity
        raw_adj = observations["topology"]
        
        # Calculate Edge Utilization Matrix (B, N, N)
        # Add epsilon to capacity to avoid div by zero
        edge_util = traffic / (raw_adj + 1e-9)
        
        # Mask out self-loops or non-existent edges if necessary
        # Assuming traffic is 0 where no edge, utilization will be 0.
        # But if capacity is 0 and traffic is 0, we get 0/0 -> nan.
        # Safe division:
        edge_util = torch.where(raw_adj > 1e-9, edge_util, torch.zeros_like(edge_util))

        # --- Node Features Aggregation ---
        
        # 1. Total Traffic (Sum) - Keeps volume info
        traffic_out_sum = traffic.sum(dim=2).unsqueeze(2)
        traffic_in_sum = traffic.sum(dim=1).unsqueeze(2)
        
        # 2. Total Capacity (Sum) - Keeps capacity info
        cap_out_sum = raw_adj.sum(dim=2).unsqueeze(2)
        cap_in_sum = raw_adj.sum(dim=1).unsqueeze(2)
        
        # 3. Max Link Utilization per Node (Botttleneck detection)
        # Max outgoing utilization from this node (Flow Traffic)
        util_out_max = edge_util.max(dim=2)[0].unsqueeze(2) 
        # Max incoming utilization to this node (Flow Traffic)
        util_in_max = edge_util.max(dim=1)[0].unsqueeze(2)
        
        # Feature: Background Link Utilization (Congestion State)
        # bg_util (B, N, N)
        bg_util = observations["link_utilization"]
        # Max outgoing BG util
        bg_util_out_max = bg_util.max(dim=2)[0].unsqueeze(2)
        # Max incoming BG util
        bg_util_in_max = bg_util.max(dim=1)[0].unsqueeze(2)
        
        # Normalize continuous features
        traffic_out_sum = torch.log1p(traffic_out_sum)
        traffic_in_sum = torch.log1p(traffic_in_sum)
        cap_out_sum = torch.log1p(cap_out_sum)
        cap_in_sum = torch.log1p(cap_in_sum)
        # Util is already a ratio [0, 1+], no log needed usually, or maybe it is?
        # Let's keep it raw as it is the target proxy.
        
        # Concatenate Features
        # (B, N, 11)
        x = torch.cat([
            is_dest, 
            is_current, 
            is_visited, 
            traffic_out_sum, 
            traffic_in_sum,
            cap_out_sum,
            cap_in_sum,
            util_out_max,
            util_in_max,
            bg_util_out_max,
            bg_util_in_max
        ], dim=2)
        
        # 3. GCN Forward
        for layer in self.gcn_layers:
            x = F.relu(layer(x, norm_adj)) # (B, N, hidden)
        
        # 4. Global Path Features (Explicit MLU extraction)
        # We gather the utilization of edges on the path and take the max
        # edge_util: (B, N, N) - representing Total Utilization (Flow + Background)
        # However, we constructed edge_util earlier as just Flow Util.
        # We need Total Util = Flow Util + BG Util
        
        # Re-construct Total Edge Util Matrix
        # flow_util = traffic / (raw_adj + 1e-9) (computed earlier as edge_util)
        # bg_util = observations["link_utilization"]
        total_edge_util = edge_util + bg_util # (B, N, N)
        
        # Vectorized gather
        # Path: (B, T)
        # We need pairs (path[:, i], path[:, i+1])
        T = path.shape[1]
        u = path[:, :-1] # (B, T-1)
        v = path[:, 1:]  # (B, T-1)
        
        # Determine valid transitions (not -1 and not padding transition)
        # valid if u != -1 and v != -1
        input_mask = (u != -1) & (v != -1) # (B, T-1)
        
        # Flatten indices for gather
        # index = u * N + v
        # We need to be careful with -1 indices. map them to 0 temporarily
        u_safe = u.clone()
        v_safe = v.clone()
        u_safe[~input_mask] = 0
        v_safe[~input_mask] = 0
        
        flat_indices = u_safe * self.n + v_safe # (B, T-1)
        flat_util = total_edge_util.view(B, self.n * self.n) # (B, N*N)
        
        path_util_values = torch.gather(flat_util, 1, flat_indices) # (B, T-1)
        
        # Apply mask (set invalid edges to -1 so max ignores them? No, utilizations are >=0. -1 is safe.)
        path_util_vals_masked = path_util_values.clone()
        path_util_vals_masked[~input_mask] = -1.0
        
        # Max Path Util
        max_path_util = path_util_vals_masked.max(dim=1)[0].unsqueeze(1) # (B, 1)
        
        # Mean Path Util (Sum / Count)
        path_lens = input_mask.sum(dim=1).clamp(min=1).float().unsqueeze(1)
        path_util_vals_zeroed = path_util_values.clone()
        path_util_vals_zeroed[~input_mask] = 0.0
        mean_path_util = path_util_vals_zeroed.sum(dim=1).unsqueeze(1) / path_lens # (B, 1)
        
        # Global Background Max (Bottleneck elsewhere)
        # bg_util (B, N, N)
        global_bg_max = bg_util.max(dim=1)[0].max(dim=1)[0].unsqueeze(1) # (B, 1)
        
        # Estimated MLU = Max(Path Max, Global BG Max)
        est_mlu = torch.maximum(max_path_util, global_bg_max)
        
        # 5. Flatten & Concat
        x_flat = self.flatten(x) # (B, N*H)
        
        # Concat global path features
        # (B, N*H + 4)
        out = torch.cat([x_flat, max_path_util, mean_path_util, global_bg_max, est_mlu], dim=1)
        
        if self.projection is not None:
             out = self.projection(out)
        
        return out
