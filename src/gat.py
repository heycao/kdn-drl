import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, heads=1, concat=True, dropout=0.0):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        self.W = nn.Linear(in_features, heads * out_features, bias=False)
        self.a = nn.Parameter(torch.Tensor(heads, 2 * out_features))
        self.leakyrelu = nn.LeakyReLU(0.2)
        
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)

    def forward(self, h, adj):
        """
        h: (B, N, in_features)
        adj: (B, N, N) - Adjacency matrix (can be weighted or binary)
             GAT usually uses binary mask for attention, but we can treat weighted adj as mask?
             Standard GAT: Attention is computed for neighbors.
             Here we mask attention where adj == 0.
        """
        B, N, _ = h.shape
        
        # Linear Transformation
        h_prime = self.W(h).view(B, N, self.heads, self.out_features) # (B, N, H, F)
        
        # Self-attention
        # Prepare for broadcasting
        # We need (B, N, N, H, 2*F) or similar to compute attention logits for all pairs
        # Memory efficient way:
        # e_ij = a^T [Wh_i || Wh_j]
        #      = a_l^T Wh_i + a_r^T Wh_j
        
        # Slice 'a' parameter
        a_l = self.a[:, :self.out_features].unsqueeze(0).unsqueeze(0) # (1, 1, H, F)
        a_r = self.a[:, self.out_features:].unsqueeze(0).unsqueeze(0) # (1, 1, H, F)
        
        # (B, N, H, F) * (1, 1, H, F) -> sum over F -> (B, N, H)
        h_prime_l = (h_prime * a_l).sum(dim=-1)
        h_prime_r = (h_prime * a_r).sum(dim=-1)
        
        # Broadcast add to form (B, N, N, H) matrix of e_ij (before bias/non-linearity)
        # e_ij = h_prime_l[i] + h_prime_r[j]
        e = h_prime_l.unsqueeze(2) + h_prime_r.unsqueeze(1) # (B, N, N, H)
        
        e = self.leakyrelu(e)
        
        # Mask Based on Adjacency
        # adj is (B, N, N). We expand to (B, N, N, H)
        # Assuming adj > 0 means edge exists.
        # Self-loops usually added in GAT.
        
        # Add self-loops if not present?
        # Standard GCN/GAT usually adds self-loops.
        # Let's assume adj has self-loops or we force them.
        # For simplicity, we force self-loops in the mask logic.
        eye = torch.eye(N, device=adj.device).unsqueeze(0).unsqueeze(-1) # (1, N, N, 1)
        adj_mask = (adj.unsqueeze(-1) > 0) | (eye > 0) # (B, N, N, 1)
        
        # Set non-neighbors to -inf
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj_mask, e, zero_vec)
        
        attention = F.softmax(attention, dim=2) # Softmax over neighbors (dim=2 is j)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Aggregation: h_i' = sum_j alpha_ij Wh_j
        # (B, N, N, H) * (B, N, H, F) -> (B, N, H, F)
        # We need to broadcast properly.
        # h_prime: (B, N, H, F). We want (B, 1, N, H, F) for j-index
        h_prime_j = h_prime.unsqueeze(1) 
        
        # attention: (B, N, N, H). Unsqueeze last: (B, N, N, H, 1)
        att_expanded = attention.unsqueeze(-1)
        
        # Weighted sum over j (dim 2)
        h_new = (att_expanded * h_prime_j).sum(dim=2) # (B, N, H, F)
        
        if self.concat:
            # Concat heads: (B, N, H*F)
            return h_new.reshape(B, N, self.heads * self.out_features)
        else:
            # Average heads: (B, N, F)
            return h_new.mean(dim=2)

class GATFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor using GAT (Graph Attention Network).
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128, 
                 hidden_dim: int = 128, n_layers: int = 2, out_dim: int = None, heads: int = 1):
        
        n = observation_space["traffic"].shape[0]
        
        # Output dim logic
        # If concat=True, hidden_dim usually means "per head" or "total"?
        # Let's assume hidden_dim is per-head output size?
        # Or hidden_dim is total? Standard PyG: out_channels is per head.
        # Let's stick to hidden_dim = out_channels per head.
        
        # For simplicity in this env, we might just use heads=1 by default to match GCN size.
        
        super_features_dim = out_dim if out_dim is not None else (n * hidden_dim * heads + 4)
        super(GATFeatureExtractor, self).__init__(observation_space, super_features_dim)
        
        self.n = n
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.out_dim = out_dim
        self.heads = heads
        
        # Input features (Same as GCN)
        input_dim = 11
        
        self.gat_layers = nn.ModuleList()
        # First layer
        self.gat_layers.append(GATLayer(input_dim, hidden_dim, heads=heads, concat=True))
        
        # Subsequent layers
        # Input to next layer is hidden_dim * heads (if previous was concat)
        for i in range(n_layers - 1):
            in_d = hidden_dim * heads
            # Last layer usually averages if it's classification, but here we want embeddings.
            # We can keep concatenating or average.
            # Let's concat for all intermediate, and maybe avg for last?
            # Or concat for all.
            # If we concat, the dimension grows or we keep projection roughly same?
            # Let's keep concat=True for valid representation power.
            self.gat_layers.append(GATLayer(in_d, hidden_dim, heads=heads, concat=True))
        
        self.flatten = nn.Flatten()
        
        # Flattened dim: N * (hidden_dim * heads)
        self.post_concat_dim = n * (hidden_dim * heads) + 4
        
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
        # 1. Extract Inputs (Shared logic with GCN)
        adj = observations["topology"]
        # Normalize? GAT doesn't strictly need Norm Adj like GCN, but scaling helps optimization.
        # However, GAT computes content-based attention. 
        # Raw Adjacency for masking is fine.
        
        B, N, _ = adj.shape
        device = adj.device
        
        # 2. Construct Node Features (Same as GCN)
        traffic = observations["traffic"]
        dest = observations["destination"].long() # (B, 1)
        path = observations["path"].long() # (B, MaxSteps)
        
        # Feature: Is Destination
        is_dest = F.one_hot(dest.squeeze(1), num_classes=self.n).float().unsqueeze(2)
        
        # Feature: Is Current Node
        mask = (path != -1)
        lens = mask.sum(dim=1)
        batch_indices = torch.arange(B, device=device)
        current_node_indices = path[batch_indices, lens - 1]
        is_current = F.one_hot(current_node_indices, num_classes=self.n).float().unsqueeze(2)

        # Feature: Is Visited
        path_safe = path.clone()
        path_safe[path < 0] = 0
        path_one_hot = F.one_hot(path_safe, num_classes=self.n).float()
        mask_t = (path != -1).float().unsqueeze(2)
        path_one_hot = path_one_hot * mask_t
        is_visited = path_one_hot.sum(dim=1).clamp(0, 1).unsqueeze(2)
        
        # Feature: Utilizations
        raw_adj = observations["topology"]
        edge_util = traffic / (raw_adj + 1e-9)
        edge_util = torch.where(raw_adj > 1e-9, edge_util, torch.zeros_like(edge_util))

        traffic_out_sum = torch.log1p(traffic.sum(dim=2).unsqueeze(2))
        traffic_in_sum = torch.log1p(traffic.sum(dim=1).unsqueeze(2))
        cap_out_sum = torch.log1p(raw_adj.sum(dim=2).unsqueeze(2))
        cap_in_sum = torch.log1p(raw_adj.sum(dim=1).unsqueeze(2))
        util_out_max = edge_util.max(dim=2)[0].unsqueeze(2) 
        util_in_max = edge_util.max(dim=1)[0].unsqueeze(2)
        
        bg_util = observations["link_utilization"]
        bg_util_out_max = bg_util.max(dim=2)[0].unsqueeze(2)
        bg_util_in_max = bg_util.max(dim=1)[0].unsqueeze(2)
        
        x = torch.cat([
            is_dest, is_current, is_visited, 
            traffic_out_sum, traffic_in_sum,
            cap_out_sum, cap_in_sum,
            util_out_max, util_in_max,
            bg_util_out_max, bg_util_in_max
        ], dim=2)
        
        # 3. GAT Forward
        for layer in self.gat_layers:
            x = F.elu(layer(x, adj)) # ELU is often used in GAT
        
        # 4. Global Path Features (Same as GCN)
        total_edge_util = edge_util + bg_util
        T = path.shape[1]
        u = path[:, :-1]
        v = path[:, 1:]
        input_mask = (u != -1) & (v != -1)
        u_safe = u.clone()
        v_safe = v.clone()
        u_safe[~input_mask] = 0
        v_safe[~input_mask] = 0
        flat_indices = u_safe * self.n + v_safe
        flat_util = total_edge_util.view(B, self.n * self.n)
        path_util_values = torch.gather(flat_util, 1, flat_indices)
        
        path_util_vals_masked = path_util_values.clone()
        path_util_vals_masked[~input_mask] = -1.0
        max_path_util = path_util_vals_masked.max(dim=1)[0].unsqueeze(1)
        
        path_lens = input_mask.sum(dim=1).clamp(min=1).float().unsqueeze(1)
        path_util_vals_zeroed = path_util_values.clone()
        path_util_vals_zeroed[~input_mask] = 0.0
        mean_path_util = path_util_vals_zeroed.sum(dim=1).unsqueeze(1) / path_lens
        
        global_bg_max = bg_util.max(dim=1)[0].max(dim=1)[0].unsqueeze(1)
        est_mlu = torch.maximum(max_path_util, global_bg_max)
        
        # 5. Connect
        x_flat = self.flatten(x)
        out = torch.cat([x_flat, max_path_util, mean_path_util, global_bg_max, est_mlu], dim=1)
        
        if self.projection is not None:
             out = self.projection(out)
        
        return out
