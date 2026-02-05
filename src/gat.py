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
        
        n = observation_space["traffic_demand"].shape[0]
        k = observation_space["edge_features"].shape[0]
        
        # Edge Embedding Dim:
        # Node_u (Hidden) + Node_v (Hidden) + Edge_Feats (4)
        self.edge_embedding_dim = hidden_dim * 2 +  head_dim if 'head_dim' in locals() else hidden_dim * 2 + 4 # This is tricky, let's just use fixed logic from GCN
        self.edge_embedding_dim = hidden_dim * 2 + 4
        
        super_features_dim = out_dim if out_dim is not None else (hidden_dim * heads * k + 6) # Simplified to match action-centric
        
        super(GATFeatureExtractor, self).__init__(observation_space, super_features_dim)
        
        self.n = n
        self.k = k
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.out_dim = out_dim
        self.heads = heads
        
        # Input features (Same as GCN)
        input_dim = 9
        
        self.gat_layers = nn.ModuleList()
        # First layer
        self.gat_layers.append(GATLayer(input_dim, hidden_dim, heads=heads, concat=True))
        
        # Subsequent layers
        for i in range(n_layers - 1):
            in_d = hidden_dim * heads
            self.gat_layers.append(GATLayer(in_d, hidden_dim, heads=heads, concat=True))
        
        self.flatten = nn.Flatten()
        
        # Final Readout Dim:
        # Flatten(EdgeEmbed) + GlobalFeatures(6)
        self.post_concat_dim = (hidden_dim * heads * 2 + 4) * k + 6 # Simplified
        
        # Actually GAT doesn't do edge gathering in its current old implementation. 
        # I should probably just make it match GCN's structure roughly but with GAT layers.
        # For conservation of time, I will make it work with the current observation space.
        
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
        # 1. Unpack Observations
        traffic = observations["traffic_demand"]
        edge_endpoints = observations["edge_endpoints"].long()
        edge_features = observations["edge_features"]
        path = observations["path"].long()
        
        B, N, _ = traffic.shape
        K = edge_features.shape[1]
        device = traffic.device
        
        # Extract Destination from Path
        mask = (path != -1)
        lens = mask.sum(dim=1)
        batch_indices = torch.arange(B, device=device)
        safe_lens = torch.clamp(lens - 1, min=0)
        dest = path[batch_indices, safe_lens].unsqueeze(1) # (B, 1)
        
        # 2. Reconstruct Adjacency Matrix (B, N, N)
        batch_ids = torch.arange(B, device=device).view(B, 1).expand(B, K)
        u_indices = edge_endpoints[:, :, 0]
        v_indices = edge_endpoints[:, :, 1]
        
        flat_indices = batch_ids * (N * N) + u_indices * N + v_indices
        flat_values = edge_features[:, :, 2].reshape(-1) # IsActive
        
        flat_adj = torch.zeros(B * N * N, device=device)
        flat_adj.scatter_add_(0, flat_indices.view(-1), flat_values)
        adj = flat_adj.view(B, N, N)
        
        # 2.1 Capacity Matrix Reconstruction
        flat_cap_vals = (edge_features[:, :, 0] * edge_features[:, :, 2]).reshape(-1)
        flat_cap_mat = torch.zeros(B * N * N, device=device)
        flat_cap_mat.scatter_add_(0, flat_indices.view(-1), flat_cap_vals)
        cap_mat = flat_cap_mat.view(B, N, N)
        
        # 3. Construct Node Features
        dest_indices = dest.squeeze(1).clone()
        dest_indices[dest_indices < 0] = 0
        is_dest = F.one_hot(dest_indices, num_classes=self.n).float().unsqueeze(2)

        src_indices = path[:, 0].clone()
        src_indices[src_indices < 0] = 0
        is_source = F.one_hot(src_indices, num_classes=self.n).float().unsqueeze(2)

        path_safe = path.clone()
        path_safe[path < 0] = 0
        path_one_hot = F.one_hot(path_safe, num_classes=self.n).float()
        mask_t = (path != -1).float().unsqueeze(2)
        path_one_hot = path_one_hot * mask_t
        is_visited = path_one_hot.sum(dim=1).clamp(0, 1).unsqueeze(2)

        target_demand_val = traffic[batch_indices, src_indices, dest.squeeze(1)].unsqueeze(1).unsqueeze(2)
        target_demand_node = torch.zeros(B, N, 2, device=device)
        target_demand_node[batch_indices, src_indices, 0] = target_demand_val.squeeze()
        target_demand_node[batch_indices, dest.squeeze(1), 1] = target_demand_val.squeeze()
        
        traffic_scaled = torch.log1p(traffic)
        cap_mat_scaled = torch.log1p(cap_mat)
        target_demand_node = torch.log1p(target_demand_node)

        traffic_out_sum = traffic_scaled.sum(dim=2).unsqueeze(2)
        traffic_in_sum = traffic_scaled.sum(dim=1).unsqueeze(2)
        cap_out_sum = cap_mat_scaled.sum(dim=2).unsqueeze(2)
        cap_in_sum = cap_mat_scaled.sum(dim=1).unsqueeze(2)

        x = torch.cat([
            is_dest, is_source, is_visited, 
            target_demand_node,
            traffic_out_sum, traffic_in_sum,
            cap_out_sum, cap_in_sum
        ], dim=2)
        
        # 4. Run GAT
        for layer in self.gat_layers:
            x = F.elu(layer(x, adj))
            
        # 5. Edge Gathering
        def batch_gather(tensor, indices):
            F_dim = tensor.shape[2]
            indices_expanded = indices.unsqueeze(2).expand(-1, -1, F_dim)
            return torch.gather(tensor, 1, indices_expanded)
            
        h_u = batch_gather(x, u_indices)
        h_v = batch_gather(x, v_indices)
        
        edge_features_scaled = edge_features.clone()
        edge_features_scaled[:, :, 0] = torch.log1p(edge_features[:, :, 0])
        edge_features_scaled[:, :, 1] = torch.log1p(edge_features[:, :, 1])
        
        edge_embeddings = torch.cat([h_u, h_v, edge_features_scaled], dim=2)
        
        # 6. Global Features and Readout
        e_flat = edge_embeddings.view(edge_embeddings.size(0), -1)
        
        global_bg_max = edge_features[:, :, 1].max(dim=1)[0].unsqueeze(1)
        global_mean_util = edge_features[:, :, 1].mean(dim=1).unsqueeze(1)
        global_max_cap = torch.log1p(edge_features[:, :, 0].max(dim=1)[0]).unsqueeze(1)
        active_ratio = edge_features[:, :, 2].mean(dim=1).unsqueeze(1)
        
        max_avg_lambda = torch.log1p(observations["maxAvgLambda"])
        total_traffic = torch.log1p(observations["traffic_demand"].sum(dim=(1, 2))).unsqueeze(1)
        
        out = torch.cat([e_flat, global_bg_max, global_mean_util, global_max_cap, active_ratio, max_avg_lambda, total_traffic], dim=1)
        
        if self.projection is not None:
            out = self.projection(out)
            
        return out
