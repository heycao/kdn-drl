import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.layer_norm = nn.LayerNorm(out_features)

    def forward(self, x, adj):
        """
        x: (batch_size, num_nodes, in_features)
        adj: (batch_size, num_nodes, num_nodes)
        """
        # A * X
        out = torch.bmm(adj, x)
        # (A * X) * W
        out = self.linear(out)
        # LayerNorm
        out = self.layer_norm(out)
        return out

class GCNFeatureExtractor(BaseFeaturesExtractor):
    """
    Action-Centric Feature Extractor using GCN.
    
    Architecture:
    1. Node Embedding Phase (GCN): 
       - Constructs Adjacency from `edge_endpoints` * `edge_features['is_active']`.
       - Input Features: Traffic aggregation.
       - Output: Node Embeddings (N x Hidden).
       
    2. Edge Gathering Phase:
       - For each action k (Edge k):
         - Gather Node Embeddings [h_u, h_v]
         - Concatenate with specific Edge Features [Capacity, Util, IsActive]
       - Output: Edge Embeddings (K x EdgeHidden).
       
    3. Global Features:
       - Extract path statistics from the Edge List directly.
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 64, 
                 hidden_dim: int = 64, n_layers: int = 2, out_dim: int = None):
        
        # Determine K (Simulating Action Space Size) and N
        # We assume K is fixed based on the observation space definition
        k = observation_space["edge_features"].shape[0]
        n_traffic = observation_space["traffic_demand"].shape[0]
        
        # Edge Embedding Dim:
        # Node_u (Hidden) + Node_v (Hidden) + Edge_Feats (4)
        self.edge_embedding_dim = hidden_dim * 2 + 4
        
        # Final Readout Dim:
        # Flatten(EdgeEmbed) + GlobalFeatures(6)
        # Global: BgMax, MeanUtil, MaxCap, ActiveRatio, MaxAvgLambda, TotalTraffic
        self.post_concat_dim = self.edge_embedding_dim * k + 6
        
        super_features_dim = out_dim if out_dim is not None else self.post_concat_dim
        
        super(GCNFeatureExtractor, self).__init__(observation_space, super_features_dim)
        
        self.n = n_traffic
        self.k = k
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.out_dim = out_dim
        
        # Node Input Dim:
        # 1. Is Destination (1)
        # 2. Is Source (1)
        # 3. Is Visited (1)
        # 4. Target Demand (Out) (1)
        # 5. Target Demand (In) (1)
        # 6. Traffic Out Sum (1)
        # 7. Traffic In Sum (1)
        # 8. Capacity Out Sum (1)
        # 9. Capacity In Sum (1)
        input_dim = 9
        
        self.gcn_layers = nn.ModuleList()
        # First layer
        self.gcn_layers.append(GCNLayer(input_dim, hidden_dim))
        
        # Subsequent layers
        for _ in range(n_layers - 1):
            self.gcn_layers.append(GCNLayer(hidden_dim, hidden_dim))
        
        # Final flattening: K * EdgeDim -> features_dim
        self.flatten = nn.Flatten()
        
        # Final Readout Projection
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
        # traffic_demand: (B, N, N)
        # edge_endpoints: (B, K, 2)
        # edge_features: (B, K, 3) -> [Cap, Util, IsActive]
        # path: (B, T)
        
        traffic = observations["traffic_demand"]
        edge_endpoints = observations["edge_endpoints"].long()
        edge_features = observations["edge_features"]
        path = observations["path"].long()
        
        B, N, _ = traffic.shape
        K = edge_features.shape[1]
        device = traffic.device
        
        # Extract Destination from Path
        # Destination is the last non-negative element in path
        # path: [src, n1, n2, ..., dst, -1, -1]
        mask = (path != -1)
        lens = mask.sum(dim=1) # (B,)
        batch_indices = torch.arange(B, device=device)
        safe_lens = torch.clamp(lens - 1, min=0)
        # Destination index is path[i, len[i]-1]
        dest = path[batch_indices, safe_lens].unsqueeze(1) # (B, 1)
        
        # 2. Reconstruct Adjacency Matrix (B, N, N)
        # Adj[u, v] = Sum(IsActive) (Handling parallel edges by accumulation)
        
        # Determine batch offsets for scatter
        batch_ids = torch.arange(B, device=device).view(B, 1).expand(B, K)
        u_indices = edge_endpoints[:, :, 0]
        v_indices = edge_endpoints[:, :, 1]
        
        flat_indices = batch_ids * (N * N) + u_indices * N + v_indices # (B * K)
        flat_values = edge_features[:, :, 2].reshape(-1) # (B * K)
        
        flat_adj = torch.zeros(B * N * N, device=device)
        flat_adj.scatter_add_(0, flat_indices.view(-1), flat_values)
        
        adj = flat_adj.view(B, N, N)
        
        # 2.1 Capacity Matrix Reconstruction (for features)
        # Similar to adj but weigh by Capacity (feat 0) * IsActive (feat 2)
        flat_cap_vals = (edge_features[:, :, 0] * edge_features[:, :, 2]).reshape(-1)
        flat_cap_mat = torch.zeros(B * N * N, device=device)
        flat_cap_mat.scatter_add_(0, flat_indices.view(-1), flat_cap_vals)
        cap_mat = flat_cap_mat.view(B, N, N)
        
        # 2.2 Global GCN Normalization
        eye = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)
        # Standard GCN normalization: D^-1/2 * (A + I) * D^-1/2
        adj_hat = adj + eye
        degree = adj_hat.sum(dim=2)
        deg_inv_sqrt = degree.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        deg_inv_sqrt_mat = torch.diag_embed(deg_inv_sqrt)
        norm_adj = torch.bmm(torch.bmm(deg_inv_sqrt_mat, adj_hat), deg_inv_sqrt_mat)

        # 3. Construct Node Features
        
        # Feature: Is Destination
        dest_indices = dest.squeeze(1).clone()
        dest_indices[dest_indices < 0] = 0
        is_dest = F.one_hot(dest_indices, num_classes=self.n).float().unsqueeze(2) # (B, N, 1)

        # Feature: Is Source (path[0] is source in DeflationEnv)
        src_indices = path[:, 0].clone()
        src_indices[src_indices < 0] = 0
        is_source = F.one_hot(src_indices, num_classes=self.n).float().unsqueeze(2) # (B, N, 1)

        # Feature: Is Visited
        path_safe = path.clone()
        path_safe[path < 0] = 0
        path_one_hot = F.one_hot(path_safe, num_classes=self.n).float() # (B, T, N)
        mask_t = (path != -1).float().unsqueeze(2)
        path_one_hot = path_one_hot * mask_t
        is_visited = path_one_hot.sum(dim=1).clamp(0, 1).unsqueeze(2)

        # Feature: Target Demand (Demand specifically for this src-dst pair)
        # Extract traffic[src, dst] for each batch
        target_demand_val = traffic[batch_indices, src_indices, dest.squeeze(1)].unsqueeze(1).unsqueeze(2) # (B, 1, 1)
        # Broadcast to src and dst nodes
        target_demand_node = torch.zeros(B, N, 2, device=device) # [From_Src, To_Dst]
        target_demand_node[batch_indices, src_indices, 0] = target_demand_val.squeeze()
        target_demand_node[batch_indices, dest.squeeze(1), 1] = target_demand_val.squeeze()
        
        # Apply scaling to numerical inputs
        traffic_scaled = torch.log1p(traffic)
        cap_mat_scaled = torch.log1p(cap_mat)
        target_demand_node = torch.log1p(target_demand_node)

        # Feature: Traffic Aggregations (on scaled traffic)
        traffic_out_sum = traffic_scaled.sum(dim=2).unsqueeze(2)
        traffic_in_sum = traffic_scaled.sum(dim=1).unsqueeze(2)

        # Feature: Capacity Aggregations (on scaled capacity)
        cap_out_sum = cap_mat_scaled.sum(dim=2).unsqueeze(2)
        cap_in_sum = cap_mat_scaled.sum(dim=1).unsqueeze(2)

        # Concat Node Features (B, N, 9)
        x = torch.cat([
            is_dest, is_source, is_visited, 
            target_demand_node, # (B, N, 2)
            traffic_out_sum, traffic_in_sum,
            cap_out_sum, cap_in_sum
        ], dim=2)
        
        # 4. Run GCN
        for layer in self.gcn_layers:
            x = F.relu(layer(x, norm_adj)) # (B, N, Hidden)
            
        # 5. Edge Gathering (Action-Centric Embeddings)
        
        def batch_gather(tensor, indices):
            # tensor: (B, N, F)
            # indices: (B, K)
            F_dim = tensor.shape[2]
            indices_expanded = indices.unsqueeze(2).expand(-1, -1, F_dim)
            return torch.gather(tensor, 1, indices_expanded)
            
        h_u = batch_gather(x, u_indices) # (B, K, H)
        h_v = batch_gather(x, v_indices) # (B, K, H)
        
        # Apply log-scaling to edge features as well (Cap and Util)
        # edge_features: [Cap, Util, IsActive, IsInPath]
        edge_features_scaled = edge_features.clone()
        edge_features_scaled[:, :, 0] = torch.log1p(edge_features[:, :, 0])
        edge_features_scaled[:, :, 1] = torch.log1p(edge_features[:, :, 1])
        
        # Concatenate with raw (now scaled) edge features
        edge_embeddings = torch.cat([h_u, h_v, edge_features_scaled], dim=2) # (B, K, 2H+4)
        
        # 6. Global Features and Readout
        # Readout: Flatten edge embeddings to preserve spatial information
        e_flat = edge_embeddings.view(edge_embeddings.size(0), -1) # (B, K * (2H+4))
        
        # Other Global features (Keep these as they provide good context)
        # Note: We still calculate these for global context summary
        global_bg_max = edge_features[:, :, 1].max(dim=1)[0].unsqueeze(1) # (B, 1)
        global_mean_util = edge_features[:, :, 1].mean(dim=1).unsqueeze(1)
        global_max_cap = torch.log1p(edge_features[:, :, 0].max(dim=1)[0]).unsqueeze(1)
        active_ratio = edge_features[:, :, 2].mean(dim=1).unsqueeze(1)
        
        max_avg_lambda = torch.log1p(observations["maxAvgLambda"]) # (B, 1)
        total_traffic = torch.log1p(observations["traffic_demand"].sum(dim=(1, 2))).unsqueeze(1) # (B, 1)
        
        # Final feature vector
        out = torch.cat([e_flat, global_bg_max, global_mean_util, global_max_cap, active_ratio, max_avg_lambda, total_traffic], dim=1)
        
        if self.projection is not None:
            out = self.projection(out)
            
        return out
