import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import pytest
from src.gcn import GCNFeatureExtractor, GCNLayer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

class TestGCNFeatureExtractor:
    
    @pytest.fixture
    def setup_gcn(self):
        n_nodes = 5
        n_edges = 10
        max_steps = 20
        hidden_dim = 128
        
        # Define observation space
        obs_space = gym.spaces.Dict({
            "traffic_demand": gym.spaces.Box(low=0, high=np.inf, shape=(n_nodes, n_nodes), dtype=np.float32),
            "path": gym.spaces.Box(low=-1, high=n_nodes, shape=(max_steps + 1,), dtype=int),
            "edge_endpoints": gym.spaces.Box(low=0, high=n_nodes, shape=(n_edges, 2), dtype=int),
            "edge_features": gym.spaces.Box(low=0, high=np.inf, shape=(n_edges, 4), dtype=np.float32),
            "maxAvgLambda": gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        })
        
        extractor = GCNFeatureExtractor(obs_space, hidden_dim=hidden_dim, n_layers=2)
        return extractor, n_nodes, n_edges

    def test_gcn_structure_custom_params(self, setup_gcn):
        extractor, n_nodes, n_edges = setup_gcn
        assert len(extractor.gcn_layers) == 2
        # Input dim is now 9 (fixed from 7)
        assert extractor.gcn_layers[0].linear.in_features == 9

    def test_gcn_forward_pass_safety(self, setup_gcn):
        extractor, n_nodes, n_edges = setup_gcn
        B = 2
        
        # Create dummy observations
        traffic = torch.rand(B, n_nodes, n_nodes)
        path = torch.randint(-1, n_nodes, (B, 21))
        # Ensure path has valid start
        path[:, 0] = 0
        
        edge_endpoints = torch.randint(0, n_nodes, (B, n_edges, 2))
        edge_features = torch.rand(B, n_edges, 4) 
        # [Cap, Util, Active, InPath]
        edge_features[:, :, 2] = (edge_features[:, :, 2] > 0.5).float()
        edge_features[:, :, 3] = (edge_features[:, :, 3] > 0.5).float()
        
        obs = {
            "traffic_demand": traffic,
            "path": path,
            "edge_endpoints": edge_endpoints,
            "edge_features": edge_features,
            "maxAvgLambda": torch.tensor([[100.0], [100.0]])
        }
        
        with torch.no_grad():
            full_out = extractor(obs)
            
        # New pooling architecture dimension: (2H + 4) * 2 + 6
        # h=128 (default in test) -> (256+4)*2 + 6 = 520+6 = 526
        expected_dim = (128 * 2 + 4) * 2 + 6
        assert full_out.shape == (B, expected_dim)

    def test_gcn_inactive_edge_sensitivity(self, setup_gcn):
        """
        Verify that the GCN output changes when an edge is marked as inactive.
        This confirms alignment with the action mask concept (inactive edges are handled differently).
        """
        extractor, n_nodes, n_edges = setup_gcn
        B = 1
        
        # 1. Base Observation (All Active)
        traffic = torch.rand(B, n_nodes, n_nodes)
        path = torch.zeros((B, 21), dtype=torch.long)
        path[:, 0] = 0
        path[:, 1:] = -1 # Simple path at node 0
        
        # Fixed edges for reproducibility
        edge_endpoints = torch.randint(0, n_nodes, (B, n_edges, 2))
        # Ensure edge 0 connects relevant nodes
        edge_endpoints[0, 0] = torch.tensor([0, 1])
        
        # All edges active initially
        edge_features = torch.ones(B, n_edges, 4) # [Cap, Util, Active, InPath]
        
        obs_active = {
            "traffic_demand": traffic,
            "path": path,
            "edge_endpoints": edge_endpoints,
            "edge_features": edge_features.clone(),
            "maxAvgLambda": torch.tensor([[100.0]])
        }
        
        # 2. Inactive Observation (Edge 0 Inactive)
        edge_features_inactive = edge_features.clone()
        edge_features_inactive[0, 0, 2] = 0.0 # Mark Edge 0 as inactive
        
        obs_inactive = {
            "traffic_demand": traffic,
            "path": path.clone(), # Ensure path[0] is not -1
            "edge_endpoints": edge_endpoints,
            "edge_features": edge_features_inactive,
            "maxAvgLambda": torch.tensor([[100.0]])
        }
        obs_inactive["path"][0, 0] = 0 # Ensure non-negative for one_hot
        
        with torch.no_grad():
            out_active = extractor(obs_active)
            out_inactive = extractor(obs_inactive)
            
        # 3. Assertions
        # A. Outputs must be different
        assert not torch.allclose(out_active, out_inactive), "GCN output should change when edge active status changes"
        
        # B. Check global feature 'ActiveRatio' (last element)
        # Active Ratio is mean of IsActive column.
        # In active case: 1.0. In inactive case: (N-1)/N = 0.9
        # active_ratio is at index -3 (before max_avg_lambda, total_traffic)
        active_ratio_idx = -3
        assert not torch.isclose(out_active[0, active_ratio_idx], out_inactive[0, active_ratio_idx]), \
            "Global ActiveRatio feature should reflect the change"
            

    # Keeping dynamic logic test for regression safety
    def test_gcn_dynamic_logic(self, setup_gcn):
        extractor, n_nodes, n_edges = setup_gcn
        B = 1
        # Create a linear graph 0-1-2
        edge_endpoints = torch.zeros(B, n_edges, 2, dtype=torch.long)
        edge_endpoints[0, 0] = torch.tensor([0, 1]) # Edge 0: 0->1
        edge_endpoints[0, 1] = torch.tensor([1, 2]) # Edge 1: 1->2
        
        edge_features = torch.zeros(B, n_edges, 4)
        edge_features[:, :, 0] = 10.0 # Cap
        edge_features[:, :, 2] = 1.0  # Active
        edge_features[:, :, 3] = 0.0  # InPath
        
        traffic = torch.zeros(B, n_nodes, n_nodes)
        
        path = torch.full((B, 21), -1, dtype=torch.long)
        path[0, 0] = 0
        
        obs = {
            "traffic_demand": traffic,
            "path": path,
            "edge_endpoints": edge_endpoints,
            "edge_features": edge_features,
            "maxAvgLambda": torch.tensor([[100.0]])
        }
        
        with torch.no_grad():
            full_out = extractor(obs)
            
        # New pooling architecture dimension: (2H + 4) * 2 + 6
        # h=128 (default in test) -> (256+4)*2 + 6 = 520+6 = 526
        expected_dim = (128 * 2 + 4) * 2 + 6
        assert full_out.shape[1] == expected_dim

    def test_gcn_probe_accuracy(self, setup_gcn):
        """
        Verify that GCN features are predictive of MLU using REAL data from Datanet.
        This uses the DeflationEnv to load existing TFRecords.
        """
        import os
        from src.deflation_env import DeflationEnv
        
        # 1. Setup Environment with Real Data
        data_dir = 'data/nsfnetbw'
        if not os.path.exists(data_dir):
            pytest.skip(f"Data directory '{data_dir}' not found. Skipping real data test.")

        # Initialize environment (using same logic as training)
        try:
            env = DeflationEnv(tfrecords_dir=data_dir, traffic_intensity=9, calc_optimal=True)
        except Exception as e:
            pytest.fail(f"Failed to initialize DeflationEnv: {e}")
            
        # 2. Initialize GCN with Env's Observation Space
        obs_space = env.observation_space
        extractor = GCNFeatureExtractor(obs_space, hidden_dim=64, n_layers=2)
        
        # 3. Collect Data (Mix of Shortest Path and Optimal Path)
        num_samples = 1500 # Increased for better R2 convergence
        X_features = []
        y_targets = []
        
        observations = []
        mlu_targets = []
        
        # Collect loop
        while len(observations) < num_samples:
            try:
                env.reset()
            except StopIteration:
                # Should handle end of dataset if num_samples > dataset size
                env.reader = iter(env.reader) # Reset reader? Or just re-init
                # For simplicity, if dataset exhaustion happens, break or handle.
                # DeflationEnv.reset handles iteration internally mostly.
                env.reset()
                
            # 1. Primary path (Shortest Path on Reset)
            observations.append(env._get_obs())
            mlu_targets.append(env.current_mlu)
            
            # 2. Optimal Path (if distinct)
            if env.optimal_path is not None and env.optimal_path != env.current_path:
                # Manually construct obs for optimal path
                base_obs = env._get_obs()
                path_arr = np.full(env.max_steps + 1, -1, dtype=int)
                path_len = min(len(env.optimal_path), env.max_steps + 1)
                path_arr[:path_len] = env.optimal_path[:path_len]
                
                # Verify MLU for optimal path
                bg_loads = env.bg_loads
                opt_mlu = env.sample.calculate_max_utilization(env.optimal_path, bg_loads)
                
                current_obs = {
                    "traffic_demand": base_obs["traffic_demand"].copy(),
                    "edge_endpoints": base_obs["edge_endpoints"].copy(),
                    "edge_features": base_obs["edge_features"].copy(),
                    "path": path_arr,
                    "maxAvgLambda": base_obs["maxAvgLambda"].copy(),
                }
                
                observations.append(current_obs)
                mlu_targets.append(opt_mlu)
        
        observations = observations[:num_samples] # Trim
        mlu_targets = mlu_targets[:num_samples]
        
        # 4. Extract Features
        # Batch processing for efficiency
        batch_size = 32
        for i in range(0, len(observations), batch_size):
            batch_obs = observations[i:i+batch_size]
            
            # Helper to batch dict of numpy arrays
            batch_dict = {k: [] for k in batch_obs[0].keys()}
            for o in batch_obs:
                for k in batch_dict:
                    batch_dict[k].append(o[k])
            
            for k in batch_dict:
                batch_dict[k] = torch.tensor(np.array(batch_dict[k]))
                if k in ["traffic_demand", "edge_features", "maxAvgLambda"]:
                    batch_dict[k] = batch_dict[k].float()
                elif k in ["path", "edge_endpoints"]:
                    batch_dict[k] = batch_dict[k].long()
            
            with torch.no_grad():
                feats = extractor(batch_dict)
            
            X_features.append(feats.numpy())
            
        X = np.concatenate(X_features, axis=0)
        y = np.array(mlu_targets)
        
        # 5. Train Probe
        # 80/20 Split
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        model = Ridge(alpha=0.1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Real Data Probe R^2: {r2:.4f}")
        
        # 6. Assert
        # Target threshold as requested by user
        assert r2 >= 0.85, f"GCN Features predictive power too low on Real Data. R2={r2:.4f}"
