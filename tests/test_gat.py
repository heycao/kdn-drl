import pytest
import numpy as np
import torch
import torch.nn.functional as F
import os
import sys
import gymnasium as gym
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.env import KDNEnvinronment
# Import GAT instead of GCN
try:
    from src.gat import GATFeatureExtractor
except ImportError:
    # Allow test file to exist before implementation
    GATFeatureExtractor = None

def import_nx():
    import networkx as nx
    return nx

# --- Shared Mocks ---

class MockEnv:
    def __init__(self, n_nodes=14, max_steps=20):
        self.observation_space = gym.spaces.Dict({
            "traffic_demand": gym.spaces.Box(low=0, high=np.inf, shape=(n_nodes, n_nodes), dtype=np.float32),
            "edge_endpoints": gym.spaces.Box(low=0, high=n_nodes, shape=(10, 2), dtype=int),
            "edge_features": gym.spaces.Box(low=0, high=np.inf, shape=(10, 4), dtype=np.float32),
            "maxAvgLambda": gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "path": gym.spaces.Box(low=-1, high=n_nodes, shape=(max_steps + 1,), dtype=int)
        })

# --- Helper Functions for Integration Tests ---

def collect_data(env, num_samples=1000, mode='random'):
    observations = []
    mlu_targets = []
    max_iters = num_samples * 50
    iters = 0
    
    while len(observations) < num_samples and iters < max_iters:
        iters += 1
        obs, info = env.reset()
        opt_path = env.optimal_path
        
        G = env.sample.topology_object
        src = env.src
        dst = env.dst
        try:
            sp_path = list(next(import_nx().all_shortest_paths(G, src, dst)))
        except:
            sp_path = None
        
        paths_to_add = []
        if mode == 'hard':
            if opt_path and sp_path:
                if len(opt_path) > len(sp_path) or opt_path != sp_path:
                    paths_to_add.append(opt_path)
        else:
            if opt_path: paths_to_add.append(opt_path)
            if sp_path: paths_to_add.append(sp_path)
        
        if not paths_to_add:
            continue
            
        if not paths_to_add:
            continue
            
        base_obs = env._get_obs()
        for path in paths_to_add:
            link_loads = env.sample.calculate_background_loads(src, dst)
            mlu = env.sample.calculate_max_utilization(path, link_loads)
            path_arr = np.full(env.max_steps + 1, -1, dtype=int)
            path_len = min(len(path), env.max_steps + 1)
            path_arr[:path_len] = path[:path_len]
            
            current_obs = {
                "traffic_demand": base_obs["traffic_demand"].copy(),
                "edge_endpoints": base_obs["edge_endpoints"].copy(),
                "edge_features": base_obs["edge_features"].copy(),
                "maxAvgLambda": base_obs["maxAvgLambda"].copy(),
                "path": path_arr
            }
            observations.append(current_obs)
            mlu_targets.append(mlu)
            if len(observations) >= num_samples:
                break
    return observations, mlu_targets

def get_features_batch(extractor, raw_obs):
    batch = {k: [] for k in raw_obs[0].keys()}
    for o in raw_obs:
        for k in batch:
            batch[k].append(o[k])
    for k in batch:
        batch[k] = torch.tensor(np.array(batch[k]), dtype=torch.float32)
    with torch.no_grad():
        feats = extractor(batch)
    return feats.numpy()

# --- Unit Tests: Logic & Structure ---

def test_gat_structure_defaults():
    if GATFeatureExtractor is None: pytest.skip("GAT not implemented")
    env = MockEnv()
    extractor = GATFeatureExtractor(env.observation_space)
    
    assert len(extractor.gat_layers) == 2
    assert extractor.hidden_dim == 128
    # (128*1*2+4)*10 + 6 = 260*10 + 6 = 2606
    expected_dim = 2606
    assert extractor._features_dim == expected_dim
    assert extractor.projection is None

def test_gat_structure_custom_params():
    if GATFeatureExtractor is None: pytest.skip("GAT not implemented")
    env = MockEnv()
    extractor = GATFeatureExtractor(env.observation_space, 
                                    hidden_dim=64, 
                                    n_layers=3, 
                                    out_dim=256)
    
    assert len(extractor.gat_layers) == 3
    assert extractor.hidden_dim == 64
    assert extractor._features_dim == 256
    assert extractor.projection is not None
    
    # Valid path generation
    B, N, K = 4, 14, 10
    path_tensor = torch.full((B, 20), -1, dtype=torch.long)
    path_tensor[:, 0] = torch.randint(0, N, (B,))
    
    obs = {
        "traffic_demand": torch.rand(B, N, N),
        "edge_endpoints": torch.randint(0, N, (B, K, 2)),
        "edge_features": torch.rand(B, K, 4),
        "maxAvgLambda": torch.rand(B, 1),
        "path": path_tensor
    }
    out = extractor(obs)
    assert out.shape == (B, 256)

def test_gat_forward_pass_safety():
    if GATFeatureExtractor is None: pytest.skip("GAT not implemented")
    n_nodes = 5
    n_edges = 10
    max_steps = 5
    mock_env = MockEnv(n_nodes, max_steps)
    extractor = GATFeatureExtractor(mock_env.observation_space, hidden_dim=32, n_layers=2)
    
    B = 2
    obs = {
        "traffic_demand": torch.rand((B, n_nodes, n_nodes)),
        "path": torch.full((B, max_steps + 1), -1, dtype=torch.long),
        "edge_endpoints": torch.randint(0, n_nodes, (B, n_edges, 2)),
        "edge_features": torch.rand((B, n_edges, 4)),
        "maxAvgLambda": torch.rand((B, 1))
    }
    obs["path"][0, 0] = 0
    obs["path"][1, 0] = 0
    obs["path"][1, 1] = 1
    
    output = extractor(obs)
    # (32*1*2+4)*10 + 6 = (64+4)*10 + 6 = 686
    expected_dim = n_edges * (32 * 2 + 4) + 6
    assert output.shape == (B, expected_dim)

# --- Integration Tests ---

@pytest.fixture(scope="module")
def real_env():
    data_dir = 'data/nsfnetbw'
    if not os.path.exists(data_dir):
        if os.path.exists('data/nsfnetbw'):
            data_dir = 'data/nsfnetbw'
        elif os.path.exists('../data/nsfnetbw'):
            data_dir = '../data/nsfnetbw'
    env = KDNEnvinronment(tfrecords_dir=data_dir, traffic_intensity=9)
    return env

def test_gat_probe_accuracy(real_env):
    if GATFeatureExtractor is None: pytest.skip("GAT not implemented")
    observation_space = real_env.observation_space
    
    num_samples = 1500
    train_obs, train_y = collect_data(real_env, num_samples=num_samples, mode='random')
    assert len(train_obs) >= num_samples
    
    extractor = GATFeatureExtractor(observation_space)
    
    X_list = []
    batch_size = 256
    for i in range(0, len(train_obs), batch_size):
        X_list.append(get_features_batch(extractor, train_obs[i:i+batch_size]))
    X = np.concatenate(X_list, axis=0)
    y = np.array(train_y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    r2_test = model.score(X_test, y_test)
    
    print(f"GAT Probe Test R^2: {r2_test:.4f}")
    assert r2_test > 0.8, f"GAT Probe failed with R^2 {r2_test:.4f} < 0.8"

def test_gat_non_shortest_generalization(real_env):
    """
    Verifies that GAT works on 'hard' cases (Optimal != Shortest).
    """
    if GATFeatureExtractor is None: pytest.skip("GAT not implemented")
    observation_space = real_env.observation_space
    extractor = GATFeatureExtractor(observation_space)
    
    # Train on General Data
    train_obs, train_y = collect_data(real_env, num_samples=2000, mode='random')
    
    X_train_list = []
    batch_size = 256
    for i in range(0, len(train_obs), batch_size):
        X_train_list.append(get_features_batch(extractor, train_obs[i:i+batch_size]))
    X_train = np.concatenate(X_train_list)
    y_train = np.array(train_y)
    
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    # Test on Hard Data
    test_obs, test_y = collect_data(real_env, num_samples=200, mode='hard')
    
    if len(test_obs) < 50:
         pytest.skip("Not enough hard samples found in reasonable time")
         
    X_test_list = []
    for i in range(0, len(test_obs), batch_size):
        X_test_list.append(get_features_batch(extractor, test_obs[i:i+batch_size]))
    X_test = np.concatenate(X_test_list)
    y_test = np.array(test_y)
    
    y_pred = model.predict(X_test)
    r2_test = r2_score(y_test, y_pred)
    
    print(f"GAT Hard Set R^2: {r2_test:.4f}")
    assert r2_test > 0.8, f"GAT failed on hard cases. R^2: {r2_test:.4f}"
