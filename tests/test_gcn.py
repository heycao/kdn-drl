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
from src.gcn import GCNFeatureExtractor

def import_nx():
    import networkx as nx
    return nx

# --- Shared Mocks ---

class MockEnv:
    def __init__(self, n_nodes=14, max_steps=20):
        self.observation_space = gym.spaces.Dict({
            "traffic": gym.spaces.Box(low=0, high=np.inf, shape=(n_nodes, n_nodes), dtype=np.float32),
            "topology": gym.spaces.Box(low=0, high=np.inf, shape=(n_nodes, n_nodes), dtype=np.float32),
            "link_utilization": gym.spaces.Box(low=0, high=np.inf, shape=(n_nodes, n_nodes), dtype=np.float32),
            "destination": gym.spaces.Box(low=0, high=n_nodes, shape=(1,), dtype=int),
            "path": gym.spaces.Box(low=-1, high=n_nodes, shape=(max_steps + 1,), dtype=int)
        })

# --- Helper Functions for Integration Tests ---

def collect_data(env, num_samples=1000, mode='random'):
    """
    mode: 'random' -> Collect random paths (Opt and SP) for training
          'hard'   -> Collect ONLY cases where Optimal != Shortest
    """
    observations = []
    mlu_targets = []
    
    # Safety limit
    max_iters = num_samples * 50
    iters = 0
    
    while len(observations) < num_samples and iters < max_iters:
        iters += 1
        obs, info = env.reset()
        
        # 1. Optimal Path
        opt_path = env.optimal_path
        
        # 2. Shortest Path (NetworkX - Hops)
        G = env.sample.topology_object
        src = env.current_node
        dst = env.destination
        try:
            sp_path = list(next(import_nx().all_shortest_paths(G, src, dst)))
        except:
            sp_path = None
        
        paths_to_add = []
        
        if mode == 'hard':
            # Check if opt path is significantly different from SP
            if opt_path and sp_path:
                if len(opt_path) > len(sp_path) or opt_path != sp_path:
                    paths_to_add.append(opt_path)
        else:
            if opt_path: paths_to_add.append(opt_path)
            if sp_path: paths_to_add.append(sp_path)
        
        if not paths_to_add:
            continue
            
        base_obs = env._get_obs()
        
        for path in paths_to_add:
            link_loads = env._calculate_background_loads(src, dst)
            mlu = env._calculate_max_utilization(path, link_loads)
            
            path_arr = np.full(env.max_steps + 1, -1, dtype=int)
            path_len = min(len(path), env.max_steps + 1)
            path_arr[:path_len] = path[:path_len]
            
            current_obs = {
                "destination": base_obs["destination"].copy(),
                "traffic": base_obs["traffic"].copy(),
                "topology": base_obs["topology"].copy(),
                "link_utilization": base_obs["link_utilization"].copy(),
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

def test_gcn_structure_defaults():
    env = MockEnv()
    extractor = GCNFeatureExtractor(env.observation_space)
    
    # Defaults: hidden=128, layers=2, out_dim=None
    # n_layers param was added, checked via len(gcn_layers)
    assert len(extractor.gcn_layers) == 2
    assert extractor.hidden_dim == 128
    # 14 nodes * 128 hidden + 4 global features
    assert extractor._features_dim == 14 * 128 + 4
    assert extractor.projection is None

def test_gcn_structure_custom_params():
    env = MockEnv()
    extractor = GCNFeatureExtractor(env.observation_space, 
                                    hidden_dim=64, 
                                    n_layers=3, 
                                    out_dim=256)
    
    # Layers should be 3
    assert len(extractor.gcn_layers) == 3
    # Hidden dim stored
    assert extractor.hidden_dim == 64
    # Features dim should be equal to out_dim
    assert extractor._features_dim == 256
    # Projection should exist
    assert extractor.projection is not None
    
    # Verify forward pass shape
    B = 4
    N = 14
    obs = {
        "traffic": torch.rand(B, N, N),
        "topology": torch.rand(B, N, N),
        "link_utilization": torch.rand(B, N, N),
        "destination": torch.randint(0, N, (B, 1)),
        "path": torch.randint(-1, N, (B, 20))
    }
    
    out = extractor(obs)
    # Shape should be (B, 256)
    assert out.shape == (B, 256)

def test_current_node_extraction():
    # Setup
    n_nodes = 5
    max_steps = 10
    mock_env = MockEnv(n_nodes, max_steps)
    extractor = GCNFeatureExtractor(mock_env.observation_space, features_dim=64)
    
    # 1. Path Length 1 (Just started: [src, -1, ..., -1])
    path_obs = torch.full((1, max_steps + 1), -1, dtype=torch.long)
    start_node = 0
    path_obs[0, 0] = start_node
    
    # Simulate logic
    mask = (path_obs != -1)
    lens = mask.sum(dim=1)
    batch_indices = torch.arange(1)
    current_node_indices = path_obs[batch_indices, lens - 1]
    
    assert lens[0] == 1
    assert current_node_indices[0] == start_node
    
    # 2. Path Length < Max (Normal op: [src, n1, n2, current, -1, ...])
    path_list = [0, 2, 4, 1]
    path_obs = torch.full((1, max_steps + 1), -1, dtype=torch.long)
    path_obs[0, :len(path_list)] = torch.tensor(path_list)
    
    mask = (path_obs != -1)
    lens = mask.sum(dim=1)
    batch_indices = torch.arange(1)
    current_node_indices = path_obs[batch_indices, lens - 1]
    
    assert lens[0] == 4
    assert current_node_indices[0] == 1

def test_gcn_forward_pass_safety():
    # Integration test with forward pass
    n_nodes = 5
    max_steps = 5
    mock_env = MockEnv(n_nodes, max_steps)
    extractor = GCNFeatureExtractor(mock_env.observation_space, hidden_dim=32, n_layers=2)
    
    # Create dummy observations
    B = 2
    obs = {
        "destination": torch.tensor([[4], [3]], dtype=torch.float32), 
        "path": torch.full((B, max_steps + 1), -1, dtype=torch.float32),
        "topology": torch.rand((B, n_nodes, n_nodes)),
        "traffic": torch.rand((B, n_nodes, n_nodes)),
        "link_utilization": torch.rand((B, n_nodes, n_nodes))
    }
    
    # Fill paths
    obs["path"][0, 0] = 0
    obs["path"][1, 0] = 0
    obs["path"][1, 1] = 1
    
    # Forward check
    output = extractor(obs)
    # Output dim = N*hidden + 4
    expected_dim = n_nodes * 32 + 4
    assert output.shape == (B, expected_dim)

# --- Integration Tests: Probe & Generalization ---

@pytest.fixture(scope="module")
def real_env():
    data_dir = 'data/nsfnetbw'
    if not os.path.exists(data_dir):
        if os.path.exists('data/nsfnetbw'):
            data_dir = 'data/nsfnetbw'
        elif os.path.exists('../data/nsfnetbw'):
            data_dir = '../data/nsfnetbw'
    
    # Use intensity 9 for consistency
    env = KDNEnvinronment(tfrecords_dir=data_dir, traffic_intensity=9)
    return env

def test_gcn_probe_accuracy(real_env):
    """
    Verifies that GCN features are sufficient to predict MLU with high accuracy (R^2 > 0.8).
    This acts as a regression test for the GCN architecture.
    """
    observation_space = real_env.observation_space
    
    # Collect Data
    # 1500 samples
    num_samples = 1500
    train_obs, train_y = collect_data(real_env, num_samples=num_samples, mode='random')
    
    assert len(train_obs) >= num_samples, f"Could not collect enough samples. Got {len(train_obs)}"
    
    # Feature Extraction
    extractor = GCNFeatureExtractor(observation_space)
    
    # Batch processing
    X_list = []
    batch_size = 256
    for i in range(0, len(train_obs), batch_size):
        X_list.append(get_features_batch(extractor, train_obs[i:i+batch_size]))
            
    X = np.concatenate(X_list, axis=0)
    y = np.array(train_y)
    
    # Train Linear Probe
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_test = model.predict(X_test)
    r2_test = r2_score(y_test, y_pred_test)
    
    print(f"GCN Probe Test R^2: {r2_test:.4f}")
    assert r2_test > 0.8, f"GCN Probe failed with R^2 {r2_test:.4f} < 0.8"

def test_gcn_non_shortest_generalization(real_env):
    """
    Verifies that GCN works on 'hard' cases (Optimal != Shortest).
    """
    observation_space = real_env.observation_space
    extractor = GCNFeatureExtractor(observation_space)
    
    # Reuse model from probe test? No, safer to retrain or train fresh to avoid state leakage/deps.
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
    
    print(f"Hard Set R^2: {r2_test:.4f}")
    assert r2_test > 0.8, f"GCN failed on hard cases. R^2: {r2_test:.4f}"
