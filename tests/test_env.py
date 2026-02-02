import pytest
import numpy as np
import networkx as nx
from src.env import KDNEnvinronment
from src.masked_env import MaskedKDNEnv


def test_env_initialization():
    """Verify that the environment initializes correctly with Datanet."""
    env = KDNEnvinronment(tfrecords_dir='data/nsfnetbw', traffic_intensity=9)
    obs, info = env.reset()
    

    assert "destination" in obs
    assert "traffic" in obs
    assert env.reader is not None
    assert env.sample is not None
    assert len(env.edges) > 0

def test_env_edge_action_space():
    """Verify the action space is edge-based."""
    env = KDNEnvinronment(tfrecords_dir='data/nsfnetbw', traffic_intensity=9)
    assert env.action_space.n == len(env.edges)

def test_observation_structure():
    """Verify the observation dictionary structure."""
    env = KDNEnvinronment(tfrecords_dir='data/nsfnetbw', traffic_intensity=9)
    obs, info = env.reset()
    
    assert isinstance(obs["destination"], np.ndarray)

    n = env.sample.get_network_size()
    assert obs["traffic"].shape == (n, n)
    assert "topology" in obs
    assert obs["topology"].shape == (n, n)
    assert "path" in obs
    assert obs["path"].shape == (env.max_steps + 1,)
    assert obs["path"][0] == env.current_node # Initially path contains just current node

def test_datanet_loading():
    """Verify datanetAPI loading works correctly."""
    env = KDNEnvinronment(tfrecords_dir='data/nsfnetbw', traffic_intensity=9)
    
    assert env.reader is not None
    assert env.iterator is not None
    
    obs, info = env.reset()
    
    n = env.sample.get_network_size()

    assert obs["destination"][0] >= 0
    assert obs["destination"][0] < n
    assert env.current_node != obs["destination"][0]


# ===== MaskedEnv Tests =====

def test_masked_env_initialization():
    """Verify MaskedAdaptivePathEnv initializes correctly."""
    env = MaskedKDNEnv(tfrecords_dir='data/nsfnetbw', traffic_intensity=9)
    obs, info = env.reset()
    
    assert hasattr(env, 'action_masks')
    assert env.reader is not None
    assert env.sample is not None

def test_action_masks_shape():
    """Verify action_masks returns correct shape."""
    env = MaskedKDNEnv(tfrecords_dir='data/nsfnetbw', traffic_intensity=9)
    env.reset()
    
    masks = env.action_masks()
    assert isinstance(masks, np.ndarray)
    assert masks.shape == (len(env.edges),)
    assert masks.dtype == bool

def test_action_masks_validity():
    """Verify action_masks correctly identifies valid edges."""
    env = MaskedKDNEnv(tfrecords_dir='data/nsfnetbw', traffic_intensity=9)
    env.reset()
    
    masks = env.action_masks()
    current_node = env.current_node
    
    for idx, (u, v, key) in enumerate(env.edges):
        if u == current_node:
            assert masks[idx] == True
        else:
            assert masks[idx] == False

def test_masked_env_at_least_one_valid_action():
    """Verify there's always at least one valid action."""
    env = MaskedKDNEnv(tfrecords_dir='data/nsfnetbw', traffic_intensity=9)
    
    for _ in range(10):
        env.reset()
        masks = env.action_masks()
        assert masks.any()


# ===== Step & Reward Logic Tests =====

def test_step_path_tracking():
    """Verify that step appends nodes to the path."""
    env = KDNEnvinronment(tfrecords_dir='data/nsfnetbw', traffic_intensity=9)
    env.reset()
    
    current_node = env.current_node
    # Find a valid neighbor
    valid_edge_idx = -1
    next_node = -1
    for idx, (u, v, _) in enumerate(env.edges):
        if u == current_node:
            valid_edge_idx = idx
            next_node = v
            break
    
    assert valid_edge_idx != -1
    
    obs, reward, terminated, truncated, info = env.step(valid_edge_idx)
    
    assert env.current_node == next_node
    # Check if we can access the path if tracked (implementation detail, usually env.path)
    if hasattr(env, 'path'):
        assert env.path[-1] == next_node

def test_step_reach_destination():
    """Verify termination when destination is reached."""
    # We force a simple scenario: src=0, dst=neighbor of 0
    env = KDNEnvinronment(tfrecords_dir='data/nsfnetbw', traffic_intensity=9)
    env.reset()
    
    # Force neighbors
    u = 0
    neighbors = [v for s, v, _ in env.edges if s == u]
    if not neighbors:
        pytest.skip("Node 0 has no neighbors")
        
    v = neighbors[0]
    env.current_node = u
    env.destination = v
    env.path = [u] # Initialize path if needed
    
    # Find action for u->v
    action = -1
    for idx, (s, d, _) in enumerate(env.edges):
        if s == u and d == v:
            action = idx
            break
            
    obs, reward, terminated, truncated, info = env.step(action)
    
    assert terminated == True
    assert env.current_node == v

def test_reward_minimize_mlu():
    """Verify reward is calculated as 1.0 - Agent_MLU."""
    env = KDNEnvinronment(tfrecords_dir='data/nsfnetbw', traffic_intensity=9)
    env.reset()
    
    # Mocking a direct step to destination to trigger reward calc
    u = env.current_node
    
    # We need a valid path to calculate anticipated MLU
    import networkx as nx
    G = env.sample.topology_object
    try:
        # Just pick a neighbor as dest to keep it simple and ensure paths exist
        neighbors = list(G.neighbors(u))
        if not neighbors:
             pytest.skip("Node has no neighbors")
        dst = neighbors[0]
        env.destination = dst
        env.path = [u] # Reset path
        
        # Action to move to dst
        action = -1
        for idx, (s, d, _) in enumerate(env.edges):
            if s == u and d == dst:
                action = idx
                break
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert terminated == True
        assert "mlu" in info
        agent_mlu = info["mlu"]
        
        # Reward = (1.0 - Agent_MLU) - (0.01 * Hops)
        # Since Agent >= Min, Reward is likely <= 1.0.
        # Calculate expected reward manually
        
        # We need to know MLU and hops to be exact, but we can check the formula logic if we trust the env.
        # Alternatively, just check it is <= 1.0 and follows the logic roughly.
        # Let's verify exact match if possible, or at least bound it.
        
        # For this test, let's just assert it is <= 2.0 (1.0 base + 1.0 bonus) and 'mlu' is in info.
        assert reward <= 2.0 + 1e-9
        assert "mlu" in info
        assert "path_length" in info
        
        # Verify formula consistency if possible:
        # reward approx 1.0 - info['mlu'] + (1.0 if is_optimal else 0.0)
        is_optimal = info.get("is_optimal", False)
        expected_reward = 1.0 - info['mlu']
        if is_optimal:
            expected_reward += 1.0
            
        assert abs(reward - expected_reward) < 1e-6
        
    except Exception as e:
        pytest.fail(f"Test failed with error: {e}")


def test_max_steps_truncation():
    """Verify that the environment truncates after max_steps."""
    # Set max_steps to a small number
    env = KDNEnvinronment(tfrecords_dir='data/nsfnetbw', traffic_intensity=9, max_steps=2)
    env.reset()
    
    # Step 1
    u = env.current_node
    neighbors = [idx for idx, (s, d, _) in enumerate(env.edges) if s == u]
    if not neighbors:
        pytest.skip("No neighbors available")
    action = neighbors[0]
    
    _, _, terminated, truncated, _ = env.step(action)
    
    if terminated:
        # If we happened to hit destination or dead end in 1 step
        assert not truncated
        return

    assert not truncated
    
    # Step 2
    u = env.current_node
    neighbors = [idx for idx, (s, d, _) in enumerate(env.edges) if s == u]
    if not neighbors:
        pytest.skip("No neighbors available for step 2")
    action = neighbors[0]
    
    _, _, terminated, truncated, _ = env.step(action)
    
    # Needs to be truncated now (2 >= 2)
    assert truncated



    # Needs to be truncated now (2 >= 2)
    assert truncated


def test_optimal_path_reward():
    """Verify that following the optimal path yields a specific bonus reward."""
    env = KDNEnvinronment(tfrecords_dir='data/nsfnetbw', traffic_intensity=9)
    env.reset()
    
    # We need to access the calculated optimal path
    optimal_path = env.optimal_path
    
    if not optimal_path or len(optimal_path) < 2:
        pytest.skip("Optimal path too short or not found")
        
    # Get the current node (should be start of optimal path)
    current_node = env.current_node
    
    # Ensure current node is start of optimal path (it should be)
    assert current_node == optimal_path[0]
    
    # Take steps until termination along optimal path
    # We already verified optimal path starts at current_node
    
    total_reward = 0.0
    terminated = False
    
    # We follow the optimal path exactly
    # optimal_path is [n0, n1, n2, ... dst]
    # current_node is n0
    
    path_idx = 0
    while not terminated and path_idx < len(optimal_path) - 1:
        u = optimal_path[path_idx]
        v = optimal_path[path_idx+1]
        
        # Find action u->v
        action = -1
        for idx, (s, d, _) in enumerate(env.edges):
            if s == u and d == v:
                action = idx
                break
        
        assert action != -1
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward = reward # Capture the last reward (terminal)
        path_idx += 1
        
    assert terminated, "Did not terminate after following optimal path"
    
    # Check reward
    # The expected terminal reward logic:
    # C. Calculate Reward (Minimize MLU)
    # 2. Bonus: +1.0 because path == optimal_path
    
    assert "is_optimal" in info
    assert info["is_optimal"] == True
    
    agent_mlu = info['mlu']
    
    base_calc = (1.0 - agent_mlu)
    expected_reward = base_calc + 1.0
    
    # Floating point comparison
    assert abs(total_reward - expected_reward) < 1e-6, f"Reward {total_reward} != Expected {expected_reward} (Base {base_calc} + 1.0)"


def test_topology_weighted():
    """Verify that the topology matrix in observation is weighted (bandwidth)."""
    env = KDNEnvinronment(tfrecords_dir='data/nsfnetbw', traffic_intensity=9)
    obs, info = env.reset()
    
    topology = obs['topology']
    n = env.sample.get_network_size()
    
    # Check shape
    assert topology.shape == (n, n)
    
    # Check that it contains values > 1.0 (bandwidths)
    # Binary matrix would only have 0.0 and 1.0. 
    # Bandwidths in this dataset are usually large numbers (e.g. 10000 kbps or similar).
    # We expect at least some edges to have weights > 1.0
    assert np.any(topology > 1.0), "Topology matrix should contain weights > 1.0"
    
    # Check symmetry if graph is undirected, or just check existence of edges matches sample
    G = env.sample.topology_object
    for u, v, data in G.edges(data=True):
        # Directed graph in networkx for this dataset usually
        w = float(data['bandwidth'])
        assert np.isclose(topology[u, v], w), f"Weight mismatch at {u}->{v}: {topology[u, v]} vs {w}"
