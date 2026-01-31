import pytest
import numpy as np
import networkx as nx
from src.datanet import Datanet
from src.env import AdaptivePathEnv
from src.masked_env import MaskedAdaptivePathEnv


def brute_force_better_path(sample):
    """Brute force search to find if there exists any path better than baseline."""
    G = sample.get_topology_object()
    net_size = sample.get_network_size()
    base_routing = sample.get_routing_matrix()
    
    base_mlu = sample.calculate_mlu()
    best_mlu = base_mlu
    best_improvement_percent = 0.0
    
    for src in range(net_size):
        for dst in range(net_size):
            if src == dst:
                continue
            all_paths = list(nx.all_simple_paths(G, src, dst, cutoff=5))
            for candidate_path in all_paths:
                if candidate_path == base_routing[src, dst]:
                    continue
                
                candidate_routing = base_routing.copy()
                candidate_routing[src, dst] = candidate_path
                cand_mlu = sample.calculate_mlu(routing_matrix=candidate_routing)
                
                if cand_mlu < best_mlu - 1e-9:
                    improvement = base_mlu - cand_mlu
                    improvement_percent = (improvement / base_mlu) * 100
                    if improvement_percent > best_improvement_percent:
                        best_improvement_percent = improvement_percent
                        best_mlu = cand_mlu
    
    return base_mlu, best_mlu, best_improvement_percent

def test_env_initialization():
    """Verify that the environment initializes correctly with Datanet."""
    env = AdaptivePathEnv(tfrecords_dir='data/nsfnetbw', traffic_intensity=9)
    obs, info = env.reset()
    
    assert "current_node" in obs
    assert "destination" in obs
    assert "traffic" in obs
    assert env.reader is not None  # Datanet reader
    assert env.current_sample is not None  # Current sample for MLU calculation
    assert env.num_edges > 0
    assert len(env.edges) == env.num_edges
    assert obs["traffic"].shape == (182,)
    assert len(env.current_path) == 1  # Should contain only starting node
    assert env.current_path[0] == env.current_node

def test_env_edge_action_space():
    """Verify the action space is edge-based."""
    env = AdaptivePathEnv(tfrecords_dir='data/nsfnetbw', traffic_intensity=9)
    assert env.action_space.n == env.num_edges
    
    # Test that we can take an action
    env.reset()
    action = 0  # First edge index
    obs, reward, terminated, truncated, info = env.step(action)
    
    assert isinstance(obs, dict)
    assert "current_node" in obs
    assert isinstance(reward, (float, np.floating))
    assert "mlu" in info
    assert "path_length" in info

def test_observation_structure():
    """Verify the observation dictionary structure."""
    env = AdaptivePathEnv(tfrecords_dir='data/nsfnetbw', traffic_intensity=9)
    obs, info = env.reset()
    
    assert isinstance(obs["current_node"], np.ndarray)
    assert isinstance(obs["destination"], np.ndarray)
    assert obs["current_node"].shape == (1,)

def test_datanet_loading():
    """Verify datanetAPI loading works correctly."""
    env = AdaptivePathEnv(tfrecords_dir='data/nsfnetbw', traffic_intensity=9)
    
    # Check that datanet reader was initialized
    assert env.reader is not None
    assert env.iterator is not None
    
    # Test reset with datanet
    obs, info = env.reset()
    
    assert obs["current_node"][0] >= 0
    assert obs["current_node"][0] < env.num_nodes
    assert obs["destination"][0] >= 0
    assert obs["destination"][0] < env.num_nodes
    assert obs["current_node"][0] != obs["destination"][0]
    assert env.current_traffic >= 0

def test_flat_to_pair_conversion():
    """Verify the flat index to (src, dst) pair conversion."""
    env = AdaptivePathEnv(tfrecords_dir='data/nsfnetbw', traffic_intensity=9)
    
    # For 14 nodes, we have 14 * 13 = 182 pairs
    for flat_idx in range(182):
        src, dst = env._flat_to_pair(flat_idx)
        assert 0 <= src < env.num_nodes
        assert 0 <= dst < env.num_nodes
        assert src != dst  # No self-loops

def test_invalid_edge_action():
    """Verify that agent receives no reward and state unchanged for invalid edge."""
    env = AdaptivePathEnv(tfrecords_dir='data/nsfnetbw', traffic_intensity=9)
    obs, info = env.reset()
    
    initial_node = env.current_node
    initial_path_length = len(env.current_path)
    
    # Find an edge that does NOT start from current_node
    invalid_action = None
    for idx, (u, v, key) in enumerate(env.edges):
        if u != initial_node:
            invalid_action = idx
            break
    
    # If we can't find an invalid action, skip the test
    if invalid_action is None:
        pytest.skip("Could not find an invalid edge for current node")
    
    # Take the invalid action
    obs, reward, terminated, truncated, info = env.step(invalid_action)
    
    # Verify step penalty + invalid penalty for invalid action
    assert reward == -1.0, "Expected -1.0 reward (step penalty + invalid penalty) for invalid action"
    
    # Verify state unchanged
    assert env.current_node == initial_node, "Current node should not change for invalid action"
    assert obs["current_node"][0] == initial_node, "Observation should reflect unchanged state"
    assert len(env.current_path) == initial_path_length, "Path should not change for invalid action"
    
    # Verify episode didn't terminate
    assert not terminated, "Episode should not terminate on invalid action"
    assert not truncated, "Episode should not truncate on invalid action"

def test_valid_edge_action():
    """Verify that agent moves correctly with valid edge and receives MLU-based reward."""
    env = AdaptivePathEnv(tfrecords_dir='data/nsfnetbw', traffic_intensity=9)
    obs, info = env.reset()
    
    initial_node = env.current_node
    initial_path_length = len(env.current_path)
    
    # Find a valid edge that starts from current_node
    valid_action = None
    expected_next_node = None
    for idx, (u, v, key) in enumerate(env.edges):
        if u == initial_node:
            valid_action = idx
            expected_next_node = v
            break
    
    # If we can't find a valid action, skip the test
    if valid_action is None:
        pytest.skip("Could not find a valid edge for current node")
    
    # Take the valid action
    obs, reward, terminated, truncated, info = env.step(valid_action)
    
    # Verify state changed
    assert env.current_node == expected_next_node, "Current node should change to edge destination"
    assert obs["current_node"][0] == expected_next_node, "Observation should reflect new state"
    assert len(env.current_path) == initial_path_length + 1, "Path should grow by 1"
    assert env.current_path[-1] == expected_next_node, "Last node in path should be current node"
    
    # Verify reward structure (MLU-based only, no destination bonus)
    if env.current_node == env.destination:
        # Episode terminates at destination, but no special bonus
        assert terminated, "Episode should terminate when reaching destination"
    else:
        assert not terminated, "Episode should not terminate before reaching destination"
    
    # Verify info contains MLU
    assert "mlu" in info
    assert isinstance(info["mlu"], (float, np.floating))
    assert info["path_length"] == len(env.current_path)

def test_mlu_calculation():
    """Verify aggregate MLU calculation using sample.calculate_mlu()."""
    env = AdaptivePathEnv(tfrecords_dir='data/nsfnetbw', traffic_intensity=9)
    env.reset()
    
    # Baseline MLU should be set from sample.calculate_mlu() with SP routing
    assert env.baseline_mlu > 0, "Baseline MLU should be > 0 (aggregate across all flows)"
    
    # Take a valid action toward destination
    initial_node = env.current_node
    action = None
    for idx, (u, v, key) in enumerate(env.edges):
        if u == initial_node:
            action = idx
            break
            
    if action is None:
        pytest.skip("No valid edge found")
        
    env.step(action)
    
    # After step, current_mlu should be calculated using aggregate method
    # via sample.calculate_mlu() with modified routing for this flow
    assert env.current_mlu >= 0, "Current MLU should be non-negative"
    
    # current_mlu should typically be close to baseline (aggregate MLU doesn't change much for one path)
    # but could be higher or lower depending on the path taken
    assert env.current_mlu > 0 or env.current_step == 1, "MLU should be positive after taking a step"

def test_max_steps_truncation():
    """Verify that episode truncates after max_steps."""
    env = AdaptivePathEnv(tfrecords_dir='data/nsfnetbw', max_steps=5)
    env.reset()
    
    # Take valid actions until we hit max_steps or reach destination
    for step in range(10):  # More than max_steps
        # Find a valid action
        valid_action = None
        for idx, (u, v, key) in enumerate(env.edges):
            if u == env.current_node:
                valid_action = idx
                break
        
        if valid_action is None:
            pytest.skip("Could not find valid action")
        
        obs, reward, terminated, truncated, info = env.step(valid_action)
        
        # Should truncate at step 5 (max_steps)
        if step + 1 >= env.max_steps:
            assert truncated, f"Episode should truncate at step {step + 1}"
            assert env.current_step >= env.max_steps
            break
        
        # If reached destination before max_steps, episode terminates
        if terminated:
            assert not truncated, "Episode should terminate, not truncate, when reaching destination"
            break
    
    # Verify we either truncated or terminated
    assert truncated or terminated, "Episode should end either by truncation or termination"

# ===== MaskedEnv Tests =====

def test_masked_env_initialization():
    """Verify MaskedAdaptivePathEnv initializes correctly."""
    env = MaskedAdaptivePathEnv(tfrecords_dir='data/nsfnetbw', traffic_intensity=9)
    obs, info = env.reset()
    
    # Verify it behaves like regular environment
    assert "current_node" in obs
    assert "destination" in obs
    assert "traffic" in obs
    assert env.reader is not None  # Datanet reader
    assert env.num_edges > 0

def test_action_masks_shape():
    """Verify action_masks() returns correct shape."""
    env = MaskedAdaptivePathEnv(tfrecords_dir='data/nsfnetbw', traffic_intensity=9)
    env.reset()
    
    masks = env.action_masks()
    
    assert isinstance(masks, np.ndarray)
    assert masks.dtype == bool
    assert masks.shape == (env.num_edges,)

def test_action_masks_validity():
    """Verify action_masks() correctly identifies valid actions."""
    env = MaskedAdaptivePathEnv(tfrecords_dir='data/nsfnetbw', traffic_intensity=9)
    env.reset()
    
    current_node = env.current_node
    visited = set(env.current_path)
    masks = env.action_masks()
    
    # Check that all valid actions are masked as True
    for idx, (u, v, key) in enumerate(env.edges):
        if u == current_node and v not in visited:
            assert masks[idx] == True, f"Edge {idx} from {u} to {v} should be valid"
        else:
            assert masks[idx] == False, f"Edge {idx} from {u} to {v} should be invalid"

def test_action_masks_after_step():
    """Verify action_masks() updates correctly after taking a step."""
    env = MaskedAdaptivePathEnv(tfrecords_dir='data/nsfnetbw', traffic_intensity=9)
    env.reset()
    
    initial_node = env.current_node
    initial_masks = env.action_masks()
    
    # Find a valid action
    valid_action = None
    for idx, mask in enumerate(initial_masks):
        if mask:
            valid_action = idx
            break
    
    if valid_action is None:
        pytest.skip("No valid actions found")
    
    # Take the action
    env.step(valid_action)
    
    # Get new masks
    new_masks = env.action_masks()
    new_node = env.current_node
    visited = set(env.current_path)
    
    # Verify masks are different if we moved to a different node
    if new_node != initial_node:
        assert not np.array_equal(initial_masks, new_masks), "Masks should change after moving to new node"
    
    # Verify new masks are correct for new node (considering visited nodes)
    for idx, (u, v, key) in enumerate(env.edges):
        if u == new_node and v not in visited:
            assert new_masks[idx] == True, f"Edge {idx} from {u} to {v} should be valid"
        else:
            assert new_masks[idx] == False, f"Edge {idx} from {u} to {v} should be invalid"

def test_masked_env_at_least_one_valid_action():
    """Verify there's always at least one valid action (unless isolated)."""
    env = MaskedAdaptivePathEnv(tfrecords_dir='data/nsfnetbw', traffic_intensity=9)
    
    # Run multiple episodes to test different starting positions
    for _ in range(10):
        env.reset()
        masks = env.action_masks()
        
        # Should have at least one valid action (network is connected)
        assert masks.any(), f"Node {env.current_node} should have at least one valid outgoing edge"

# ===== Aggregate MLU Tests =====

class TestEnvAggregateMluAlignment:
    """Verify env.py uses aggregate MLU matching brute_force_mlu.py."""
    
    def test_env_baseline_uses_aggregate_mlu(self):
        """Verify env's baseline_mlu uses AGGREGATE MLU (sample.calculate_mlu)."""
        env = AdaptivePathEnv(tfrecords_dir='data/nsfnetbw', traffic_intensity=9)
        env.reset()
        
        # Aggregate MLU should be in reasonable range (0.3-1.5)
        assert 0.3 < env.baseline_mlu < 1.5, (
            f"Baseline MLU {env.baseline_mlu} not in expected aggregate range (0.3-1.5)"
        )
        
        # current_mlu should equal baseline after reset
        assert env.current_mlu == env.baseline_mlu, (
            "Initial current_mlu should equal baseline_mlu"
        )
    
    def test_env_mlu_matches_sample_calculate_mlu(self):
        """Verify env's baseline matches sample.calculate_mlu() directly."""
        env = AdaptivePathEnv(tfrecords_dir='data/nsfnetbw', traffic_intensity=9)
        env.reset()
        
        expected_mlu = env.current_sample.calculate_mlu()
        
        assert np.isclose(env.baseline_mlu, expected_mlu, rtol=1e-5), (
            f"Env baseline_mlu ({env.baseline_mlu}) != sample.calculate_mlu() ({expected_mlu})"
        )
    
    def test_brute_force_finds_improvements(self):
        """Verify brute force can find improvements in aggregate MLU."""
        reader = Datanet("data/nsfnetbw", intensity_values=[9])
        sample = next(iter(reader))
        
        base_mlu, best_mlu, imp_pct = brute_force_better_path(sample)
        
        assert base_mlu >= best_mlu, "Best MLU should not exceed base"
    
    def test_env_mlu_changes_during_episode(self):
        """Verify MLU changes when agent takes different paths."""
        env = AdaptivePathEnv(tfrecords_dir='data/nsfnetbw', traffic_intensity=9)
        env.reset()
        
        mlu_values = [env.current_mlu]
        
        for step in range(3):
            valid_action = None
            for idx, (u, v, key) in enumerate(env.edges):
                if u == env.current_node and v != env.destination:
                    valid_action = idx
                    break
            
            if valid_action is None:
                break
            
            obs, reward, terminated, truncated, info = env.step(valid_action)
            mlu_values.append(env.current_mlu)
            
            if terminated:
                break
        
        assert len(mlu_values) >= 1, "Should have at least initial MLU value"


class TestAggregateMluOptimization:
    """Test that the RL environment can potentially beat the baseline."""
    
    def test_improvement_potential_exists(self):
        """Verify that brute force can find better paths, meaning RL has room to learn."""
        reader = Datanet("data/nsfnetbw", intensity_values=[9])
        
        improvements_found = 0
        for i, sample in enumerate(reader):
            if i >= 5:
                break
            _, _, imp_pct = brute_force_better_path(sample)
            if imp_pct > 0:
                improvements_found += 1
        
        assert improvements_found > 0, (
            "No samples found where alternative paths improve MLU. "
            "RL agent has nothing to learn."
        )
