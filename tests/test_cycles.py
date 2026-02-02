import pytest
import numpy as np
from src.masked_env import MaskedKDNEnv
from src.env import KDNEnvinronment

def test_action_masks_prevent_cycles():
    """Verify action_masks filter out visited nodes."""
    env = MaskedKDNEnv(tfrecords_dir='data/nsfnetbw', traffic_intensity=9)
    env.reset()
    
    # 1. Step to a neighbor
    u = env.current_node
    neighbors_indices = [i for i, (s, d, _) in enumerate(env.edges) if s == u]
    if not neighbors_indices:
        pytest.skip("No neighbors")
        
    action_idx = neighbors_indices[0]
    next_node = env.edges[action_idx][1]
    
    env.step(action_idx)
    
    # Now at next_node. Path is [u, next_node].
    # Check masks from next_node. Edge returning to u should be masked False.
    masks = env.action_masks()
    
    for idx, (s, d, _) in enumerate(env.edges):
        if s == next_node and d == u:
            # This edge points back to start, which is in path. Should be masked.
            assert masks[idx] == False, f"Edge {s}->{d} should be masked as {d} is in path {env.path}"

def test_dead_end_termination():
    """Verify episode terminates with 0 reward if forced to revisit (dead end response)."""
    env = KDNEnvinronment(tfrecords_dir='data/nsfnetbw', traffic_intensity=9)
    env.reset()
    
    # Manually force a revisit scenario (simulate the environment response, independent of masking)
    # Move u -> v
    u = env.current_node
    neighbors_indices = [i for i, (s, d, _) in enumerate(env.edges) if s == u]
    if not neighbors_indices:
        pytest.skip("No neighbors")
    
    action_idx = neighbors_indices[0]
    v = env.edges[action_idx][1]
    env.step(action_idx)
    
    # Now try to move back to u (revisit)
    # Find action v -> u
    back_action_idx = -1
    for i, (s, d, _) in enumerate(env.edges):
        if s == v and d == u:
            back_action_idx = i
            break
            
    if back_action_idx == -1:
        pytest.skip("No back edge")
        
    # Take the back action (revisit u)
    obs, reward, terminated, truncated, info = env.step(back_action_idx)
    
    assert terminated == True
    assert reward == 0.0
    assert info.get("dead_end") == True
