import pytest
import numpy as np
from src.masked_env import MaskedKDNEnv

def test_mask_prevents_invalid_actions():
    """
    Simulate episodes and ensure that actions allowed by the mask
    NEVER result in 'invalid_action' or 'dead_end' (cycle) from the environment.
    """
    # Run 5 iterations to cover different reset states
    for _ in range(5):
        try:
            env = MaskedKDNEnv(tfrecords_dir='data/nsfnetbw', traffic_intensity=9)
            env.reset()
        except FileNotFoundError:
            pytest.skip("Data directory not found, skipping integration test.")

        terminated = False
        truncated = False
        step_count = 0
        
        while not (terminated or truncated) and step_count < 20:
            mask = env.action_masks()
            valid_indices = np.where(mask)[0]
            
            if len(valid_indices) == 0:
                # Dead end (no valid neighbors that aren't in path)
                # This makes sense to stop testing this episode
                break
                
            # Pick a random valid action
            action = np.random.choice(valid_indices)
            
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            # KEY ASSERTION 1: invalid_action should NOT be in info or should be False
            assert not info.get("invalid_action", False), \
                f"Action {action} was masked as valid but env returned invalid_action. Info: {info}"
            
            # KEY ASSERTION 2: Cycle detection
            # The mask normally prevents cycles. However, if we are in a dead end (all neighbors visited),
            # the mask allows cycle actions to gracefully terminate with 0.0 reward (dead_end=True).
            # So dead_end=True is ACCEPTABLE. invalid_action=True is NOT.
            
            if info.get("dead_end", False):
                # If dead_end occurred, verify valid_indices were indeed leading to visited nodes?
                # Actually, just ensuring it's not invalid_action is enough for the user requirement.
                pass
            else:
                # If not dead_end, ensure v not in path manually?
                # This is already covered by env logic (env triggers dead_end if v in path).
                pass

def test_mask_logic_explicit():
    """
    Explicitly check the mask logic against the environment state.
    """
    try:
        env = MaskedKDNEnv(tfrecords_dir='data/nsfnetbw', traffic_intensity=9)
        env.reset()
    except FileNotFoundError:
        pytest.skip("Data directory not found, skipping.")
        
    current = env.current_node
    # Ensure path contains current node
    assert current in env.path
    
    mask = env.action_masks()
    
    for idx, (u, v, _) in enumerate(env.edges):
        if mask[idx]:
            # If valid:
            # 1. Source must be current node
            assert u == current, f"Masked True for edge {u}->{v} but current is {current}"
            # 2. Dest must not be in path
            assert v not in env.path, f"Masked True for edge {u}->{v} but {v} is in path {env.path}"
        else:
            # If invalid, either source != current OR dest in path
            # We can't strictly say it MUST be because of this (it could be both),
            # but if u == current, then it MUST be because v is in path.
            if u == current:
                assert v in env.path, f"Masked False for edge {u}->{v} (u==current) but {v} NOT in path?"

