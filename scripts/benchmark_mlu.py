#!/usr/bin/env python3
"""
Benchmark script to evaluate Maximum Link Utilization (MLU) performance.

Compares trained agents' MLU against shortest path (SP) routing baseline
using TFRecords from the evaluation dataset.
Supports both PPO and MaskablePPO agent types.
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import glob
from pathlib import Path
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from collections import defaultdict

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.env import AdaptivePathEnv
from src.masked_env import MaskedAdaptivePathEnv
from src.kdn import KDN


def load_tfrecord_samples(tfrecord_pattern, num_samples=None):
    """Load traffic samples from TFRecords."""
    files = glob.glob(tfrecord_pattern)
    if not files:
        raise FileNotFoundError(f"No tfrecords found matching pattern: {tfrecord_pattern}")
    
    print(f"Found {len(files)} TFRecord files")
    
    dataset = tf.data.TFRecordDataset(files)
    feature_description = {
        'traffic': tf.io.FixedLenFeature([182], tf.float32),
    }
    
    samples = []
    for raw_record in dataset:
        if num_samples and len(samples) >= num_samples:
            break
        example = tf.io.parse_single_example(raw_record, feature_description)
        samples.append(example['traffic'].numpy())
    
    return np.array(samples)


def flat_to_pair(flat_idx, num_nodes=14):
    """Convert flat traffic matrix index to (src, dst) pair."""
    n = num_nodes
    src = flat_idx // (n - 1)
    dst_offset = flat_idx % (n - 1)
    dst = dst_offset if dst_offset < src else dst_offset + 1
    return src, dst


def evaluate_agent_mlu(agent_path, env, traffic_samples, max_episodes=1000, agent_type='PPO'):
    """Evaluate trained agent's MLU on traffic samples.
    
    Args:
        agent_path: Path to the trained model
        env: Environment instance
        traffic_samples: Traffic matrix samples
        max_episodes: Maximum number of episodes to evaluate
        agent_type: 'PPO' or 'MaskPPO'
    """
    print(f"\nEvaluating {agent_type} agent from: {agent_path}")
    
    if not os.path.exists(agent_path):
        raise FileNotFoundError(f"Agent model not found: {agent_path}")
    
    # Load appropriate model type
    if agent_type == 'MaskPPO':
        model = MaskablePPO.load(agent_path)
        use_masking = True
    else:
        model = PPO.load(agent_path)
        use_masking = False
    
    mlus = []
    path_lengths = []
    success_count = 0
    
    for i, traffic_matrix in enumerate(traffic_samples[:max_episodes]):
        if i % 100 == 0:
            print(f"  Evaluating episode {i}/{min(max_episodes, len(traffic_samples))}...")
        
        # Manually set traffic and pick a random src-dst pair
        env.traffic_matrix = traffic_matrix
        flat_idx = np.random.randint(0, len(traffic_matrix))
        src, dst = flat_to_pair(flat_idx)
        
        env.current_node = src
        env.destination = dst
        env.current_traffic = traffic_matrix[flat_idx]
        env.current_path = [src]
        env.current_mlu = 0.0
        env.current_step = 0
        
        obs = env._get_obs()
        
        # Run episode
        done = False
        truncated = False
        steps = 0
        max_steps = 50
        
        while not done and not truncated and steps < max_steps:
            if use_masking:
                # Get action mask for MaskablePPO
                action_masks = env.action_masks()
                action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
        
        mlus.append(env.current_mlu)
        path_lengths.append(len(env.current_path))
        if done:
            success_count += 1
    
    results = {
        'mlu_mean': np.mean(mlus),
        'mlu_std': np.std(mlus),
        'mlu_median': np.median(mlus),
        'mlu_min': np.min(mlus),
        'mlu_max': np.max(mlus),
        'path_length_mean': np.mean(path_lengths),
        'success_rate': success_count / len(mlus),
        'num_episodes': len(mlus)
    }
    
    return results


def evaluate_baseline_sp(env, traffic_samples, max_episodes=1000):
    """Evaluate baseline shortest path routing MLU."""
    print("\nEvaluating baseline (Shortest Path) routing...")
    
    import networkx as nx
    
    mlus = []
    path_lengths = []
    
    for i, traffic_matrix in enumerate(traffic_samples[:max_episodes]):
        if i % 100 == 0:
            print(f"  Evaluating episode {i}/{min(max_episodes, len(traffic_samples))}...")
        
        # Pick a random src-dst pair
        flat_idx = np.random.randint(0, len(traffic_matrix))
        src, dst = flat_to_pair(flat_idx)
        traffic = traffic_matrix[flat_idx]
        
        # Compute shortest path
        try:
            path = nx.shortest_path(env.graph, source=src, target=dst, weight='weight')
        except nx.NetworkXNoPath:
            continue
        
        # Calculate MLU for this path
        mlu = 0.0
        for j in range(len(path) - 1):
            u, v = path[j], path[j + 1]
            # Find the edge key
            edge_data = env.graph.get_edge_data(u, v)
            if edge_data:
                key = list(edge_data.keys())[0]  # Get first key
                capacity = env.kdn.link_caps.get((u, v, key), 0.0)
                utilization = traffic / capacity if capacity > 0 else 1.0
                mlu = max(mlu, utilization)
        
        mlus.append(mlu)
        path_lengths.append(len(path))
    
    results = {
        'mlu_mean': np.mean(mlus),
        'mlu_std': np.std(mlus),
        'mlu_median': np.median(mlus),
        'mlu_min': np.min(mlus),
        'mlu_max': np.max(mlus),
        'path_length_mean': np.mean(path_lengths),
        'success_rate': 1.0,  # SP always finds a path if one exists
        'num_episodes': len(mlus)
    }
    
    return results


def print_results(results_dict, baseline_results):
    """Print comparison results for multiple agents.
    
    Args:
        results_dict: Dictionary mapping agent names to their results
        baseline_results: Baseline (SP) results
    """
    print("\n" + "="*90)
    print("MLU BENCHMARK RESULTS")
    print("="*90)
    
    # Build header
    header_parts = [f"{'Metric':<30}"]
    for agent_name in sorted(results_dict.keys()):
        header_parts.append(f"{agent_name:<20}")
    header_parts.append(f"{'Baseline (SP)':<20}")
    print(" ".join(header_parts))
    print("-"*90)
    
    metrics = [
        ('Mean MLU', 'mlu_mean'),
        ('Median MLU', 'mlu_median'),
        ('Std Dev MLU', 'mlu_std'),
        ('Min MLU', 'mlu_min'),
        ('Max MLU', 'mlu_max'),
        ('Mean Path Length', 'path_length_mean'),
        ('Success Rate (%)', 'success_rate'),
        ('Episodes Evaluated', 'num_episodes'),
    ]
    
    for label, key in metrics:
        row_parts = [f"{label:<30}"]
        
        for agent_name in sorted(results_dict.keys()):
            agent_val = results_dict[agent_name][key]
            if key == 'success_rate':
                row_parts.append(f"{agent_val*100:.2f}%".ljust(20))
            elif key == 'num_episodes':
                row_parts.append(f"{int(agent_val)}".ljust(20))
            else:
                row_parts.append(f"{agent_val:.4f}".ljust(20))
        
        baseline_val = baseline_results[key]
        if key == 'success_rate':
            row_parts.append(f"{baseline_val*100:.2f}%".ljust(20))
        elif key == 'num_episodes':
            row_parts.append(f"{int(baseline_val)}".ljust(20))
        else:
            row_parts.append(f"{baseline_val:.4f}".ljust(20))
        
        print(" ".join(row_parts))
    
    # Improvement calculation
    print("\n" + "="*90)
    print("PERFORMANCE COMPARISON (vs Baseline)")
    print("="*90)
    
    for agent_name in sorted(results_dict.keys()):
        agent_results = results_dict[agent_name]
        mlu_improvement = ((baseline_results['mlu_mean'] - agent_results['mlu_mean']) / baseline_results['mlu_mean']) * 100
        
        print(f"\n{agent_name}:")
        print(f"  MLU Improvement: {mlu_improvement:+.2f}%")
        if mlu_improvement > 0:
            print(f"  ✅ {agent_name} performs BETTER than baseline (lower MLU)")
        elif mlu_improvement < 0:
            print(f"  ❌ {agent_name} performs WORSE than baseline (higher MLU)")
        else:
            print(f"  ➖ {agent_name} performs SAME as baseline")
    
    print("\n" + "="*90)


def main():
    parser = argparse.ArgumentParser(description="Benchmark MLU performance")
    parser.add_argument("--ppo_path", type=str, default="ppo_adaptive_path.zip", 
                        help="Path to trained PPO agent model")
    parser.add_argument("--maskppo_path", type=str, default=None, 
                        help="Path to trained MaskPPO agent model")
    parser.add_argument("--agent_type", type=str, default="PPO", 
                        choices=['PPO', 'MaskPPO', 'all'],
                        help="Agent type to benchmark: PPO, MaskPPO, or all")
    parser.add_argument("--tfrecords_dir", type=str, default="data/nsfnetbw/tfrecords/evaluate",
                        help="Path to evaluation TFRecords directory")
    parser.add_argument("--traffic_intensity", type=int, default=9,
                        help="Traffic intensity to filter TFRecords")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of traffic samples to load (default: all)")
    parser.add_argument("--max_episodes", type=int, default=1000,
                        help="Maximum episodes to evaluate")
    parser.add_argument("--graph_path", type=str, default="data/nsfnetbw/graph_attr.txt",
                        help="Path to graph topology file")
    
    args = parser.parse_args()
    
    # Build TFRecord pattern
    pattern = f"{args.tfrecords_dir}/results_nsfnetbw_{args.traffic_intensity}_Routing_SP_k_*.tfrecords"
    
    print("="*90)
    print("MLU BENCHMARK - Evaluating Agents vs Baseline (Shortest Path)")
    print("="*90)
    print(f"\nConfiguration:")
    print(f"  Agent type: {args.agent_type}")
    if args.agent_type in ['PPO', 'all']:
        print(f"  PPO model: {args.ppo_path}")
    if args.agent_type in ['MaskPPO', 'all']:
        maskppo_path = args.maskppo_path or "maskppo_adaptive_path.zip"
        print(f"  MaskPPO model: {maskppo_path}")
    print(f"  TFRecords pattern: {pattern}")
    print(f"  Traffic intensity: {args.traffic_intensity}")
    print(f"  Max episodes: {args.max_episodes}")
    
    # Load traffic samples
    print("\nLoading traffic samples...")
    traffic_samples = load_tfrecord_samples(pattern, num_samples=args.num_samples)
    print(f"Loaded {len(traffic_samples)} traffic samples")
    
    # Determine which agents to evaluate
    agents_to_eval = []
    if args.agent_type == 'all':
        agents_to_eval = ['PPO', 'MaskPPO']
    else:
        agents_to_eval = [args.agent_type]
    
    # Collect results for all agents
    results_dict = {}
    
    for agent_type in agents_to_eval:
        if agent_type == 'PPO':
            # Initialize PPO environment
            print(f"\nInitializing environment for PPO...")
            env = AdaptivePathEnv(
                graph_path=args.graph_path,
                tfrecords_dir=args.tfrecords_dir,
                traffic_intensity=args.traffic_intensity
            )
            # Evaluate PPO agent
            results_dict['PPO'] = evaluate_agent_mlu(
                args.ppo_path, env, traffic_samples, args.max_episodes, agent_type='PPO'
            )
        
        elif agent_type == 'MaskPPO':
            # Initialize MaskPPO environment
            print(f"\nInitializing masked environment for MaskPPO...")
            env = MaskedAdaptivePathEnv(
                graph_path=args.graph_path,
                tfrecords_dir=args.tfrecords_dir,
                traffic_intensity=args.traffic_intensity
            )
            # Evaluate MaskPPO agent
            maskppo_path = args.maskppo_path or "maskppo_adaptive_path.zip"
            results_dict['MaskPPO'] = evaluate_agent_mlu(
                maskppo_path, env, traffic_samples, args.max_episodes, agent_type='MaskPPO'
            )
    
    # Evaluate baseline (use regular env, doesn't matter which)
    print("\nInitializing environment for baseline...")
    env = AdaptivePathEnv(
        graph_path=args.graph_path,
        tfrecords_dir=args.tfrecords_dir,
        traffic_intensity=args.traffic_intensity
    )
    baseline_results = evaluate_baseline_sp(env, traffic_samples, args.max_episodes)
    
    # Print comparison
    print_results(results_dict, baseline_results)


if __name__ == "__main__":
    main()
