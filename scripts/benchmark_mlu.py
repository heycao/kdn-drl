#!/usr/bin/env python3
"""
Benchmark script to evaluate Maximum Link Utilization (MLU) performance.
"""

import os
import sys
import argparse

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.benchmark import BenchmarkRunner


def main():
    parser = argparse.ArgumentParser(description="Benchmark MLU performance")
    parser.add_argument("--ppo_path", type=str, default="agents/ppo_model", 
                        help="Path to trained PPO agent model")
    parser.add_argument("--maskppo_path", type=str, default="agents/maskppo_model", 
                        help="Path to trained MaskPPO agent model")
    parser.add_argument("--agent_type", type=str, default="MaskPPO", 
                        choices=['PPO', 'MaskPPO', 'all'],
                        help="Agent type to benchmark")
    parser.add_argument("--tfrecords_dir", type=str, default="data/nsfnetbw",
                        help="Path to evaluation parent directory (containing tfrecords folder)")
    parser.add_argument("--traffic_intensity", type=int, default=9,
                        help="Traffic intensity to filter TFRecords")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of network samples to evaluate (default: 10)")
    
    args = parser.parse_args()
    
    runner = BenchmarkRunner(
        tfrecords_dir=args.tfrecords_dir, 
        traffic_intensity=args.traffic_intensity, 
        num_samples=args.num_samples,
        ppo_path=args.ppo_path,
        maskppo_path=args.maskppo_path,
        agent_type=args.agent_type
    )
    
    runner.run()

if __name__ == "__main__":
    main()
