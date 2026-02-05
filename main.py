#!/usr/bin/env python3
"""
Main entry point for KDN-DRL training and benchmarking.
"""

import argparse
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.trainer import Trainer
from src.benchmark import BenchmarkRunner

def main():
    parser = argparse.ArgumentParser(description="KDN-DRL: Train and Benchmark")
    
    # helper for boolean flag
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # --- Common Arguments ---
    parser.add_argument("--tfrecords_dir", type=str, default="data/nsfnetbw", help="Path to dataset root")
    parser.add_argument("--traffic_intensity", type=int, default=15, help="Traffic intensity filter")
    parser.add_argument("--dataset_name", type=str, default=None, help="Dataset name (optional)")
    parser.add_argument("--model_type", type=str, default="MaskPPO", help="Model type identifier (default: MaskPPO)")
    parser.add_argument("--gnn_type", type=str, default="none", choices=["gcn", "gat", "none"], help="GNN type: 'gcn', 'gat', or 'none'")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, cuda, mps). Auto-detect if None.")

    # --- Training Arguments ---
    parser.add_argument("--total_timesteps", type=int, default=100_000, help="Total training timesteps")
    parser.add_argument("--n_envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--log_interval", type=int, default=1, help="Log interval")
    parser.add_argument("--model_path", type=str, default="final_model", help="Filename for saved model")
    parser.add_argument("--data_filter", type=str, default="all", choices=["all", "sp", "optimal"], help="Data filter strategy")
    parser.add_argument("--min_hops", type=int, default=1, help="Minimum shortest path hops for filtering (default: 1)")
    
    # --- Benchmark Arguments ---
    parser.add_argument("--run_benchmark", type=str2bool, default=True, help="Run benchmark after training?")

    parser.add_argument("--num_samples", type=int, default=100, help="Number of benchmark samples")
    parser.add_argument("--env_type", type=str, default="deflection", choices=["kdn", "deflection"], help="Environment type: 'kdn' or 'deflection'")

    args = parser.parse_args()

    # 1. TRAINING PHASE
    print("\n" + "="*50)
    print("PHASE 1: TRAINING")
    print("="*50)

    trainer = Trainer(
        tfrecords_dir=args.tfrecords_dir, 
        traffic_intensity=args.traffic_intensity, 
        data_filter=args.data_filter, 
        n_envs=args.n_envs,
        dataset_name=args.dataset_name,
        model_type=args.model_type,
        gnn_type=args.gnn_type,
        env_type=args.env_type,
        device=args.device,
        min_hops=args.min_hops
    )
    
    trainer.train(
        total_timesteps=args.total_timesteps, 
        log_interval=args.log_interval, 
        model_path=args.model_path
    )
    
    # Construct expected model path for benchmark
    # Trainer saves to: agents/{dataset}_{intensity}_{model}_{gnn}_{env}/{model_path}
    save_dir = trainer.base_save_dir
    final_model_full_path = os.path.join(save_dir, args.model_path)


    # 2. BENCHMARK PHASE
    if args.run_benchmark:
        print("\n" + "="*50)
        print("PHASE 2: BENCHMARKING")
        print("="*50)
        
        # Determine actual model type used (for BenchmarkRunner selection)
        # If Trainer used PPO (because env has no masking), we use 'PPO'
        use_masking = hasattr(trainer.model, "action_masks") # Only MaskablePPO has this in SB3-contrib directly? No, actually trainer knows.
        # Simple check: 
        from sb3_contrib import MaskablePPO
        agent_type_arg = "MaskPPO" if isinstance(trainer.model, MaskablePPO) else "PPO"

        # Paths
        ppo_path = final_model_full_path if agent_type_arg == "PPO" else "agents/ppo_model"
        maskppo_path = final_model_full_path if agent_type_arg == "MaskPPO" else "agents/maskppo_model"

        runner = BenchmarkRunner(
            tfrecords_dir=args.tfrecords_dir, 
            traffic_intensity=args.traffic_intensity, 
            num_samples=args.num_samples,
            ppo_path=ppo_path,
            maskppo_path=maskppo_path,
            agent_type=agent_type_arg,
            env_type=args.env_type,
            model_instance=trainer.model
        )
        
        runner.run()

if __name__ == "__main__":
    main()
