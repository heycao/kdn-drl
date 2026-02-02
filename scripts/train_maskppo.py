import argparse
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trainer import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MaskablePPO on MaskedAdaptivePathEnv")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000, help="Total training timesteps")
    parser.add_argument("--tfrecords_dir", type=str, default="data/nsfnetbw", help="Path to training dataset root")

    parser.add_argument("--model_path", type=str, default="final_model", help="Filename for the final trained model")
    parser.add_argument("--n_envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--log_interval", type=int, default=1, help="Number of iterations between console logs")
    parser.add_argument("--traffic_intensity", type=int, default=9, help="Traffic intensity for TFRecords filtering (default: 9)")
    parser.add_argument("--dataset_name", type=str, default=None, help="Dataset name (auto-detected from tfrecords_dir if not provided)")
    parser.add_argument("--model_type", type=str, default="MaskPPO", help="Model type identifier (default: MaskPPO)")
    parser.add_argument("--gnn_type", type=str, default="gcn", choices=["gcn", "gat"], help="Type of GNN to use: 'gcn' or 'gat' (default: gcn)")
    parser.add_argument("--data_filter", type=str, default="all", choices=["all", "sp", "optimal"], help="Filter data: 'sp' (shortest path is optimal), 'optimal' (optimal != shortest path), 'all' (default: all)")
    
    args = parser.parse_args()
    
    trainer = Trainer(
        tfrecords_dir=args.tfrecords_dir, 
        traffic_intensity=args.traffic_intensity, 
        data_filter=args.data_filter, 
        n_envs=args.n_envs,
        dataset_name=args.dataset_name,
        model_type=args.model_type,
        gnn_type=args.gnn_type
    )
    
    trainer.train(
        total_timesteps=args.total_timesteps, 
        log_interval=args.log_interval, 
        model_path=args.model_path
    )
