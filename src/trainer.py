import os
import torch
import numpy as np
import networkx as nx
from collections import deque
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from tqdm import tqdm

from src.env import DeflectionEnv
from src.masked_env import MaskedDeflectionEnv
from src.gcn import GCNFeatureExtractor
from src.gat import GATFeatureExtractor
from src.datanet import Datanet

class CustomLoggingCallback(BaseCallback):
    """
    Callback for logging custom metrics (mlu, success_rate) to Tensorboard.
    Tracks moving average over the last 100 episodes.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.mlus = deque(maxlen=100)
        self.successes = deque(maxlen=100)
        self.optimal_rates = deque(maxlen=100)
        self.optimal_sp_rates = deque(maxlen=100)
        self.improvements = deque(maxlen=100)
        self.gap_scores = deque(maxlen=100)
        self.peak_loss_rates = deque(maxlen=100)

    def _on_step(self) -> bool:
        # Check if any of the environments finished an episode
        for done, info in zip(self.locals['dones'], self.locals['infos']):
            if done:
                if 'mlu' in info:
                    self.mlus.append(info['mlu'])
                if 'is_success' in info:
                    self.successes.append(float(info['is_success']))
                if 'is_optimal' in info:
                    self.optimal_rates.append(float(info['is_optimal']))
                if 'is_optimal_shortest' in info:
                    self.optimal_sp_rates.append(float(info['is_optimal_shortest']))
                if 'improvement' in info:
                    self.improvements.append(info['improvement'])
                if 'gap_score' in info:
                    self.gap_scores.append(info['gap_score'])
                if 'peak_loss_rate' in info:
                    self.peak_loss_rates.append(info['peak_loss_rate'])
        
        # Record metrics if we have data
        if len(self.mlus) > 0:
            self.logger.record("rollout/mlu", np.mean(self.mlus))
        if len(self.successes) > 0:
            self.logger.record("rollout/success_rate", np.mean(self.successes))
        # if len(self.optimal_rates) > 0:
        #     self.logger.record("rollout/optimal_rate", np.mean(self.optimal_rates))
        if len(self.optimal_sp_rates) > 0:
            self.logger.record("rollout/optimal_sp_rate", np.mean(self.optimal_sp_rates))
        if len(self.improvements) > 0:
            self.logger.record("rollout/improvement", np.mean(self.improvements))
        if len(self.gap_scores) > 0:
            self.logger.record("rollout/gap_score", np.mean(self.gap_scores))
        if len(self.peak_loss_rates) > 0:
            self.logger.record("rollout/peak_loss_rate", np.mean(self.peak_loss_rates))
            
        return True

def mask_fn(env):
    """Wrapper function to extract action masks from environment."""
    # Unwrap to get the underlying MaskedAdaptivePathEnv
    return env.unwrapped.action_masks()

class Trainer:
    def __init__(self, tfrecords_dir, traffic_intensity, data_filter="all", n_envs=8, 
                 dataset_name=None, model_type="MaskPPO", gnn_type="gcn", 
                 env_type="base", device=None, min_hops=4):
        self.tfrecords_dir = tfrecords_dir
        self.traffic_intensity = traffic_intensity
        self.data_filter = data_filter
        self.n_envs = n_envs
        self.dataset_name = dataset_name or os.path.basename(os.path.normpath(tfrecords_dir))
        self.model_type = model_type
        self.gnn_type = gnn_type
        self.env_type = env_type
        self.min_hops = min_hops
        
        # Device detection
        if device:
            self.device = device
        else:
            self.device = "cpu"
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
        
        self.base_save_dir = f"agents/{self.dataset_name}_{self.traffic_intensity}_{self.model_type}_{self.gnn_type}_{self.env_type}"
        print(f"Trainer initialized on device {self.device}. Save dir: {self.base_save_dir}")

    def pre_process_data(self, max_samples=2000):
        """
        Pre-processes data to find valid (Sample, src, dst) pairs based on the filter.
        Returns a list of tuples or None if 'all'.
        """
        if self.data_filter == 'all':
            return None
            
        print(f"Pre-processing samples from {self.tfrecords_dir} with filter: {self.data_filter}...")
        reader = Datanet(self.tfrecords_dir, intensity_values=[self.traffic_intensity])
        valid_samples = []
        
        with tqdm(total=max_samples, desc="Valid Samples", unit="pair") as pbar:
            total_checked = 0
            for i, sample in enumerate(reader):
                n = sample.get_network_size()
                
                # Check all pairs in this sample
                for src in range(n):
                    for dst in range(n):
                        if src == dst: continue
                        
                        total_checked += 1
                        if total_checked % 1000 == 0:
                            pbar.set_postfix({"checked": total_checked})
                        
                        # --- Filter Logic ---
                        # 1. Calc BG Loads & Optimal Path
                        bg_loads = sample.calculate_background_loads(src, dst)
                        optimal_path = sample.search_optimal_path(src, dst, bg_loads, max_steps=100)
                        
                        # 2. Calc Shortest Path (Hops)
                        try:
                            shortest_path = nx.shortest_path(sample.topology_object, src, dst, weight='weight')
                        except nx.NetworkXNoPath:
                            shortest_path = None
                            
                        # 3. Check Condition
                        is_valid = False
                        if self.data_filter == 'sp':
                            # shortest_path IS optimal path
                            if shortest_path and optimal_path and shortest_path == optimal_path:
                                # Check hops condition: len(path) > min_hops + 1
                                if len(shortest_path) > self.min_hops + 1:
                                    is_valid = True
                        elif self.data_filter == 'optimal':
                            # optimal path should be > 10% better than shortest path
                            if shortest_path and optimal_path:
                                sp_mlu = sample.calculate_max_utilization(shortest_path, bg_loads)
                                opt_mlu = sample.calculate_max_utilization(optimal_path, bg_loads)
                                
                                if sp_mlu > 0:
                                    improvement = (sp_mlu - opt_mlu) / sp_mlu
                                    if improvement > 0.01:
                                        # Check hops condition: len(path) > min_hops + 1
                                        if len(shortest_path) > self.min_hops + 1:
                                            is_valid = True
                        
                        if is_valid:
                            valid_samples.append((sample, src, dst))
                            pbar.update(1)
                            if len(valid_samples) >= max_samples:
                                break
                
                if len(valid_samples) >= max_samples:
                    break
        
        print(f"Pre-processing complete. Found {len(valid_samples)} valid (Sample, Src, Dst) pairs.")
        
        if len(valid_samples) == 0:
            print("WARNING: No valid samples found with the current filter! Training might fail or behave unexpectedly.")
            
        return valid_samples

    def train(self, total_timesteps, log_interval=1, model_path="final_model"):
        # --- Pre-process Data ---
        prefiltered_samples = self.pre_process_data()
        
        # Environment parameters
        env_kwargs = {
            "tfrecords_dir": self.tfrecords_dir,
            "traffic_intensity": self.traffic_intensity,
            "prefiltered_samples": prefiltered_samples, # Pass filtered list
        }

        # Select Environment Class
        if self.env_type == "masked":
            env_cls = MaskedDeflectionEnv
        else:
            # Default to base deflection environment
            env_cls = DeflectionEnv

        # Auto-detect if environment supports masking
        use_masking = hasattr(env_cls, 'action_masks')
        print(f"Environment {env_cls.__name__} masking support: {use_masking}")

        # Initialize training environment
        print(f"Initializing {self.n_envs} training environments ({self.env_type})...")
        env = make_vec_env(
            env_cls,
            n_envs=self.n_envs,
            env_kwargs=env_kwargs,
            vec_env_cls=None,
        )

        # Set up Policy and Features Extractor
        policy_kwargs = {}
        features_extractor_class = None
        
        if self.gnn_type == "gat":
            features_extractor_class = GATFeatureExtractor
            print("Using GAT Feature Extractor")
        elif self.gnn_type == "gcn":
            features_extractor_class = GCNFeatureExtractor
            print("Using GCN Feature Extractor")
        elif self.gnn_type == "none" or not self.gnn_type:
            print("Using Default Feature Extractor (MLP/Flatten)")
        else:
            print(f"Unknown gnn_type '{self.gnn_type}', falling back to MLP.")

        if features_extractor_class:
            policy_kwargs = dict(
                features_extractor_class=features_extractor_class,
                features_extractor_kwargs=dict(
                    hidden_dim=64,
                    n_layers=2,
                    out_dim=None
                )
            )
        
        # Select Model Class
        if use_masking:
            model_cls = MaskablePPO
            print("Using MaskablePPO (Masking Enabled)")
        else:
            model_cls = PPO
            print("Using Standard PPO (Masking Disabled)")

        self.model = model_cls(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=3e-4,
            n_steps=256,   
            batch_size=64, 
            n_epochs=10,
            gamma=0.99,
            # gae_lambda=0.95,
            # clip_range=0.2,
            # ent_coef=0.01,
            tensorboard_log="./logs/maskppo_tensorboard/",
            device=self.device
        )

        # Callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=5000 // self.n_envs,
            save_path=f"{self.base_save_dir}/checkpoints/",
            name_prefix="maskppo_model"
        )
        custom_logging_callback = CustomLoggingCallback()
        
        callback = CallbackList([checkpoint_callback, custom_logging_callback])

        # Train the model
        print(f"Starting MaskablePPO training for {total_timesteps} steps on {self.n_envs} environments with log_interval {log_interval}...")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True,
            log_interval=log_interval
        )

        # Save the final model
        final_model_path = f"{self.base_save_dir}/{model_path}" if not model_path.startswith('agents/') else model_path
        self.model.save(final_model_path)
        print(f"Final model saved to {final_model_path}")
        print(f"Best model saved to {self.base_save_dir}/best_model/")
        print(f"Checkpoints saved to {self.base_save_dir}/checkpoints/")
