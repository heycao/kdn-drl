import os
import sys
import argparse
import numpy as np
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.env import AdaptivePathEnv

class CustomLoggingCallback(BaseCallback):
    """
    Callback for logging custom metrics (mlu, success_rate) to Tensorboard.
    Tracks moving average over the last 100 episodes.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.mlus = deque(maxlen=100)
        self.successes = deque(maxlen=100)

    def _on_step(self) -> bool:
        # Check if any of the environments finished an episode
        for done, info in zip(self.locals['dones'], self.locals['infos']):
            if done:
                if 'mlu' in info:
                    self.mlus.append(info['mlu'])
                if 'is_success' in info:
                    self.successes.append(float(info['is_success']))
        
        # Record metrics if we have data
        if len(self.mlus) > 0:
            self.logger.record("rollout/mlu", np.mean(self.mlus))
        if len(self.successes) > 0:
            self.logger.record("rollout/success_rate", np.mean(self.successes))
            
        return True

def train(total_timesteps, tfrecords_dir, eval_tfrecords_dir, model_path, n_envs, log_interval, traffic_intensity, dataset_name, model_type):
    # Extract dataset name if not provided
    if dataset_name is None:
        # Extract from tfrecords_dir, e.g., 'data/nsfnetbw/tfrecords/train' -> 'nsfnetbw'
        dataset_name = tfrecords_dir.split('/')[-3] if len(tfrecords_dir.split('/')) >= 3 else 'unknown'
    
    # Create base save directory
    base_save_dir = f"agents/{dataset_name}_{traffic_intensity}_{model_type}"
    
    # Environment parameters
    env_kwargs = {
        "tfrecords_dir": tfrecords_dir,
        "traffic_intensity": traffic_intensity,
        "alpha": 10.0,
        "max_steps": 50
    }

    # Initialize training environment
    print(f"Initializing {n_envs} training environments...")
    env = make_vec_env(
        AdaptivePathEnv,
        n_envs=n_envs,
        env_kwargs=env_kwargs,
        vec_env_cls=DummyVecEnv
    )

    # Initialize evaluation environment
    eval_env_kwargs = {
        "tfrecords_dir": eval_tfrecords_dir,
        "traffic_intensity": traffic_intensity,
        "alpha": 10.0,
        "max_steps": 50
    }
    print(f"Initializing evaluation environment...")
    eval_env = make_vec_env(
        AdaptivePathEnv,
        n_envs=n_envs,
        env_kwargs=eval_env_kwargs,
        vec_env_cls=DummyVecEnv
    )

    # Set up PPO model
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=1024,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log="./logs/ppo_tensorboard/"
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=5000 // n_envs,
        save_path=f"{base_save_dir}/checkpoints/",
        name_prefix="ppo_model"
    )
    custom_logging_callback = CustomLoggingCallback()
    
    # Delayed evaluation: only start eval at 80% of total timesteps
    eval_start_timestep = int(0.3 * total_timesteps)
    
    class DelayedEvalCallback(EvalCallback):
        """EvalCallback that only starts evaluating after a specified timestep.
        Also skips saving best model if success_rate < 100%.
        """
        def __init__(self, *args, start_timestep=0, **kwargs):
            super().__init__(*args, **kwargs)
            self.start_timestep = start_timestep
            self._started = False
        
        def _on_step(self) -> bool:
            if self.num_timesteps < self.start_timestep:
                return True  # Skip evaluation until start_timestep
            if not self._started:
                print(f"\n>>> Starting evaluation at timestep {self.num_timesteps} (threshold: {self.start_timestep})")
                self._started = True
            
            # Call parent _on_step which does the evaluation
            result = super()._on_step()
            
            # After evaluation, check if we should keep the "best" model
            # Only save if success_rate == 100%
            if self.last_mean_reward is not None and hasattr(self, '_is_success_buffer'):
                success_rate = np.mean(self._is_success_buffer) if len(self._is_success_buffer) > 0 else 0.0
                if success_rate < 1.0 and self.best_model_save_path is not None:
                    # Revert best_mean_reward to prevent saving
                    self.best_mean_reward = float('inf')
                    
            return result
    
    eval_callback = DelayedEvalCallback(
        eval_env,
        best_model_save_path=f"{base_save_dir}/best_model/",
        log_path=f"{base_save_dir}/eval_results/",
        eval_freq=100000 // n_envs,  # Eval every ~10k steps
        n_eval_episodes=20000,
        deterministic=True,
        render=False,
        start_timestep=eval_start_timestep
    )
    
    callback = CallbackList([checkpoint_callback, custom_logging_callback, eval_callback])

    # Train the model
    print(f"Starting training for {total_timesteps} steps on {n_envs} environments with log_interval {log_interval}...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
        log_interval=log_interval
    )

    # Save the final model
    final_model_path = f"{base_save_dir}/{model_path}" if not model_path.startswith('agents/') else model_path
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    print(f"Best model saved to {base_save_dir}/best_model/")
    print(f"Checkpoints saved to {base_save_dir}/checkpoints/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SB3 PPO on AdaptivePathEnv")
    parser.add_argument("--total_timesteps", type=int, default=2_000_000, help="Total training timesteps")
    parser.add_argument("--tfrecords_dir", type=str, default="data/nsfnetbw/tfrecords/train", help="Path to training TFRecords")
    parser.add_argument("--eval_tfrecords_dir", type=str, default="data/nsfnetbw/tfrecords/evaluate", help="Path to evaluation TFRecords")
    parser.add_argument("--model_path", type=str, default="final_model", help="Filename for the final trained model")
    parser.add_argument("--n_envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--log_interval", type=int, default=10, help="Number of iterations between console logs")
    parser.add_argument("--traffic_intensity", type=int, default=9, help="Traffic intensity for TFRecords filtering (default: 9)")
    parser.add_argument("--dataset_name", type=str, default=None, help="Dataset name (auto-detected from tfrecords_dir if not provided)")
    parser.add_argument("--model_type", type=str, default="PPO", help="Model type identifier (default: PPO)")
    
    args = parser.parse_args()
    
    train(
        total_timesteps=args.total_timesteps,
        tfrecords_dir=args.tfrecords_dir,
        eval_tfrecords_dir=args.eval_tfrecords_dir,
        model_path=args.model_path,
        n_envs=args.n_envs,
        log_interval=args.log_interval,
        traffic_intensity=args.traffic_intensity,
        dataset_name=args.dataset_name,
        model_type=args.model_type
    )
