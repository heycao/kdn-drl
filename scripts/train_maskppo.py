import os
import sys
import argparse
import numpy as np
from collections import deque
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.masked_env import MaskedAdaptivePathEnv

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

def mask_fn(env):
    """Wrapper function to extract action masks from environment."""
    # Unwrap to get the underlying MaskedAdaptivePathEnv
    return env.unwrapped.action_masks()

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

    # Initialize training environment with action masking
    print(f"Initializing {n_envs} training environments with action masking...")
    env = make_vec_env(
        MaskedAdaptivePathEnv,
        n_envs=n_envs,
        env_kwargs=env_kwargs,
        vec_env_cls=DummyVecEnv,
        wrapper_class=ActionMasker,
        wrapper_kwargs={'action_mask_fn': mask_fn}
    )

    # Initialize evaluation environment with action masking
    eval_env_kwargs = {
        "tfrecords_dir": eval_tfrecords_dir,
        "traffic_intensity": traffic_intensity,
        "alpha": 10.0,
        "max_steps": 50
    }
    print(f"Initializing evaluation environment with action masking...")
    eval_env = make_vec_env(
        MaskedAdaptivePathEnv,
        n_envs=n_envs,
        env_kwargs=eval_env_kwargs,
        vec_env_cls=DummyVecEnv,
        wrapper_class=ActionMasker,
        wrapper_kwargs={'action_mask_fn': mask_fn}
    )

    # Set up MaskablePPO model
    model = MaskablePPO(
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
        tensorboard_log="./logs/maskppo_tensorboard/"
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=5000 // n_envs,
        save_path=f"{base_save_dir}/checkpoints/",
        name_prefix="maskppo_model"
    )
    custom_logging_callback = CustomLoggingCallback()
    
    # Delayed evaluation: only start eval at 30% of total timesteps
    eval_start_timestep = int(0.3 * total_timesteps)
    
    class MaskableEvalCallback(BaseCallback):
        """Custom evaluation callback that properly uses action masks.
        
        The standard EvalCallback doesn't pass action_masks to predict(),
        so we implement our own evaluation loop that handles masking.
        """
        def __init__(self, eval_env, eval_freq=10000, n_eval_episodes=10, 
                     best_model_save_path=None, log_path=None, 
                     start_timestep=0, deterministic=True, verbose=1):
            super().__init__(verbose)
            self.eval_env = eval_env
            self.eval_freq = eval_freq
            self.n_eval_episodes = n_eval_episodes
            self.best_model_save_path = best_model_save_path
            self.log_path = log_path
            self.start_timestep = start_timestep
            self.deterministic = deterministic
            self.best_mean_reward = -np.inf
            self._started = False
            self.evaluations_results = []
            self.evaluations_timesteps = []
        
        def _on_step(self) -> bool:
            if self.num_timesteps < self.start_timestep:
                return True
            
            if not self._started:
                print(f"\n>>> Starting evaluation at timestep {self.num_timesteps} (threshold: {self.start_timestep})")
                self._started = True
            
            if self.n_calls % self.eval_freq == 0:
                # Run evaluation with action masks
                episode_rewards = []
                episode_lengths = []
                successes = []
                
                for ep in range(self.n_eval_episodes):
                    obs = self.eval_env.reset()
                    done = [False]
                    episode_reward = 0.0
                    episode_length = 0
                    
                    while not done[0]:
                        # Get action masks from the vectorized environment
                        action_masks = np.array(self.eval_env.env_method("action_masks"))
                        action, _ = self.model.predict(
                            obs, 
                            deterministic=self.deterministic,
                            action_masks=action_masks
                        )
                        obs, reward, done, info = self.eval_env.step(action)
                        episode_reward += reward[0]
                        episode_length += 1
                    
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)
                    if 'is_success' in info[0]:
                        successes.append(float(info[0]['is_success']))
                
                mean_reward = np.mean(episode_rewards)
                std_reward = np.std(episode_rewards)
                mean_length = np.mean(episode_lengths)
                success_rate = np.mean(successes) if successes else 0.0
                
                self.evaluations_results.append(mean_reward)
                self.evaluations_timesteps.append(self.num_timesteps)
                
                # Log metrics
                self.logger.record("eval/mean_reward", mean_reward)
                self.logger.record("eval/mean_ep_length", mean_length)
                self.logger.record("eval/success_rate", success_rate)
                
                print(f"Eval num_timesteps={self.num_timesteps}, "
                      f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_length:.2f}, Success rate: {success_rate*100:.2f}%")
                
                # Save best model
                if mean_reward > self.best_mean_reward and self.best_model_save_path:
                    self.best_mean_reward = mean_reward
                    print("New best mean reward!")
                    os.makedirs(self.best_model_save_path, exist_ok=True)
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
            
            return True
    
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=f"{base_save_dir}/best_model/",
        log_path=f"{base_save_dir}/eval_results/",
        eval_freq=100000 // n_envs,  # Eval every ~10k steps
        n_eval_episodes=100,
        deterministic=True,
        start_timestep=eval_start_timestep
    )
    
    callback = CallbackList([checkpoint_callback, custom_logging_callback, eval_callback])

    # Train the model
    print(f"Starting MaskablePPO training for {total_timesteps} steps on {n_envs} environments with log_interval {log_interval}...")
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
    parser = argparse.ArgumentParser(description="Train MaskablePPO on MaskedAdaptivePathEnv")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000, help="Total training timesteps")
    parser.add_argument("--tfrecords_dir", type=str, default="data/nsfnetbw/tfrecords/train", help="Path to training TFRecords")
    parser.add_argument("--eval_tfrecords_dir", type=str, default="data/nsfnetbw/tfrecords/evaluate", help="Path to evaluation TFRecords")
    parser.add_argument("--model_path", type=str, default="final_model", help="Filename for the final trained model")
    parser.add_argument("--n_envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--log_interval", type=int, default=10, help="Number of iterations between console logs")
    parser.add_argument("--traffic_intensity", type=int, default=9, help="Traffic intensity for TFRecords filtering (default: 9)")
    parser.add_argument("--dataset_name", type=str, default=None, help="Dataset name (auto-detected from tfrecords_dir if not provided)")
    parser.add_argument("--model_type", type=str, default="MaskPPO", help="Model type identifier (default: MaskPPO)")
    
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
