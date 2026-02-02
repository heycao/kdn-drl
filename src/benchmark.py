import os
import sys
import numpy as np
import torch
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from tqdm.rich import tqdm
import networkx as nx

from src.env import KDNEnvinronment
from src.masked_env import MaskedKDNEnv
from src.datanet import Datanet
from src.gcn import GCNFeatureExtractor
from joblib import Parallel, delayed
import multiprocessing

class BenchmarkRunner:
    def __init__(self, tfrecords_dir, traffic_intensity, num_samples=10, 
                 ppo_path="agents/ppo_model", maskppo_path="agents/maskppo_model", 
                 agent_type="MaskPPO", model_instance=None):
        self.tfrecords_dir = tfrecords_dir
        self.traffic_intensity = traffic_intensity
        self.num_samples = num_samples
        self.ppo_path = ppo_path
        self.maskppo_path = maskppo_path
        self.agent_type = agent_type
        self.model_instance = model_instance
        
        # Determine device
        self.device = "cpu"
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"

    def load_samples(self):
        """Load samples using Datanet."""
        print(f"Loading samples from {self.tfrecords_dir} with intensity {self.traffic_intensity}...")
        # Initialize Datanet reader
        reader = Datanet(self.tfrecords_dir, intensity_values=[self.traffic_intensity])
        
        samples = []
        for i, sample in enumerate(reader):
            if self.num_samples and i >= self.num_samples:
                break
            samples.append(sample)
            
        print(f"Loaded {len(samples)} samples.")
        return samples

    def evaluate_agent_mlu(self, agent_path, env, samples, agent_label='PPO', model_instance=None):
        """Evaluate trained agent's MLU on Datanet samples (Parallelized)."""
        print(f"\nEvaluating {agent_label} agent using parallel workers...")
        
        # We'll use a helper function to evaluate a single sample
        # To avoid pickling issues/overhead, each worker will load the model if agent_path is provided.
        # If model_instance is provided, we might have pickling issues with joblib 'loky' backend.
        # So we'll prefer loading from path in workers.
        
        n_jobs = min(multiprocessing.cpu_count(), len(samples), 8)
        
        # If we have a model instance, we might need to save it to a temporary file if we want to parallelize
        # because MaskablePPO/PPO instances are hard to pickle across processes.
        # For simplicity, if model_instance is provided, we save it to a temp file once.
        temp_model_path = None
        if model_instance:
            temp_model_path = "agents/temp_eval_model"
            model_instance.save(temp_model_path)
            agent_path = temp_model_path

        def eval_single_sample(sample, agent_path, agent_label, traffic_intensity, tfrecords_dir):
            # Create a local environment for this worker
            if agent_label == 'MaskPPO':
                local_env = MaskedKDNEnv(tfrecords_dir=tfrecords_dir, traffic_intensity=traffic_intensity)
                custom_objects = {"features_extractor_class": GCNFeatureExtractor}
                local_model = MaskablePPO.load(agent_path, device="cpu", custom_objects=custom_objects)
                use_masking = True
            else:
                local_env = KDNEnvinronment(tfrecords_dir=tfrecords_dir, traffic_intensity=traffic_intensity)
                local_model = PPO.load(agent_path, device="cpu")
                use_masking = False
            
            local_env.sample = sample
            n = sample.get_network_size()
            
            sample_mlus = []
            sample_path_lengths = []
            sample_success_count = 0
            sample_episodes = 0
            
            for src in range(n):
                for dst in range(n):
                    if src == dst: continue
                    
                    sample_episodes += 1
                    local_env.current_node = src
                    local_env.destination = dst
                    local_env.path = [src]
                    local_env.current_step = 0
                    local_env.optimal_path = None
                    
                    obs = local_env._get_obs()
                    done = truncated = False
                    steps = 0
                    max_steps = 50
                    episode_mlu = 1.0
                    
                    while not done and not truncated and steps < max_steps:
                        if use_masking:
                            action_masks = local_env.action_masks()
                            action, _ = local_model.predict(obs, deterministic=True, action_masks=action_masks)
                        else:
                            action, _ = local_model.predict(obs, deterministic=True)
                        
                        obs, reward, done, truncated, info = local_env.step(action)
                        steps += 1
                        
                        if done:
                            if 'mlu' in info: episode_mlu = info['mlu']
                            if info.get('is_success', False): sample_success_count += 1
                    
                    sample_mlus.append(episode_mlu)
                    sample_path_lengths.append(len(local_env.path))
            
            return sample_mlus, sample_path_lengths, sample_success_count, sample_episodes

        # Run in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(eval_single_sample)(s, agent_path, agent_label, self.traffic_intensity, self.tfrecords_dir) 
            for s in tqdm(samples, desc=f"Parallel Eval {agent_label}")
        )
        
        # Aggregate results
        mlus = []
        path_lengths = []
        success_count = 0
        total_episodes = 0
        
        for s_mlus, s_paths, s_success, s_episodes in results:
            mlus.extend(s_mlus)
            path_lengths.extend(s_paths)
            success_count += s_success
            total_episodes += s_episodes
            
        if temp_model_path and os.path.exists(temp_model_path + ".zip"):
            os.remove(temp_model_path + ".zip")

        return {
            'mlu_mean': np.mean(mlus),
            'mlu_std': np.std(mlus),
            'mlu_median': np.median(mlus),
            'mlu_min': np.min(mlus),
            'mlu_max': np.max(mlus),
            'path_length_mean': np.mean(path_lengths),
            'success_rate': success_count / total_episodes if total_episodes > 0 else 0.0,
            'num_episodes': total_episodes
        }

    def evaluate_baseline_sp(self, env, samples):
        """Evaluate baseline shortest path routing MLU (Parallelized)."""
        print("\nEvaluating baseline (Shortest Path) routing using parallel workers...")
        
        n_jobs = min(multiprocessing.cpu_count(), len(samples), 8)

        def eval_single_baseline(sample, traffic_intensity, tfrecords_dir):
            local_env = KDNEnvinronment(tfrecords_dir=tfrecords_dir, traffic_intensity=traffic_intensity)
            local_env.sample = sample
            G = sample.topology_object
            n = sample.get_network_size()
            
            sample_mlus = []
            sample_path_lengths = []
            sample_episodes = 0
            
            for src in range(n):
                for dst in range(n):
                    if src == dst: continue
                    
                    sample_episodes += 1
                    try:
                        path = nx.shortest_path(G, source=src, target=dst, weight='weight')
                        bg_loads = sample.calculate_background_loads(src, dst)
                        mlu = sample.calculate_max_utilization(path, bg_loads)
                        sample_mlus.append(mlu)
                        sample_path_lengths.append(len(path))
                    except nx.NetworkXNoPath:
                        sample_path_lengths.append(0)
                        sample_mlus.append(2.0)
            
            return sample_mlus, sample_path_lengths, sample_episodes

        results = Parallel(n_jobs=n_jobs)(
            delayed(eval_single_baseline)(s, self.traffic_intensity, self.tfrecords_dir) 
            for s in tqdm(samples, desc="Parallel Eval SP Baseline")
        )
        
        mlus = []
        path_lengths = []
        total_episodes = 0
        
        for s_mlus, s_paths, s_episodes in results:
            mlus.extend(s_mlus)
            path_lengths.extend(s_paths)
            total_episodes += s_episodes
                        
        return {
            'mlu_mean': np.mean(mlus),
            'mlu_std': np.std(mlus),
            'mlu_median': np.median(mlus),
            'mlu_min': np.min(mlus),
            'mlu_max': np.max(mlus),
            'path_length_mean': np.mean(path_lengths),
            'success_rate': 1.0,
            'num_episodes': total_episodes
        }

    def print_results(self, results_dict, baseline_results):
        """Print comparison results."""
        print("\n" + "="*90)
        print("MLU BENCHMARK RESULTS")
        print("="*90)
        
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
            # Positive % means lower MLU (better)
            mlu_improvement = ((baseline_results['mlu_mean'] - agent_results['mlu_mean']) / baseline_results['mlu_mean']) * 100
            
            print(f"\n{agent_name}:")
            print(f"  MLU Improvement: {mlu_improvement:+.2f}%")
            if mlu_improvement > 0.1:
                print(f"  ✅ {agent_name} performs BETTER than baseline (lower MLU)")
            elif mlu_improvement < -0.1:
                print(f"  ❌ {agent_name} performs WORSE than baseline (higher MLU)")
            else:
                print(f"  ➖ {agent_name} performs SAME as baseline")
        
        print("\n" + "="*90)

    def run(self):
        print("="*90)
        print("MLU BENCHMARK - Evaluating Agents vs Baseline (Shortest Path)")
        print("="*90)
        
        # Load samples
        samples = self.load_samples()
        if not samples:
            print("No samples loaded. Exiting.")
            return

        # Collect results
        results_dict = {}
        
        # Check agents
        if self.agent_type in ['PPO', 'all']:
            print(f"\nInitializing environment for PPO...")
            env = KDNEnvinronment(tfrecords_dir=self.tfrecords_dir, traffic_intensity=self.traffic_intensity)
            try:
                # Pass model_instance if available, but checking agent_type match if needed.
                # Assuming model_instance corresponds to agent_type.
                model_inst = self.model_instance if self.agent_type == 'PPO' else None
                res = self.evaluate_agent_mlu(self.ppo_path, env, samples, 'PPO', model_instance=model_inst)
                results_dict['PPO'] = res
            except Exception as e:
                print(f"Failed to evaluate PPO: {e}")
                
        if self.agent_type in ['MaskPPO', 'all']:
            print(f"\nInitializing environment for MaskPPO...")
            env = MaskedKDNEnv(tfrecords_dir=self.tfrecords_dir, traffic_intensity=self.traffic_intensity)
            try:
                model_inst = self.model_instance if self.agent_type == 'MaskPPO' else None
                res = self.evaluate_agent_mlu(self.maskppo_path, env, samples, 'MaskPPO', model_instance=model_inst)
                results_dict['MaskPPO'] = res
            except Exception as e:
                print(f"Failed to evaluate MaskPPO: {e}")
                import traceback
                traceback.print_exc()

        # Baseline
        print(f"\nInitializing environment for Baseline...")
        env = KDNEnvinronment(tfrecords_dir=self.tfrecords_dir, traffic_intensity=self.traffic_intensity)
        baseline_results = self.evaluate_baseline_sp(env, samples)
        
        if results_dict:
            self.print_results(results_dict, baseline_results)
        else:
            print("No agent results collected.")
