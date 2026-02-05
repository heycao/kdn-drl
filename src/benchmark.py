import os
import sys
import numpy as np
import torch
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from tqdm import tqdm
import networkx as nx

from src.env import DeflectionEnv
from src.masked_env import MaskedDeflectionEnv
from src.datanet import Datanet
from src.gcn import GCNFeatureExtractor
from joblib import Parallel, delayed
import multiprocessing

class BenchmarkRunner:
    def __init__(self, tfrecords_dir, traffic_intensity, num_samples=10, 
                 ppo_path="agents/ppo_model", maskppo_path="agents/maskppo_model", 
                 agent_type="MaskPPO", env_type="base", model_instance=None, seed=None):
        self.tfrecords_dir = tfrecords_dir
        self.traffic_intensity = traffic_intensity
        self.num_samples = num_samples
        self.ppo_path = ppo_path
        self.maskppo_path = maskppo_path
        self.agent_type = agent_type
        self.env_type = env_type
        self.model_instance = model_instance
        self.seed = seed
        
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

        def eval_single_sample(sample, agent_path, agent_label, traffic_intensity, tfrecords_dir, env_type="base"):
            # Create a local environment for this worker
            if agent_label == 'MaskPPO':
                if env_type == "masked":
                    local_env = MaskedDeflectionEnv(tfrecords_dir=tfrecords_dir, traffic_intensity=traffic_intensity)
                else:
                    local_env = DeflectionEnv(tfrecords_dir=tfrecords_dir, traffic_intensity=traffic_intensity)
                    
                # Attempt to load with custom extractors if they exist
                custom_objects = {}
                try:
                    custom_objects["features_extractor_class"] = GCNFeatureExtractor
                except:
                    pass
                local_model = MaskablePPO.load(agent_path, device="cpu", custom_objects=custom_objects)
                use_masking = True
            else:
                # PPO case (usually unmasked)
                if env_type == "masked":
                     local_env = MaskedDeflectionEnv(tfrecords_dir=tfrecords_dir, traffic_intensity=traffic_intensity)
                else:
                    local_env = DeflectionEnv(tfrecords_dir=tfrecords_dir, traffic_intensity=traffic_intensity)
                local_model = PPO.load(agent_path, device="cpu")
                use_masking = False
            
            # local_env.sample = sample # Handled in loop or reset now
            n = sample.get_network_size()
            
            sample_mlus = []
            sample_path_lengths = []
            sample_success_count = 0
            sample_episodes = 0
            
            for src in range(n):
                for dst in range(n):
                    if src == dst: continue
                    
                    sample_episodes += 1
                    # Use proper reset to initialize Env state (topology copies, MLU calcs, etc.)
                    obs, _ = local_env.reset(options={"sample": sample, "src": src, "dst": dst})
                    done = truncated = False
                    steps = 0
                    max_steps = 50
                    episode_mlu = 1.0 # Default High
                    episode_success = False
                    
                    while not done and not truncated and steps < max_steps:
                        if use_masking:
                            action_masks = local_env.action_masks()
                            action, _ = local_model.predict(obs, deterministic=True, action_masks=action_masks)
                        else:
                            action, _ = local_model.predict(obs, deterministic=True)
                        
                        obs, reward, done, truncated, info = local_env.step(action)
                        steps += 1
                        
                        if done and env_type != "deflection":
                            if 'mlu' in info: episode_mlu = info['mlu']
                            if info.get('is_success', False): episode_success = True
                            
                    # Use BEST path metrics (best MLU found during episode)
                    episode_mlu = local_env.min_mlu_so_far
                    p_len = len(local_env.best_path_so_far)
                    
                    if local_env.original_mlu - episode_mlu > 1e-4:
                        episode_success = True
                    
                    if episode_success:
                        sample_success_count += 1
                    
                    sample_mlus.append(episode_mlu)
                    sample_path_lengths.append(p_len)
            
            return sample_mlus, sample_path_lengths, sample_success_count, sample_episodes

        # Run in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(eval_single_sample)(s, agent_path, agent_label, self.traffic_intensity, self.tfrecords_dir, self.env_type) 
            for s in tqdm(samples, desc=f"Parallel Eval {agent_label}")
        )
        
        # Aggregate results
        mlus = []
        path_lengths = []
        loss_rates = []
        success_count = 0
        total_episodes = 0
        
        for s_mlus, s_paths, s_success, s_episodes in results:
            mlus.extend(s_mlus)
            path_lengths.extend(s_paths)
            success_count += s_success
            total_episodes += s_episodes
            
            for m in s_mlus:
                loss = max(0.0, 1.0 - 1.0/m) if m > 0 else 0.0
                loss_rates.append(loss)
            
        if temp_model_path and os.path.exists(temp_model_path + ".zip"):
            os.remove(temp_model_path + ".zip")

        return {
            'mlu_mean': np.mean(mlus),
            'mlu_std': np.std(mlus),
            'mlu_median': np.median(mlus),
            'mlu_min': np.min(mlus),
            'mlu_max': np.max(mlus),
            'loss_mean': np.mean(loss_rates),
            'loss_max': np.max(loss_rates),
            'path_length_mean': np.mean(path_lengths),
            'success_rate': success_count / total_episodes if total_episodes > 0 else 0.0,
            'num_episodes': total_episodes
        }

    def evaluate_baselines(self, env, samples):
        """Evaluate baseline SP and Oracle (Optimal) MLU (Parallelized)."""
        print("\nEvaluating Baselines (Shortest Path & Oracle) using parallel workers...")
        
        n_jobs = min(multiprocessing.cpu_count(), len(samples), 8)

        def eval_single_baseline(sample, traffic_intensity, tfrecords_dir):
            local_env = DeflectionEnv(tfrecords_dir=tfrecords_dir, traffic_intensity=traffic_intensity)
            local_env.sample = sample
            G = sample.topology_object
            n = sample.get_network_size()
            
            sample_sp_mlus = []
            sample_opt_mlus = []
            sample_episodes = 0
            
            for src in range(n):
                for dst in range(n):
                    if src == dst: continue
                    
                    sample_episodes += 1
                    bg_loads = sample.calculate_background_loads(src, dst)
                    
                    # 1. Shortest Path (Baseline)
                    try:
                        sp_path = nx.shortest_path(G, source=src, target=dst, weight='weight')
                        sp_mlu = sample.calculate_max_utilization(sp_path, bg_loads)
                    except nx.NetworkXNoPath:
                        sp_mlu = 2.0 # Penalty
                    
                    # 2. Optimal Path (Oracle)
                    opt_path = sample.search_optimal_path(src, dst, bg_loads, max_steps=100)
                    if opt_path:
                        opt_mlu = sample.calculate_max_utilization(opt_path, bg_loads)
                    else:
                        opt_mlu = sp_mlu # Fallback
                        
                    sample_sp_mlus.append(sp_mlu)
                    sample_opt_mlus.append(opt_mlu)
            
            return sample_sp_mlus, sample_opt_mlus, sample_episodes

        results = Parallel(n_jobs=n_jobs)(
            delayed(eval_single_baseline)(s, self.traffic_intensity, self.tfrecords_dir) 
            for s in tqdm(samples, desc="Parallel Eval Baselines")
        )
        
        sp_mlus = []
        opt_mlus = []
        
        sp_loss_rates = []
        opt_loss_rates = []
        
        total_episodes = 0
        
        for s_sp, s_opt, s_episodes in results:
            sp_mlus.extend(s_sp)
            opt_mlus.extend(s_opt)
            total_episodes += s_episodes
            
            for m in s_sp:
                sp_loss_rates.append(max(0.0, 1.0 - 1.0/m) if m > 0 else 0.0)
                
            for m in s_opt:
                opt_loss_rates.append(max(0.0, 1.0 - 1.0/m) if m > 0 else 0.0)
            
        return {
            'sp': {
                'mlu_mean': np.mean(sp_mlus),
                'mlu_std': np.std(sp_mlus),
                'mlu_median': np.median(sp_mlus),
                'mlu_max': np.max(sp_mlus),
                'loss_mean': np.mean(sp_loss_rates),
                'loss_max': np.max(sp_loss_rates),
                'num_episodes': total_episodes
            },
            'oracle': {
                'mlu_mean': np.mean(opt_mlus),
                'mlu_std': np.std(opt_mlus),
                'mlu_median': np.median(opt_mlus),
                'mlu_max': np.max(opt_mlus),
                'loss_mean': np.mean(opt_loss_rates),
                'loss_max': np.max(opt_loss_rates),
                'num_episodes': total_episodes
            }
        }

    def print_results(self, results_dict, baseline_results):
        """Print comparison results."""
        print("\n" + "="*110)
        print("MLU BENCHMARK RESULTS")
        print("="*110)
        
        sp_res = baseline_results['sp']
        opt_res = baseline_results['oracle']
        
        header_parts = [f"{'Metric':<30}"]
        for agent_name in sorted(results_dict.keys()):
            header_parts.append(f"{agent_name:<15}")
        header_parts.append(f"{'Baseline(SP)':<15}")
        header_parts.append(f"{'Oracle(Opt)':<15}")
        print(" ".join(header_parts))
        print("-"*110)
        
        metrics = [
            ('Mean MLU', 'mlu_mean'),
            ('Median MLU', 'mlu_median'),
            ('Std Dev MLU', 'mlu_std'),
            ('Max MLU', 'mlu_max'),
            ('Mean Peak Loss Rate', 'loss_mean'), # NEW
            ('Max Peak Loss Rate', 'loss_max'),   # NEW
        ]
        
        for label, key in metrics:
            row_parts = [f"{label:<30}"]
            
            # Agents
            for agent_name in sorted(results_dict.keys()):
                agent_val = results_dict[agent_name][key] if key in results_dict[agent_name] else 0.0
                if 'loss' in key:
                    row_parts.append(f"{agent_val:.4f}".ljust(15)) # e.g. 0.1234
                else:
                    row_parts.append(f"{agent_val:.4f}".ljust(15))
            
            # SP Baseline
            if key in sp_res:
                val = sp_res[key]
                row_parts.append(f"{val:.4f}".ljust(15))
            else: 
                row_parts.append("-".ljust(15))
                
            # Oracle
            if key in opt_res:
                val = opt_res[key]
                row_parts.append(f"{val:.4f}".ljust(15))
            else:
                 row_parts.append("-".ljust(15))
            
            print(" ".join(row_parts))
        
        # Improvement & GAP SCORE
        print("\n" + "="*110)
        print("PERFORMANCE COMPARISON (Gap Closure Score)")
        print("="*110)
        
        sp_mean = sp_res['mlu_mean']
        opt_mean = opt_res['mlu_mean']
        max_possible_gain = sp_mean - opt_mean
        
        print(f"Max Possible Gain (SP - Oracle): {max_possible_gain:.4f}")
        
        for agent_name in sorted(results_dict.keys()):
            agent_mean = results_dict[agent_name]['mlu_mean']
            agent_gain = sp_mean - agent_mean
            
            # Gap Score Calculation
            # Score = (SP - Agent) / (SP - Optimal) * 100
            if max_possible_gain > 1e-9:
                gap_score = (agent_gain / max_possible_gain) * 100
            else:
                gap_score = 0.0 # No gain possible
                
            print(f"\n{agent_name}:")
            print(f"  MLU Improvement: {agent_gain:.4f}")
            print(f"  Gap Score:       {gap_score:.2f}% (Percentage of optimality gap closed)")
            
            if gap_score > 0:
                 print(f"  ✅ {agent_name} closes {gap_score:.1f}% of the gap to optimal.")
        
        print("\n" + "="*110)

    def run(self):
        print("="*90)
        print("MLU BENCHMARK - Evaluating Agents vs Baseline vs Oracle")
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
            env = DeflectionEnv(tfrecords_dir=self.tfrecords_dir, traffic_intensity=self.traffic_intensity)
            try:
                model_inst = self.model_instance if self.agent_type == 'PPO' else None
                res = self.evaluate_agent_mlu(self.ppo_path, env, samples, 'PPO', model_instance=model_inst)
                results_dict['PPO'] = res
            except Exception as e:
                print(f"Failed to evaluate PPO: {e}")
                
        if self.agent_type in ['MaskPPO', 'all']:
            print(f"\nInitializing environment for MaskPPO ({self.env_type})...")
            if self.env_type == "masked":
                env = MaskedDeflectionEnv(tfrecords_dir=self.tfrecords_dir, traffic_intensity=self.traffic_intensity)
            else:
                env = DeflectionEnv(tfrecords_dir=self.tfrecords_dir, traffic_intensity=self.traffic_intensity)
            try:
                model_inst = self.model_instance if self.agent_type == 'MaskPPO' else None
                res = self.evaluate_agent_mlu(self.maskppo_path, env, samples, 'MaskPPO', model_instance=model_inst)
                results_dict['MaskPPO'] = res
            except Exception as e:
                print(f"Failed to evaluate MaskPPO: {e}")
                import traceback
                traceback.print_exc()

        # Baseline & Oracle
        print(f"\nInitializing environment for Baselines...")
        env = DeflectionEnv(tfrecords_dir=self.tfrecords_dir, traffic_intensity=self.traffic_intensity)
        baseline_results = self.evaluate_baselines(env, samples)
        
        if results_dict:
            self.print_results(results_dict, baseline_results)
        else:
            print("No agent results collected.")
