import sys
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tqdm import tqdm
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.env import KDNEnvinronment
from src.gcn import GCNFeatureExtractor

def import_nx():
    import networkx as nx
    return nx

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def collect_data(env, num_samples=2000):
    print(f"Collecting {num_samples} samples...")
    observations = []
    mlu_targets = []
    
    max_iters = num_samples * 10
    iters = 0
    
    pbar = tqdm(total=num_samples)
    
    while len(observations) < num_samples and iters < max_iters:
        iters += 1
        obs, info = env.reset()
        
        # Collect both Opt and SP if valid
        paths_to_eval = []
        if env.optimal_path:
             paths_to_eval.append(env.optimal_path)
             
        # SP
        try:
             G = env.sample.topology_object
             sp = list(next(import_nx().all_shortest_paths(G, env.current_node, env.destination)))
             if sp != env.optimal_path:
                 paths_to_eval.append(sp)
        except:
            pass
            
        if not paths_to_eval:
             continue
             
        base_obs = env._get_obs()
        
        for path in paths_to_eval:
            link_loads = env._calculate_background_loads(env.current_node, env.destination)
            mlu = env._calculate_max_utilization(path, link_loads)
            
            # Construct obs dict
            path_arr = np.full(env.max_steps + 1, -1, dtype=int)
            plen = min(len(path), env.max_steps + 1)
            path_arr[:plen] = path[:plen]
            
            new_obs = {
                "destination": base_obs["destination"].copy(),
                "traffic": base_obs["traffic"].copy(),
                "topology": base_obs["topology"].copy(),
                "link_utilization": base_obs["link_utilization"].copy(),
                "path": path_arr
            }
            observations.append(new_obs)
            mlu_targets.append(mlu)
            
            if len(observations) >= num_samples:
                break
                
        pbar.update(len(paths_to_eval))
    
    pbar.close()
    return observations, mlu_targets

def get_features(extractor, raw_obs, batch_size=256):
    # Collate
    def collate(obs_list):
        batch = {k: [] for k in obs_list[0].keys()}
        for o in obs_list:
            for k in batch:
                batch[k].append(o[k])
        for k in batch:
            batch[k] = torch.tensor(np.array(batch[k]), dtype=torch.float32)
        return batch

    features_list = []
    with torch.no_grad():
        for i in range(0, len(raw_obs), batch_size):
            batch = collate(raw_obs[i:i+batch_size])
            feats = extractor(batch)
            features_list.append(feats.numpy())
            
    return np.concatenate(features_list, axis=0)

def run_tuning():
    data_dir = 'data/nsfnetbw'
    if not os.path.exists(data_dir):
        # Handle running from root or scripts/
        if os.path.exists('../data/nsfnetbw'):
            data_dir = '../data/nsfnetbw'
            
    env = KDNEnvinronment(tfrecords_dir=data_dir, traffic_intensity=9)
    observation_space = env.observation_space
    
    # 1. Collect Data Once
    raw_obs, targets = collect_data(env, num_samples=2000)
    y = np.array(targets)
    
    # Grid Search Space
    # Constraint: n_layers = 2
    # Constraint: Fit well with PPO [64, 64] -> Small output dim needed.
    
    configs = []
    # Test low hidden dims for direct fit (out_dim=None)
    for h in [4, 8, 16, 32, 64]:
         configs.append({'hidden_dim': h, 'n_layers': 2, 'out_dim': None})

    # Test projection to 64
    for h in [32, 64, 128]:
         configs.append({'hidden_dim': h, 'n_layers': 2, 'out_dim': 64})
         
    results = []
    
    # Calculate effective output dim for display
    # Assuming N=14 from NSFNetBW
    N = 14
    
    print(f"\nEvaluating {len(configs)} configurations...\n")
    print(f"{'Hidden':<8} | {'Layers':<8} | {'OutDim':<8} | {'EffDim':<8} | {'Params':<10} | {'R^2':<8}")
    print("-" * 75)
    
    for cfg in configs:
        h, l, o = cfg['hidden_dim'], cfg['n_layers'], cfg['out_dim']
        
        # Build Model
        extractor = GCNFeatureExtractor(
            observation_space,
            hidden_dim=h,
            n_layers=l,
            out_dim=o
        )
        
        # Effective Dim
        eff_dim = extractor._features_dim
        
        # Count Params
        n_params = count_parameters(extractor)
        
        # Extract Features
        X = get_features(extractor, raw_obs)
        
        # Train Probe (Ridge)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        
        results.append({
            'hidden_dim': h,
            'n_layers': l,
            'out_dim': str(o),
            'eff_dim': eff_dim,
            'params': n_params,
            'r2': score
        })
        
        print(f"{h:<8} | {l:<8} | {str(o):<8} | {eff_dim:<8} | {n_params:<10} | {score:.4f}")

    # Find Pareto Front / Best Tradeoff
    # Sort by R2 descending
    df = pd.DataFrame(results)
    df = df.sort_values(by='r2', ascending=False)
    
    print("\nTop 5 Configurations by Accuracy:")
    print(df.head(5))
    
    print("\nTop 5 Efficient Configurations (R^2 > 0.9, Smallest Params):")
    efficient = df[df['r2'] > 0.9].sort_values(by='params')
    print(efficient.head(5))

if __name__ == "__main__":
    run_tuning()
