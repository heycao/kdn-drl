import sys
import os
import numpy as np
import networkx as nx
from tqdm import tqdm
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.env import KDNEnvinronment

def analyze_differences(env, num_samples=1000):
    print(f"Collecting {num_samples} hard samples (Opt != SP)...")
    
    data = []
    
    iters = 0
    max_iters = num_samples * 20
    
    with tqdm(total=num_samples) as pbar:
        while len(data) < num_samples and iters < max_iters:
            iters += 1
            obs, info = env.reset()
            
            src = env.current_node
            dst = env.destination
            G = env.sample.topology_object
            
            # 1. Get Optimal Path
            opt_path = env.optimal_path
            if not opt_path: continue
            
            # 2. Get Shortest Path (Hops)
            try:
                sp_path = list(next(nx.all_shortest_paths(G, src, dst)))
            except:
                continue
                
            # Filter: We only care when Opt != SP
            if opt_path == sp_path:
                continue
                
            # Calculate Metrics
            bg_loads = env.bg_loads # Pre-calculated in reset
            
            def get_path_metrics(path):
                # Max Link Utilization (The Objective)
                mlu = env._calculate_max_utilization(path, bg_loads)
                
                # Path Length
                hops = len(path)
                
                # Bottleneck Analysis
                max_bg_util = 0.0
                min_cap = float('inf')
                sum_bg_util = 0.0
                
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    
                    # Capacity
                    cap = float(G[u][v][0]['bandwidth'])
                    min_cap = min(min_cap, cap)
                    
                    # BG Load
                    load = bg_loads.get((u, v), 0.0)
                    bg_u = load / cap if cap > 0 else 0
                    
                    max_bg_util = max(max_bg_util, bg_u)
                    sum_bg_util += bg_u
                    
                avg_bg_util = sum_bg_util / max(1, (len(path) - 1))
                
                return {
                    'mlu': mlu,
                    'hops': hops,
                    'max_bg_util': max_bg_util,
                    'avg_bg_util': avg_bg_util,
                    'min_cap': min_cap
                }

            opt_metrics = get_path_metrics(opt_path)
            sp_metrics = get_path_metrics(sp_path)
            
            # Store difference
            data.append({
                'opt_mlu': opt_metrics['mlu'],
                'sp_mlu': sp_metrics['mlu'],
                'mlu_diff': sp_metrics['mlu'] - opt_metrics['mlu'], # How much worse SP is
                
                'opt_hops': opt_metrics['hops'],
                'sp_hops': sp_metrics['hops'],
                'hops_diff': opt_metrics['hops'] - sp_metrics['hops'], # How much longer Opt is
                
                'opt_max_bg': opt_metrics['max_bg_util'],
                'sp_max_bg': sp_metrics['max_bg_util'],
                'max_bg_diff': sp_metrics['max_bg_util'] - opt_metrics['max_bg_util'], # How clearer Opt is
                
                'opt_min_cap': opt_metrics['min_cap'],
                'sp_min_cap': sp_metrics['min_cap'],
            })
            
            pbar.update(1)
            
    df = pd.DataFrame(data)
    return df

def run_analysis():
    data_dir = 'data/nsfnetbw'
    if not os.path.exists(data_dir):
        if os.path.exists('../data/nsfnetbw'):
            data_dir = '../data/nsfnetbw'
            
    env = KDNEnvinronment(tfrecords_dir=data_dir, traffic_intensity=9)
    
    df = analyze_differences(env, num_samples=500)
    
    print("\n" + "="*50)
    print("Analysis of 500 Hard Samples (Opt != SP)")
    print("="*50)
    
    print("\n1. MLU Improvement (Why Opt is better)")
    print(f"Avg SP MLU:  {df['sp_mlu'].mean():.4f}")
    print(f"Avg Opt MLU: {df['opt_mlu'].mean():.4f}")
    print(f"Avg Reduction: {df['mlu_diff'].mean():.4f}")
    
    print("\n2. Cost of Optimality (Path Length)")
    print(f"Avg SP Hops:  {df['sp_hops'].mean():.4f}")
    print(f"Avg Opt Hops: {df['opt_hops'].mean():.4f}")
    print(f"Avg Hop Increase: {df['hops_diff'].mean():.4f}")
    
    print("\n3. Key Feature: Max Background Utilization (Bottleneck)")
    print(f"Avg SP Max BG Util:  {df['sp_max_bg'].mean():.4f}")
    print(f"Avg Opt Max BG Util: {df['opt_max_bg'].mean():.4f}")
    print(f"Avg Difference:      {df['max_bg_diff'].mean():.4f}")
    
    correlation = df['mlu_diff'].corr(df['max_bg_diff'])
    print(f"\nCorrelation Check: Is MLU improvement correlated with avoiding High BG Util links?")
    print(f"Correlation(MLU Diff, Max BG Util Diff): {correlation:.4f}")
    
    print("\nSummary:")
    if df['max_bg_diff'].mean() > 0.1:
        print("Feature Identified: 'Max Link Utilization (Background)'")
        print("Shortest Paths consistently hit links that are ALREADY congested (High BG Util).")
        print("Optimal Paths detour to avoid these specific links.")
        print("Recommendation: Ensure GCN emphasizes 'bg_util_out_max' or 'link_utilization' features.")
    else:
        print("Feature Unclear. Bottleneck might be mostly due to the Flow itself (Small Capacity).")
        
    print("-" * 50)
    print("Sample Data (First 5):")
    print(df[['sp_mlu', 'opt_mlu', 'sp_max_bg', 'opt_max_bg', 'hops_diff']].head())

if __name__ == "__main__":
    run_analysis()
