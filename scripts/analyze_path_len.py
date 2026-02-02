import sys
import os
import networkx as nx
from tqdm import tqdm
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.env import KDNEnvinronment

def run_analysis():
    data_dir = 'data/nsfnetbw'
    if not os.path.exists(data_dir):
        if os.path.exists('../data/nsfnetbw'):
            data_dir = '../data/nsfnetbw'
            
    env = KDNEnvinronment(tfrecords_dir=data_dir, traffic_intensity=9)
    G = env.sample.topology_object
    
    # 1. Graph Statistics
    try:
        diameter = nx.diameter(G)
        avg_shortest_path = nx.average_shortest_path_length(G)
    except:
        diameter = "N/A (Disconnected?)"
        avg_shortest_path = "N/A"
        
    print(f"\nGraph Statistics (NSFNet):")
    print(f"Nodes: {len(G.nodes)}")
    print(f"Edges: {len(G.edges)}")
    print(f"Diameter: {diameter}")
    print(f"Avg Shortest Path Len: {avg_shortest_path}")
    
    # 2. Collect Data (Opt vs SP)
    print(f"\nCollecting Samples...")
    data = []
    num_samples = 500
    
    iters = 0
    with tqdm(total=num_samples) as pbar:
        while len(data) < num_samples and iters < 10000:
            iters += 1
            obs, info = env.reset()
            src, dst = env.current_node, env.destination
            
            opt_path = env.optimal_path
            
            try:
                sp_path = list(next(nx.all_shortest_paths(G, src, dst)))
            except:
                continue
                
            if not opt_path: continue
            
            # Filter for Hard cases where Opt != SP
            if opt_path == sp_path:
                continue
            
            data.append({
                'sp_len': len(sp_path) - 1, # Hops = Nodes - 1
                'opt_len': len(opt_path) - 1,
                'len_diff': len(opt_path) - len(sp_path)
            })
            pbar.update(1)
            
    df = pd.DataFrame(data)
    
    print("\nPath Length Analysis (Hard Cases: Opt != SP):")
    print(f"Avg SP Hops:  {df['sp_len'].mean():.2f}")
    print(f"Avg Opt Hops: {df['opt_len'].mean():.2f}")
    print(f"Avg Diff:     {df['len_diff'].mean():.2f}")
    
    # Check Receptive Field
    print(f"\nReceptive Field Analysis:")
    # Percentage of cases where Opt Path Length > n_layers
    # If len > n_layers, GCN(n_layers) might not propagate destination info to source?
    # Actually, GCN propagates info for K steps.
    # If Distance(Src, Dst) > K, then Src's embedding contains NO info about Dst?
    # Let's check for K=2 and K=3
    
    for k in [2, 3, 4]:
        blind_cases = df[df['sp_len'] > k] # Even SP is longer than K
        print(f"Cases where Distance(Src, Dst) > {k} (Blind Spot?): {len(blind_cases)} / {len(df)} ({len(blind_cases)/len(df)*100:.1f}%)")
        
    print("\nInterpretation:")
    print("If GCN layers (K) < Distance, the Source node embedding theoretically receives ZERO signal from Destination node feature.")
    print("If the agent relies solely on 'Is Dest' feature propagating through GCN, it is blind for distance > K.")
    
if __name__ == "__main__":
    run_analysis()
