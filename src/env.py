import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
from src.datanet import Datanet

class DeflectionEnv(gym.Env):
    """
    Environment where the agent iteratively removes edges to force rerouting
    to find an optimal path with minimal MLU.
    Action Space: Discrete(num_edges) - Remove edge at index.
    Observation Space: Dict containing traffic, path, and edge features.
    """
    def __init__(self, tfrecords_dir=None, traffic_intensity=15, max_steps=20, prefiltered_samples=None):
        super().__init__()
        
        if not tfrecords_dir:
            raise ValueError("tfrecords_dir is required")
            
        self.max_steps = max_steps
        self.prefiltered_samples = prefiltered_samples
        
        # Initialize Datanet reader
        self.reader = Datanet(tfrecords_dir, intensity_values=[traffic_intensity])
        self.iterator = iter(self.reader)
        
        # Load initial sample to define spaces
        if self.prefiltered_samples and len(self.prefiltered_samples) > 0:
            self.sample = self.prefiltered_samples[0][0]
        else:
            self.sample = next(self.iterator)
        
        # Original topology edges for action space definition
        # We collect (u, v, key) to handle multigraphs correctly
        self.all_edges = list(self.sample.topology_object.edges(keys=True))
        self.action_space = spaces.Discrete(len(self.all_edges))
        
        n = self.sample.get_network_size()
        k = len(self.all_edges)
        
        self.observation_space = spaces.Dict({
            "traffic_demand": spaces.Box(low=0, high=np.inf, shape=(n, n), dtype=np.float32),
            "path": spaces.Box(low=-1, high=n, shape=(self.max_steps + 1,), dtype=int),
            # Action-Centric Observations
            "edge_endpoints": spaces.Box(low=0, high=n, shape=(k, 2), dtype=int),
            "edge_features": spaces.Box(low=0, high=np.inf, shape=(k, 4), dtype=np.float32), 
            # Features: [Capacity, Bg_Util, Is_Active, Is_In_Path]
            "maxAvgLambda": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
        })
        
        # Pre-compute edge endpoints (Static for the environment instance if topology structure is fixed)
        self.edge_endpoints = np.array([(u, v) for u, v, k in self.all_edges], dtype=int)
        
        self.current_topology = None
        self.src = 0
        self.dst = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Checking for options (used in Benchmarking to force specific sample/src/dst)
        if options and "sample" in options:
            self.sample = options["sample"]
            if "src" in options and "dst" in options:
                src = options["src"]
                dst = options["dst"]
            else:
                n = self.sample.get_network_size()
                flat_idx = self.np_random.integers(0, n * (n - 1))
                src = flat_idx // (n - 1)
                dst_offset = flat_idx % (n - 1)
                dst = dst_offset if dst_offset < src else dst_offset + 1
                 
        elif self.prefiltered_samples and len(self.prefiltered_samples) > 0:
            idx = self.np_random.integers(0, len(self.prefiltered_samples))
            self.sample, src, dst = self.prefiltered_samples[idx]
        else:
            while True:
                try:
                    self.sample = next(self.iterator)
                except StopIteration:
                    self.iterator = iter(self.reader)
                    self.sample = next(self.iterator)
                
                n = self.sample.get_network_size()
                flat_idx = self.np_random.integers(0, n * (n - 1))
                src = flat_idx // (n - 1)
                dst_offset = flat_idx % (n - 1)
                dst = dst_offset if dst_offset < src else dst_offset + 1
                break

        self.src = src
        self.dst = dst
        self.current_step = 0
        
        # Initialize current topology as a copy of the original
        self.current_topology = self.sample.topology_object.copy()
        
        # Calculate initial path on full topology
        try:
            self.current_path = nx.shortest_path(self.current_topology, src, dst, weight='weight')
        except nx.NetworkXNoPath:
            self.current_path = [src] 
        
        # Calculate initial MLU on ORIGINAL topology
        self.bg_loads = self.sample.calculate_background_loads(src, dst)
        self.current_mlu = self.sample.calculate_max_utilization(self.current_path, self.bg_loads)
        self.original_mlu = self.current_mlu
        self.min_mlu_so_far = self.current_mlu
        self.best_path_so_far = list(self.current_path)
        
        # Calculate Optimal Path for this request
        self.optimal_path = self.sample.search_optimal_path(src, dst, self.bg_loads, max_steps=100)
        if self.optimal_path:
            self.optimal_mlu = self.sample.calculate_max_utilization(self.optimal_path, self.bg_loads)
        else:
            self.optimal_mlu = None
        
        return self._get_obs(), {}

    def step(self, action):
        truncated = False
        terminated = False
        info = {}
        
        # 0. Check if we are already optimal (e.g. from reset)
        if self.optimal_mlu is not None and self.current_mlu <= self.optimal_mlu + 1e-5:
             # Already optimal
             peak_loss_rate = 0.0
             if self.current_mlu > 1.0:
                 peak_loss_rate = 1.0 - (1.0 / self.current_mlu)
                 
             return self._get_obs(), 0.0, True, False, {
                 "reached_optimal_mlu": True, 
                 "is_success": True, 
                 "is_optimal": True,
                 "mlu": self.current_mlu,
                 "gap_score": 100.0,
                 "peak_loss_rate": peak_loss_rate
             }

        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = True

        # Action: Remove edge
        u, v, key = self.all_edges[action]
        
        top_modified = False
        
        if self.current_topology.has_edge(u, v, key=key):
            self.current_topology.remove_edge(u, v, key=key)
            top_modified = True
            
        if not top_modified:
            # Illegal action (removing already removed edge or non-existent)
            return self._get_obs(), -0.1, False, truncated, {"invalid_action": True}

        # Check connectivity
        if not nx.has_path(self.current_topology, self.src, self.dst):
             terminated = True
             reward = -10.0 # Heavy penalty for disconnecting src-dst
             info["is_success"] = False
             return self._get_obs(), reward, terminated, truncated, info

        # Find new Shortest Path
        try:
            new_path = nx.shortest_path(self.current_topology, self.src, self.dst, weight='weight')
        except nx.NetworkXNoPath:
             terminated = True
             reward = -10.0
             return self._get_obs(), reward, terminated, truncated, info

        # Calculate Reward
        current_mlu = self.sample.calculate_max_utilization(new_path, self.bg_loads)
        
        # Dense Reward Signal
        gap = self.original_mlu - current_mlu
        reward = gap * 10.0
        
        # Update best so far
        if current_mlu < self.min_mlu_so_far:
            self.min_mlu_so_far = current_mlu
            self.best_path_so_far = list(new_path)
            
        # Check success condition
        if self.original_mlu - current_mlu > 1e-4:
            info["is_success"] = True
            
        # Update state
        self.current_mlu = current_mlu
        self.current_path = new_path
        
        info["mlu"] = current_mlu
        improvement_val = self.original_mlu - current_mlu
        info["improvement"] = improvement_val
        info["is_success"] = (improvement_val > 1e-4)
        
        # Calculate Gap Score
        gap_score = 0.0
        if self.optimal_mlu is not None:
            max_gain = self.original_mlu - self.optimal_mlu
            agent_gain = improvement_val
            
            if max_gain > 1e-9:
                gap_score = (agent_gain / max_gain) * 100.0
            else:
                if agent_gain >= -1e-9:
                    gap_score = 100.0
                else:
                    gap_score = 0.0
        
        info["gap_score"] = gap_score
        
        # Calculate Peak Loss Rate
        peak_loss_rate = 0.0
        if current_mlu > 1.0:
            peak_loss_rate = 1.0 - (1.0 / current_mlu)
        info["peak_loss_rate"] = peak_loss_rate

        # Check if we found the optimal path
        if self.optimal_mlu is not None:
             info["is_optimal"] = (self.current_mlu <= self.optimal_mlu + 1e-5)
        else:
             info["is_optimal"] = False
        
        # Terminate if no valid actions left
        if hasattr(self, "action_masks"):
            masks = self.action_masks()
            if not np.any(masks):
                terminated = True
        
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        n = self.sample.get_network_size()
        
        # Traffic
        tm = self.sample.traffic_matrix
        traffic = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                if i != j:
                    traffic[i, j] = tm[i, j].get('AggInfo', {}).get('AvgBw', 0.0)

        # Path
        path_arr = np.full(self.max_steps + 1, -1, dtype=int)
        path_len = min(len(self.current_path), self.max_steps + 1)
        path_arr[:path_len] = self.current_path[:path_len]
        
        # Edge Features
        k = len(self.all_edges)
        edge_features = np.zeros((k, 4), dtype=np.float32)
        
        orig_G = self.sample.topology_object
        
        for idx, (u, v, key) in enumerate(self.all_edges):
            # 1. Capacity
            try:
                cap = float(orig_G[u][v][key]['bandwidth'])
            except:
                cap = 1.0
            
            # 2. Background Utilization
            load = self.bg_loads.get((u, v), 0.0)
            if cap > 0:
                util = load / cap
            else:
                util = 0.0
                
            # 3. Is Active
            is_active = 1.0 if self.current_topology.has_edge(u, v, key=key) else 0.0
            
            # 4. Is In Current Path
            in_path = 0.0
            for i in range(len(self.current_path) - 1):
                if (u == self.current_path[i] and v == self.current_path[i+1]):
                    in_path = 1.0
                    break
            
            edge_features[idx] = [cap, util, is_active, in_path]

        return {
            "traffic_demand": traffic,
            "path": path_arr,
            "edge_endpoints": self.edge_endpoints,
            "edge_features": edge_features,
            "maxAvgLambda": np.array([self.sample.get_maxAvgLambda()], dtype=np.float32),
        }


