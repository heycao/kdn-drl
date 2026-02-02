import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
from src.datanet import Datanet


class KDNEnvinronment(gym.Env):
    """
    Routing environment using Datanet samples as data proxy.
    Action Space: Discrete(num_edges)
    """
    def __init__(self, tfrecords_dir=None, traffic_intensity=9, max_steps=100, data_filter="all", prefiltered_samples=None):
        super().__init__()
        
        if not tfrecords_dir:
            raise ValueError("tfrecords_dir is required")
            
        self.max_steps = max_steps
        self.prefiltered_samples = prefiltered_samples
        
        self.reader = Datanet(tfrecords_dir, intensity_values=[traffic_intensity])
        self.iterator = iter(self.reader)
        
        if self.prefiltered_samples and len(self.prefiltered_samples) > 0:
            # Use the first filtered sample for space definition
            self.sample = self.prefiltered_samples[0][0]
        else:
            self.sample = next(self.iterator)
        
        # Define Action Space
        self.edges = [(u, v, 0) for u, v in self.sample.topology_object.edges()]
        self.action_space = spaces.Discrete(len(self.edges))
        
        # Define Observation Space
        n = self.sample.get_network_size()
        self.observation_space = spaces.Dict({
            "destination": spaces.Box(low=0, high=n, shape=(1,), dtype=int),
            "traffic": spaces.Box(low=0, high=np.inf, shape=(n, n), dtype=np.float32),
            "topology": spaces.Box(low=0, high=np.inf, shape=(n, n), dtype=np.float32),
            "link_utilization": spaces.Box(low=0, high=np.inf, shape=(n, n), dtype=np.float32),
            "path": spaces.Box(low=-1, high=n, shape=(self.max_steps + 1,), dtype=int),
        })
        
        # Restart iterator to include first sample
        self.iterator = iter(self.reader)
        
        # Episode state
        self.current_node = 0
        self.destination = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.prefiltered_samples and len(self.prefiltered_samples) > 0:
             # Use pre-filtered data: list of (sample, src, dst)
             idx = self.np_random.integers(0, len(self.prefiltered_samples))
             self.sample, src, dst = self.prefiltered_samples[idx]
        else:
            # Fallback to random sampling from iterator (No Filter)
            while True:
                # Sample next traffic scenario
                try:
                    self.sample = next(self.iterator)
                except StopIteration:
                    self.iterator = iter(self.reader)
                    self.sample = next(self.iterator)
                
                # Select random source-destination pair
                n = self.sample.get_network_size()
                flat_idx = self.np_random.integers(0, n * (n - 1))
                src = flat_idx // (n - 1)
                dst_offset = flat_idx % (n - 1)
                dst = dst_offset if dst_offset < src else dst_offset + 1
                
                break # No filtering here anymore
            
        self.current_node = src
        self.destination = dst
        self.path = [src]
        self.current_step = 0
        
        # Pre-calculate Optimal Path (using Sample methods)
        self.bg_loads = self.sample.calculate_background_loads(src, dst)
        self.optimal_path = self.sample.search_optimal_path(src, dst, self.bg_loads, self.max_steps)
        
        # Pre-calculate Shortest Path (Weights) for Improvement Logging
        try:
             # Match BenchmarkRunner baseline logic (uses weight='weight' if available)
             # 'weight' in nsfnetbw typically means latency/cost. 
             # If weight='weight' is not there, it defaults to hops.
             self.sp_path = nx.shortest_path(self.sample.topology_object, src, dst, weight='weight')
             self.sp_mlu = self.sample.calculate_max_utilization(self.sp_path, self.bg_loads)
        except nx.NetworkXNoPath:
             self.sp_path = None
             self.sp_mlu = 1.0 # Or some penalty
        
        # Keep shortest_path (hops) for is_optimal_shortest check
        try:
             self.shortest_path = nx.shortest_path(self.sample.topology_object, src, dst, weight=None)
        except nx.NetworkXNoPath:
             self.shortest_path = None

        return self._get_obs(), {}


    def step(self, action):
        """Takes an action (edge index) and moves the agent if valid."""
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        
        # 1. Action Validation
        u, v, _ = self.edges[action]
        if u != self.current_node:
            return self._get_obs(), -10.0, False, truncated, {"invalid_action": True}
        
        # 2. Update State
        
        # Check for Infinite Loop / Dead End Revisit
        if v in self.path:
            # Cycle detected. Terminate immediately.
            terminated = True
            reward = 0.0
            info = {"dead_end": True, "mlu": 0.0, "is_success": False} # mlu 0 since invalid
            
            self.current_node = v # Technically we moved? Or just stay? 
            # If we terminate, state update matters less, but let's record the move for consistency.
            self.path.append(v)
            
            return self._get_obs(), reward, terminated, truncated, info

        self.current_node = v
        self.path.append(v)
        
        # 3. Check Termination
        terminated = False
        reward = 0.0
        info = {}
        
        if self.current_node == self.destination:
            terminated = True
            
            # --- Reward Calculation ---
            src, dst = self.path[0], self.destination
            
            # A. Calculate Background Loads (all other flows)
            bg_loads = self.sample.calculate_background_loads(src, dst)
            
            # B. Calculate Agent MLU
            agent_mlu = self.sample.calculate_max_utilization(self.path, bg_loads)

            
            # C. Calculate Reward
            # Base success reward + efficiency bonus
            # Scale (1.0 - agent_mlu) so it's positive and significant
            # agent_mlu is typically [0, 1]
            reward = 1.0 - agent_mlu
            
            # D. Logging Metrics
            info["mlu"] = agent_mlu
            info["is_success"] = True
            info["path_length"] = len(self.path)
            
            # Add beat_baseline reward
            is_optimal = False
            if self.optimal_path and self.path == self.optimal_path:
                reward = 1.0
                is_optimal = True
            
            info["is_optimal"] = is_optimal
            
            # Optimal Shortest Path check
            is_optimal_shortest = False
            if is_optimal and self.shortest_path and len(self.path) == len(self.shortest_path):
                 if self.path == self.shortest_path:
                      is_optimal_shortest = True
            info["is_optimal_shortest"] = is_optimal_shortest
            
            # Improvement vs SP
            # Positive means agent is better (lower MLU)
            if hasattr(self, 'sp_mlu') and self.sp_mlu is not None and self.sp_mlu > 0:
                info["mlu_improvement"] = (self.sp_mlu - agent_mlu) / self.sp_mlu
            else:
                info["mlu_improvement"] = 0.0

        else:
            # Step reward
            # User Request: Only reward if self.current_node == self.destination
            # Meaning no intermediate rewards.
            reward = 0.0
            
        return self._get_obs(), reward, terminated, truncated, info



    def _get_obs(self):
        tm = self.sample.traffic_matrix
        n = self.sample.get_network_size()
        
        # NxN Traffic Matrix
        traffic = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                if i != j:
                    traffic[i, j] = tm[i, j]['AggInfo']['AvgBw']
        
        # Topology Matrix
        topology = np.zeros((n, n), dtype=np.float32)
        for u, v, data in self.sample.topology_object.edges(data=True):
            if 'bandwidth' in data:
                topology[u, v] = float(data['bandwidth'])

        # Pad path to fixed length
        path_arr = np.full(self.max_steps + 1, -1, dtype=int)
        path_len = min(len(self.path), self.max_steps + 1)
        path_arr[:path_len] = self.path[:path_len]
        
        # Link Utilization (Background)
        link_util = np.zeros((n, n), dtype=np.float32)
        if hasattr(self, 'bg_loads'):
             for (u, v), load in self.bg_loads.items():
                 # Cap is in topology
                 cap_data = self.sample.topology_object[u][v][0]['bandwidth']
                 cap = float(cap_data)
                 if cap > 0:
                     link_util[u, v] = load / cap

        return {
            "destination": np.array([self.destination], dtype=int),
            "traffic": traffic,
            "topology": topology,
            "path": path_arr,
            "link_utilization": link_util,
        }
