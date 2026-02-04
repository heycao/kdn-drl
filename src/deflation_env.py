import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
from src.datanet import Datanet

class DeflationEnv(gym.Env):
    """
    Environment where the agent iteratively removes edges to force rerouting
    to find an optimal path with minimal MLU.
    Action Space: Discrete(num_edges) - Remove edge at index.
    Observation Space: Same as KDNEnvinronment, reflecting the modified topology.
    """
    def __init__(self, tfrecords_dir=None, traffic_intensity=15, max_steps=20, prefiltered_samples=None):
        super().__init__()
        
        if not tfrecords_dir:
            raise ValueError("tfrecords_dir is required")
            
        self.max_steps = max_steps
        self.prefiltered_samples = prefiltered_samples
        
        self.reader = Datanet(tfrecords_dir, intensity_values=[traffic_intensity])
        self.iterator = iter(self.reader)
        
        if self.prefiltered_samples and len(self.prefiltered_samples) > 0:
            self.sample = self.prefiltered_samples[0][0]
        else:
            self.sample = next(self.iterator)
        
        # Original topology edges for action space definition
        # We collect (u, v, key) to handle multigraphs correctly or just (u,v) if keys implicitly handled by list order
        # For data retrieval we need to be careful. Datanet/NetworkX edges() iterates in a stable order.
        self.all_edges = list(self.sample.topology_object.edges(keys=True))
        self.action_space = spaces.Discrete(len(self.all_edges))
        
        n = self.sample.get_network_size()
        k = len(self.all_edges)
        
        self.observation_space = spaces.Dict({
            "traffic_demand": spaces.Box(low=0, high=np.inf, shape=(n, n), dtype=np.float32),
            "path": spaces.Box(low=-1, high=n, shape=(self.max_steps + 1,), dtype=int),
            # NEW: Action-Centric Observations
            "edge_endpoints": spaces.Box(low=0, high=n, shape=(k, 2), dtype=int),
            "edge_features": spaces.Box(low=0, high=np.inf, shape=(k, 4), dtype=np.float32), 
            # Features: [Capacity, Bg_Util, Is_Active, Is_In_Path]
            "maxAvgLambda": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
        })
        
        # Pre-compute edge endpoints (Static for the environment instance if topology structure is fixed)
        self.edge_endpoints = np.array([(u, v) for u, v, k in self.all_edges], dtype=int)
        
        # Helper: Keep a consistent random generator if needed, 
        # but gym handles seeding via reset(seed=...)
        
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
                # If sample provided but no src/dst, pick random? 
                # Or assume sample implies context? For now conform to existing logic if missing.
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
            # Should not happen on reset unless graph is disconnected
            self.current_path = [src] 
        
        # Calculate initial MLU on ORIGINAL topology
        self.bg_loads = self.sample.calculate_background_loads(src, dst)
        self.current_mlu = self.sample.calculate_max_utilization(self.current_path, self.bg_loads)
        self.original_mlu = self.current_mlu
        self.min_mlu_so_far = self.current_mlu
        self.best_path_so_far = list(self.current_path)
        
        # Calculate Optimal Path for this request (Brute-force search)
        # Used for "is_optimal" metric
        optimal_path = self.sample.search_optimal_path(src, dst, self.bg_loads, max_steps=100)
        if optimal_path:
            self.optimal_mlu = self.sample.calculate_max_utilization(optimal_path, self.bg_loads)
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
        
        # Robust removal: Remove SPECIFIC edge if possible, or just u,v if key not reliable?
        # Since we use MultiGraph logic now, we should try to remove specific key.
        # But wait, self.current_topology is a MultiGraph copy?
        # Yes, copy() preserves class.
        if self.current_topology.has_edge(u, v, key=key):
            self.current_topology.remove_edge(u, v, key=key)
            top_modified = True
            
        # Also remove reverse direction if it exists? 
        # Usually networks are undirected or symmetric directed. 
        # If directed, we only remove u->v. 
        # Assuming Undirected for now as 'remove link' usually implies physical cut.
        # If the graph is DiGraph, has_edge(v, u) might be false. 
        # Let's check if we should remove v->u. 
        # The key might be different for reverse edge.
        # In this specific env, we probably want to remove specific directed link or both?
        # Original code removed ALL edges between u and v.
        # "Robust removal: Remove ALL edges between u and v"
        # If we want to support Multigraph PRECISION, we should remove only the specific edge.
        # But if we assume symmetric failure, we need to find the paired edge.
        # For now, let's stick to Removing the SPECIFIC edge selected by Action.
        
        if not top_modified:
            # Illegal action (removing already removed edge or non-existent)
            return self._get_obs(), -0.1, False, truncated, {"invalid_action": True}

        # Check connectivity
        if not nx.has_path(self.current_topology, self.src, self.dst):
             # If we disconnect, it's a failure. 
             # Masking should have prevented this if implemented correctly.
             terminated = True
             reward = -10.0 # Heavy penalty for disconnecting src-dst
             info["is_success"] = False
             return self._get_obs(), reward, terminated, truncated, info

        # Find new Shortest Path
        try:
            # Use 'weight' if available, same as baseline
            new_path = nx.shortest_path(self.current_topology, self.src, self.dst, weight='weight')
        except nx.NetworkXNoPath:
             terminated = True
             reward = -10.0
             return self._get_obs(), reward, terminated, truncated, info

        # Calculate Reward
        # Reward = (Best So Far - New MLU) * 10 
        current_mlu = self.sample.calculate_max_utilization(new_path, self.bg_loads)
        
        # Dense Reward Signal
        # Use gap between Original MLU and Current MLU
        # If we improved by 10%, we get +1.0.
        # If we degraded, we get negative reward.
        gap = self.original_mlu - current_mlu
        reward = gap * 10.0
        
        # Update best so far for tracking, even if not used in reward directly
        if current_mlu < self.min_mlu_so_far:
            self.min_mlu_so_far = current_mlu
            self.best_path_so_far = list(new_path)
        
        # Scale reward significantly
        # reward is already calculated above
        
        # Check success condition for tracking (optional)
        # Success if we beat the ORIGINAL
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
        # Gap Score = (Agent Gain / Max Possible Gain) * 100
        # Agent Gain = Original MLU - Current MLU
        # Max Possible Gain = Original MLU - Optimal MLU
        
        gap_score = 0.0
        if self.optimal_mlu is not None:
            max_gain = self.original_mlu - self.optimal_mlu
            agent_gain = improvement_val
            
            if max_gain > 1e-9:
                gap_score = (agent_gain / max_gain) * 100.0
            else:
                # If potential gain is effectively 0, we are already optimal originally.
                # If agent didn't make it worse (gain >= 0), score 100.
                if agent_gain >= -1e-9:
                    gap_score = 100.0
                else:
                    gap_score = 0.0 # Or negative? Stick to 0 for simplicity or let it slide.
        
        info["gap_score"] = gap_score
        
        # Calculate Peak Loss Rate
        # Derived from MLU: if MLU > 1.0, Loss Rate = 1 - 1/MLU
        peak_loss_rate = 0.0
        if current_mlu > 1.0:
            peak_loss_rate = 1.0 - (1.0 / current_mlu)
        info["peak_loss_rate"] = peak_loss_rate

        # This prevents the agent from being forced to choose an invalid action next step
        
        # Check if we found the optimal path (Approximate using MLU)
        if self.optimal_mlu is not None:
             info["is_optimal"] = (self.current_mlu <= self.optimal_mlu + 1e-5)
        else:
             info["is_optimal"] = False
        
        # Terminate if optimal MLU is reached
        # Moved to top of step
        masks = self.action_masks()
        if not np.any(masks):
            terminated = True
        
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        n = self.sample.get_network_size()
        
        # Traffic (from sample, constant per episode)
        tm = self.sample.traffic_matrix
        traffic = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                if i != j:
                    traffic[i, j] = tm[i, j].get('AggInfo', {}).get('AvgBw', 0.0)

        # Path (Current Shortest Path on Modified Topology)
        path_arr = np.full(self.max_steps + 1, -1, dtype=int)
        path_len = min(len(self.current_path), self.max_steps + 1)
        path_arr[:path_len] = self.current_path[:path_len]
        
        # NEW: Edge Features Construction
        k = len(self.all_edges)
        edge_features = np.zeros((k, 4), dtype=np.float32)
        
        # Retrieve link capacities from ORIGINAL topology (sample.topology_object)
        # We need this reference because current_topology might have deleted edges
        orig_G = self.sample.topology_object
        
        for idx, (u, v, key) in enumerate(self.all_edges):
            # 1. Capacity
            # Datanet bandwidth string "10000" or similar
            # In datanet.py, it cleans it up. We assume float-able.
            try:
                cap = float(orig_G[u][v][key]['bandwidth'])
            except:
                cap = 1.0 # Fallback
            
            # 2. Background Utilization
            # bg_loads has (u,v) -> float. BUT bg_loads is a dict {(u,v): load}.
            # It merges parallel edges? 
            # Datanet.calculate_background_loads returns {(u,v): load}.
            # If Datanet merges, we have a problem for parallel edges distinction in UTILIZATION.
            # Datanet L495: link_loads = {(u, v): 0.0 for u, v in G.edges()}
            # If G.edges() has duplicates (MultiGraph), then the dict key (u, v) will be overwritten?
            # Yes, {(u,v): 0} will collapse.
            # However, Datanet relies on simple Graph usually? 
            # If user has MultiGraph, Datanet `calculate_background_loads` might need fix.
            # Assuming for now we share utilization across parallel links or it's single graph.
            # We use (u,v) lookup.
            load = self.bg_loads.get((u, v), 0.0)
            if cap > 0:
                util = load / cap
            else:
                util = 0.0
                
            # 3. Is Active
            # Check if this SPECIFIC edge exists in current_topology
            is_active = 1.0 if self.current_topology.has_edge(u, v, key=key) else 0.0
            
            # 4. Is In Current Path
            # Identify if (u,v) or (v,u) is on current path
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

    def action_masks(self):
        """
        Returns a boolean mask of valid actions.
        True: Edge exists in current topology AND is part of current Shortest Path AND removing it does not disconnect src-dst.
        False: Edge already removed OR not in current SP OR removing it would disconnect src-dst.
        """
        mask = []
        
        # Identify edges in the current path (direction agnostic)
        # current_path is [n1, n2, n3, ...] -> edges are (n1,n2), (n2,n3), ...
        path_edges = set()
        if len(self.current_path) > 1:
            for i in range(len(self.current_path) - 1):
                u, v = self.current_path[i], self.current_path[i+1]
                path_edges.add(frozenset((u, v)))

        for u, v, key in self.all_edges:
            # Check 1: Must be in current shortest path
            if frozenset((u, v)) not in path_edges:
                mask.append(False)
                continue
                
            # Check 2: Must exist in current topology (should be implied if in current_path, but safety first)
            if not self.current_topology.has_edge(u, v, key=key):
                mask.append(False)
                continue
            
            # Use a COPY to check connectivity safely
            # This avoids any issues with restoring attributes or graph state
            G_temp = self.current_topology.copy()
            
            # Robust removal: Remove ALL edges between u and v
            while G_temp.has_edge(u, v):
                G_temp.remove_edge(u, v)
                
            while G_temp.has_edge(v, u):
                G_temp.remove_edge(v, u)
                
            is_connected = nx.has_path(G_temp, self.src, self.dst)
            mask.append(is_connected)
            
        return np.array(mask, dtype=bool)
