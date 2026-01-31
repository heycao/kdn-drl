import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
import tensorflow as tf
import glob
from src.datanet import Datanet

class AdaptivePathEnv(gym.Env):
    """
    KDN-based Routing Environment.
    Action Space: Discrete(num_edges)
    Loads traffic scenarios from TFRecords.
    Reward based on Maximum Link Utilization (MLU) changes.
    """
    def __init__(self, tfrecords_dir=None, traffic_intensity=9, alpha=10.0, max_steps=50):
        super().__init__()
        
        # Topology will be loaded from Datanet samples
        self.graph = None
        self.num_nodes = 0
        self.edges = []
        self.num_edges = 0
        self.link_caps = {}  # Cache for link capacities

        # Load TFRecords - required for this environment
        self.tfrecords_dir = tfrecords_dir
        self.traffic_intensity = traffic_intensity
        self.reader = None
        self.iterator = None
        
        if tfrecords_dir:
            self._init_datanet(tfrecords_dir, traffic_intensity)
        else:
            raise ValueError("tfrecords_dir is required for AdaptivePathEnv")
        
        # Action: Index of the edge to use
        self.action_space = spaces.Discrete(self.num_edges)

        # Observation: current node, destination, and full traffic matrix
        # 182 = 14 * 13 (14 nodes, off-diagonal traffic matrix)
        self.observation_space = spaces.Dict({
            "current_node": spaces.Box(low=0, high=self.num_nodes, shape=(1,), dtype=int),
            "destination": spaces.Box(low=0, high=self.num_nodes, shape=(1,), dtype=int),
            "traffic": spaces.Box(low=0, high=np.inf, shape=(self.num_nodes * (self.num_nodes - 1),), dtype=np.float32),
        })
        
        # Simulation state
        self.source = 0  # Original source node (fixed during episode)
        self.current_node = 0
        self.destination = 0
        self.current_traffic = 0.0
        self.traffic_matrix = np.zeros(182, dtype=np.float32)
        self.current_path = []  # Track path during episode
        self.current_mlu = 0.0  # Track running Maximum Link Utilization (aggregate)
        self.current_step = 0  # Track step count
        self.baseline_mlu = 0.0  # Baseline MLU from Shortest Path (aggregate)
        self.current_sample = None  # Current Datanet sample for MLU calculation
        self.current_routing = None  # Current routing matrix (modified during episode)
        
        # Episode parameters
        self.alpha = alpha
        self.max_steps = max_steps

    def _init_datanet(self, data_dir, traffic_intensity):
        """Initialize datanetAPI reader and load topology from first sample."""
        # datanetAPI filters by intensity. We use [traffic_intensity] for a single value.
        self.reader = Datanet(data_dir, intensity_values=[traffic_intensity])
        self.iterator = iter(self.reader)
        
        # Load first sample to get topology
        first_sample = next(self.iterator)
        self._init_topology_from_sample(first_sample)
        
        # Restart iterator to include first sample
        self.iterator = iter(self.reader)
    
    def _init_topology_from_sample(self, sample):
        """Initialize topology from a Datanet sample."""
        self.graph = sample.get_topology_object()
        self.num_nodes = self.graph.number_of_nodes()
        
        # Mapping for edge-based actions
        # Each index corresponds to an edge (u, v)
        self.edges = [(u, v, 0) for u, v in self.graph.edges()]  # Add key=0 for consistency
        self.num_edges = len(self.edges)
        
        # Cache link capacities
        # datanetAPI stores bandwidth as string e.g. "10000" (in bps after processing)
        self.link_caps = {}
        for u, v in self.graph.edges():
            bw = float(self.graph[u][v][0]['bandwidth'])
            self.link_caps[(u, v, 0)] = bw

    def _sample_traffic(self):
        """Sample a traffic matrix and baseline paths from datanetAPI."""
        try:
            sample = next(self.iterator)
        except StopIteration:
            # Restart iterator if we reach the end
            self.iterator = iter(self.reader)
            sample = next(self.iterator)
        
        # Store sample for aggregate MLU calculation
        self.current_sample = sample
        
        # datanetAPI returns 2D matrices (NxN) of dictionaries
        traffic_matrix = sample.get_traffic_matrix()
        routing_matrix = sample.get_routing_matrix()
        
        # Flatten traffic matrix for observation space (182 = 14 * 13)
        # We only take off-diagonal elements (src != dst)
        flat_traffic = []
        n = self.num_nodes
        for i in range(n):
            for j in range(n):
                if i != j:
                    # In datanetAPI, traffic is typically extracted from 'AggInfo'['AvgBw'] or similar
                    # For this environment, we use 'AvgBw' as the intensity measure
                    traffic_val = traffic_matrix[i, j]['AggInfo']['AvgBw']
                    flat_traffic.append(traffic_val)
        
        return np.array(flat_traffic, dtype=np.float32), routing_matrix

    def _flat_to_pair(self, flat_idx):
        """Convert flat traffic matrix index to (src, dst) pair.
        
        For N nodes, traffic matrix has N*(N-1) entries (no self-loops).
        Example for N=14: indices 0-12 are src=0, dst=1..13
                          indices 13-25 are src=1, dst=0,2..13, etc.
        """
        n = self.num_nodes
        src = flat_idx // (n - 1)
        dst_offset = flat_idx % (n - 1)
        dst = dst_offset if dst_offset < src else dst_offset + 1
        return src, dst

    def _calculate_aggregate_mlu(self, path_for_current_flow=None):
        """Calculate aggregate MLU using sample.calculate_mlu().
        
        If path_for_current_flow is provided, uses modified routing for (source, destination).
        Otherwise uses the baseline routing matrix.
        """
        if self.current_sample is None:
            return 0.0
        
        if path_for_current_flow is not None:
            # Create modified routing matrix with agent's path for current flow
            modified_routing = self.current_routing.copy()
            # Use self.source (original source) not self.current_node (current position)
            modified_routing[self.source, self.destination] = path_for_current_flow
            return self.current_sample.calculate_mlu(routing_matrix=modified_routing)
        else:
            return self.current_sample.calculate_mlu()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.iterator:
            # 1. Get Traffic AND Baseline Routing Matrix
            traffic_matrix_flat, routing_matrix = self._sample_traffic()
            
            # 2. Select a random source-destination pair
            flat_idx = self.np_random.integers(0, len(traffic_matrix_flat))
            src, dst = self._flat_to_pair(flat_idx)
            
            self.source = src  # Track original source (fixed during episode)
            self.current_node = src
            self.destination = dst
            self.current_traffic = traffic_matrix_flat[flat_idx]
            self.traffic_matrix = traffic_matrix_flat
            self.current_routing = routing_matrix.copy()  # Store for modification during episode
            
            # 3. Calculate Baseline MLU using AGGREGATE method (all flows)
            self.baseline_mlu = self._calculate_aggregate_mlu()
            
            # Initialize State
            self.current_path = [self.current_node]
            self.current_mlu = self.baseline_mlu  # Start with baseline aggregate MLU
            self.current_step = 0
        else:
            raise ValueError(
                "Datanet reader not initialized. "
                "Please initialize the environment with tfrecords_dir parameter (pointing to dataset root)"
            )
        
        return self._get_obs(), {}

    def _get_link_utilization(self, u, v, key):
        """Get utilization of a specific link for current traffic."""
        capacity = self.link_caps.get((u, v, key), 0.0)
        return self.current_traffic / capacity if capacity > 0 else 1.0

    def step(self, action):
        """
        Takes an action (edge index) and moves the agent if valid.
        
        Reward based on aggregate MLU change using sample.calculate_mlu().
        """
        # action is an index into self.edges
        u, v, key = self.edges[action]
        
        # Increment step counter
        self.current_step += 1
        
        # Track MLU before action
        mlu_before = self.current_mlu
        
        # Reward components
        step_penalty = 0.0 # Penalty for every step taken
        invalid_penalty = -1.0  # Strong penalty for selecting invalid edge
        terminated = False
        mlu_reward = 0.0
        
        # Check if the selected edge is valid (starts from current node)
        if u == self.current_node:
            # Valid action: move to the next node
            self.current_node = v
            self.current_path.append(v)
            
            # Recalculate AGGREGATE MLU with agent's current path
            # This replaces the old single-flow incremental update
            self.current_mlu = self._calculate_aggregate_mlu(path_for_current_flow=self.current_path)
            
            # Congestion-based component: delta in aggregate MLU
            mlu_reward = -self.alpha * (self.current_mlu - mlu_before)
            
            # Total reward for valid action: step_penalty + mlu_reward
            reward = step_penalty + mlu_reward
            
            # Check if reached destination
            if self.current_node == self.destination:
                terminated = True
                
                # --- INTELLIGENCE BONUS (Beat the Baseline) ---
                # Reward = (Baseline_MLU - Agent_MLU) based on aggregate MLU
                gap = self.baseline_mlu - self.current_mlu
                bonus = 10.0 * gap  # Scale up to make it significant
                
                reward += bonus
        else:
            # Invalid action: combined step_penalty and invalid_penalty
            reward = step_penalty + invalid_penalty
        
        # Check if max steps reached
        truncated = self.current_step >= self.max_steps
        
        # Calculate gap even if not done, for monitoring
        gap = self.baseline_mlu - self.current_mlu
        
        info = {
            'mlu': self.current_mlu,
            'path_length': len(self.current_path),
            'is_success': terminated,
            'mlu_reward': mlu_reward,
            'gap': gap if terminated else 0.0
        }
        
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return {
            "current_node": np.array([self.current_node], dtype=int),
            "destination": np.array([self.destination], dtype=int),
            "traffic": self.traffic_matrix.copy(),
        }
