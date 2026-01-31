import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.datanet import Datanet


class KDNEnvinronment(gym.Env):
    """
    Routing environment using Datanet samples as data proxy.
    Action Space: Discrete(num_edges)
    """
    def __init__(self, tfrecords_dir=None, traffic_intensity=9):
        super().__init__()
        
        if not tfrecords_dir:
            raise ValueError("tfrecords_dir is required")
        
        # Initialize Datanet reader
        self.reader = Datanet(tfrecords_dir, intensity_values=[traffic_intensity])
        self.iterator = iter(self.reader)
        
        # Load first sample to define spaces and topology
        self.sample = next(self.iterator)
        
        # Define Action Space
        # We need mapping from action_index -> (u, v)
        self.edges = [(u, v, 0) for u, v in self.sample.topology_object.edges()]
        self.action_space = spaces.Discrete(len(self.edges))
        
        # Define Observation Space
        n = self.sample.get_network_size()
        self.observation_space = spaces.Dict({
            "current_node": spaces.Box(low=0, high=n, shape=(1,), dtype=int),
            "destination": spaces.Box(low=0, high=n, shape=(1,), dtype=int),
            "traffic": spaces.Box(low=0, high=np.inf, shape=(n * (n - 1),), dtype=np.float32),
        })
        
        # Restart iterator to include first sample
        self.iterator = iter(self.reader)
        
        # Episode state
        self.current_node = 0
        self.destination = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
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
        
        self.current_node = src
        self.destination = dst
        
        return self._get_obs(), {}

    def step(self, action):
        """Takes an action (edge index) and moves the agent if valid."""
        raise NotImplementedError("step() needs to be implemented")

    def _get_obs(self):
        tm = self.sample.traffic_matrix
        n = self.sample.get_network_size()
        traffic = np.array([
            tm[i, j]['AggInfo']['AvgBw'] 
            for i in range(n) for j in range(n) if i != j
        ], dtype=np.float32)
        
        return {
            "current_node": np.array([self.current_node], dtype=int),
            "destination": np.array([self.destination], dtype=int),
            "traffic": traffic,
        }
