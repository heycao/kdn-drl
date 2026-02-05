import unittest
import numpy as np
import networkx as nx
import os
from src.env import DeflectionEnv

class TestDeflectionEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # path to data
        cls.tfrecords_dir = os.path.abspath("data/nsfnetbw")
        if not os.path.exists(cls.tfrecords_dir):
            raise unittest.SkipTest("Data directory not found at data/nsfnetbw")

    def setUp(self):
        self.env = DeflectionEnv(tfrecords_dir=self.tfrecords_dir, traffic_intensity=9, max_steps=10)
        self.obs, _ = self.env.reset()

    def test_initial_state(self):
        """Test initial state and observation structure"""
        self.assertIsNotNone(self.env.current_topology)
        n = self.env.sample.get_network_size()
        
        self.assertEqual(self.obs['traffic_demand'].shape, (n, n), "Traffic demand shape wrong")
        self.assertIn('edge_features', self.obs)
        self.assertIn('edge_endpoints', self.obs)
        
        # Check defaults
        # use .n for Discrete space size
        self.assertEqual(self.env.action_space.n, len(self.env.all_edges))

    def test_step_remove_edge(self):
        """Test that base environment allows removing any edge (no masking by default)"""
        # Pick an arbitrary edge to remove
        action = 0 
        u, v, key = self.env.all_edges[action]
        
        # Verify edge exists before
        self.assertTrue(self.env.current_topology.has_edge(u, v, key=key), "Edge should exist before removal")
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if terminated and info.get('reached_optimal_mlu'):
             self.skipTest("Initialized in optimal state, cannot test edge removal.")
        
        # Check for invalid action penalty
        if info.get("invalid_action"):
            self.fail(f"Step returned invalid_action for existing edge {u}-{v} key={key}")

        # Check edge removal
        self.assertFalse(self.env.current_topology.has_edge(u, v, key=key), "Edge should be removed from current_topology")
        
        # Check reward structure
        self.assertIsInstance(reward, float)
        self.assertIn("mlu", info)


if __name__ == '__main__':
    unittest.main()
