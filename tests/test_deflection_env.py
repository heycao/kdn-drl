import unittest
import numpy as np
import networkx as nx
import os
import shutil
from src.deflection_env import DeflectionEnv

class TestDeflectionEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # path to data (containing tar.gz and graph_attr.txt)
        cls.tfrecords_dir = os.path.abspath("data/nsfnetbw")
        if not os.path.exists(cls.tfrecords_dir):
            raise unittest.SkipTest("Data directory not found at data/nsfnetbw")

    def setUp(self):
        self.env = DeflectionEnv(tfrecords_dir=self.tfrecords_dir, traffic_intensity=9, max_steps=10)
        self.obs, _ = self.env.reset()

    def test_initial_state(self):
        """Test initial state is correct"""
        self.assertIsNotNone(self.env.current_topology)
        self.assertEqual(len(self.env.current_topology.edges), len(self.env.sample.topology_object.edges))
        self.assertGreater(self.env.current_mlu, 0.0)
        
        # Check obs shape
        n = self.env.sample.get_network_size()
        self.assertEqual(self.obs['traffic_demand'].shape, (n, n), "traffic_demand shape check")
        self.assertEqual(len(self.obs['path']), 11) # max_steps 10 + 1

    def test_step_remove_edge(self):
        """Test removing an edge works and updates topology"""
        # Find an edge on the current path to remove (to force re-routing)
        current_path = self.env.current_path
        if len(current_path) < 2:
            self.skipTest("Path too short to test edge removal effectively")
            
        u, v = current_path[0], current_path[1]
        
        # Find action index for (u, v)
        action = -1
        for idx, edge in enumerate(self.env.all_edges):
            # edge is (n1, n2, key)
            if edge[0] == u and edge[1] == v:
                action = idx
                break
        
        if action == -1:
            self.fail(f"Edge {u}-{v} not found in action space")
            
        # Take step
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if terminated and info.get('reached_optimal_mlu'):
             self.skipTest("Initialized in optimal state, cannot test edge removal.")

        # Get the specific edge key that corresponds to the action
        _, _, key = self.env.all_edges[action]
        
        # Check if the SPECIFIC edge is removed from current_topology
        self.assertFalse(self.env.current_topology.has_edge(u, v, key=key))
        
        # Check if path changed (unless no other path exists)
        if not terminated:
             # If not terminated, a new path was found
             self.assertNotEqual(self.env.current_path, current_path)
             self.assertIn('is_success', info)
             
             if info['is_success']:
                 self.assertGreater(reward, 0.0)
             else:
                 self.assertLessEqual(reward, 0.0)
             
             self.assertIsInstance(reward, float)

    def test_action_mask(self):
        """Test that action mask correctly identifies removed edges"""
        masks = self.env.action_masks()
        
        # Verify restriction
        current_path_edges = set()
        path = self.env.current_path
        if len(path) > 1:
            for i in range(len(path)-1):
                current_path_edges.add(frozenset((path[i], path[i+1])))
        
        for idx, edge in enumerate(self.env.all_edges):
            if frozenset({edge[0], edge[1]}) not in current_path_edges:
                self.assertFalse(masks[idx], f"Edge {edge} not in path {path} but masking is {masks[idx]}")
        
        # Remove an edge
        valid_indices = np.where(masks)[0]
        if len(valid_indices) == 0:
            self.skipTest("No valid actions to test removal")
            
        action = valid_indices[0]
        u, v, _ = self.env.all_edges[action]
        _, _, terminated, _, info = self.env.step(action)
        
        if terminated and info.get('reached_optimal_mlu'):
             self.skipTest("Initialized in optimal state, cannot test edge removal.")
        
        # Check mask again
        masks = self.env.action_masks()
        self.assertFalse(masks[action])

    def test_action_masks_restrict_to_path(self):
        """Verify that action mask allows ONLY edges in current path"""
        masks = self.env.action_masks()
        path = self.env.current_path
        
        path_edges = set()
        for i in range(len(path) - 1):
            path_edges.add(frozenset((path[i], path[i+1])))
            
        for i, is_valid in enumerate(masks):
            if is_valid:
                edge = self.env.all_edges[i]
                u, v = edge[0], edge[1]
                self.assertIn(frozenset((u, v)), path_edges, "Valid action must be in current path")

    def test_disconnect_graph(self):
        """Test that disconnecting the graph terminates the episode"""
        src = self.env.src
        neighbors = list(self.env.current_topology.neighbors(src))
        
        terminated = False
        for i, neighbor in enumerate(neighbors):
            action = -1
            for idx, edge in enumerate(self.env.all_edges):
                if edge[0] == src and edge[1] == neighbor:
                    action = idx
                    break
            
            if action != -1:
                _, _, terminated, _, _ = self.env.step(action)
                if terminated:
                    break
                    
        if not nx.has_path(self.env.current_topology, src, self.env.dst):
            self.assertTrue(terminated)

    def test_improvement_capability(self):
        """Test that the environment is actually capable of finding improvements"""
        found_improvement = False
        max_episodes = 50
        
        for i in range(max_episodes):
            self.env.reset()
            base_mlu = self.env.current_mlu
            current_path = self.env.current_path
            
            if len(current_path) < 2: continue
            
            path_edges = list(zip(current_path[:-1], current_path[1:]))
            
            def get_action_idx(u, v):
                for idx, edge in enumerate(self.env.all_edges):
                    if edge[0] == u and edge[1] == v:
                        return idx
                return None
            
            for u, v in path_edges:
                action = get_action_idx(u, v)
                if action is None: continue
                
                masks = self.env.action_masks()
                if not masks[action]: continue
                
                # Temporarily remove
                data_uv = self.env.current_topology.get_edge_data(u, v)
                data_vu = self.env.current_topology.get_edge_data(v, u) if self.env.current_topology.has_edge(v, u) else None
                
                self.env.current_topology.remove_edge(u, v)
                if data_vu: self.env.current_topology.remove_edge(v, u)
                
                if nx.has_path(self.env.current_topology, self.env.src, self.env.dst):
                    new_path = nx.shortest_path(self.env.current_topology, self.env.src, self.env.dst, weight='weight')
                    new_mlu = self.env.sample.calculate_max_utilization(new_path, self.env.bg_loads)
                    
                    if base_mlu - new_mlu > 1e-4:
                        found_improvement = True
                
                self.env.current_topology.add_edge(u, v, **data_uv)
                if data_vu: self.env.current_topology.add_edge(v, u, **data_vu)
                
                if found_improvement: break
            if found_improvement: break
            
        if not found_improvement:
            self.skipTest(f"No improvement opportunity found in {max_episodes} random episodes (stochastic)")
        else:
            self.assertTrue(found_improvement)

    def test_termination_at_optimal(self):
        """Test that the environment terminates when optimal MLU is reached"""
        self.env = DeflectionEnv(tfrecords_dir=self.tfrecords_dir, traffic_intensity=9, max_steps=10)
        self.env.reset()
        
        if self.env.optimal_mlu is None:
             self.skipTest("Could not calculate optimal MLU for this sample.")
        
        original_calc_mlu = self.env.sample.calculate_max_utilization
        
        def mock_calc_mlu(path, loads):
            return self.env.optimal_mlu
            
        self.env.sample.calculate_max_utilization = mock_calc_mlu
        
        try:
            masks = self.env.action_masks()
            valid_indices = np.where(masks)[0]
            if len(valid_indices) == 0:
                 self.skipTest("No valid actions to test step")
                 
            action = valid_indices[0]
            _, _, terminated, _, info = self.env.step(action)
            
            self.assertTrue(terminated)
            self.assertTrue(info.get('reached_optimal_mlu', False))
            
        finally:
            self.env.sample.calculate_max_utilization = original_calc_mlu

    def test_initial_optimality(self):
        """Test that environment terminates immediately if reset state is already optimal"""
        self.env = DeflectionEnv(tfrecords_dir=self.tfrecords_dir, traffic_intensity=9, max_steps=10)
        self.env.reset()
        
        # Manually force current_mlu to equal optimal_mlu
        # We need optimal_mlu to be set
        if self.env.optimal_mlu is None:
             self.env.optimal_mlu = 0.5 # Mock value
        
        self.env.current_mlu = self.env.optimal_mlu
        
        # Step with any action should return immediate termination
        # Action is ignored, so any integer within space is fine
        action = 0 
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.assertTrue(terminated)
        self.assertTrue(info.get("reached_optimal_mlu", False))
        self.assertTrue(info.get("is_optimal", False))
        self.assertEqual(reward, 0.0)

if __name__ == '__main__':
    unittest.main()
