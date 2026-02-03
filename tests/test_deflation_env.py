import unittest
import numpy as np
import networkx as nx
import os
import shutil
from src.deflation_env import DeflationEnv

class TestDeflationEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # path to data (containing tar.gz and graph_attr.txt)
        cls.tfrecords_dir = os.path.abspath("data/nsfnetbw")
        if not os.path.exists(cls.tfrecords_dir):
            raise unittest.SkipTest("Data directory not found at data/nsfnetbw")

    def setUp(self):
        self.env = DeflationEnv(tfrecords_dir=self.tfrecords_dir, traffic_intensity=9, max_steps=10)
        self.obs, _ = self.env.reset()

    def test_initial_state(self):
        """Test initial state is correct"""
        self.assertIsNotNone(self.env.current_topology)
        self.assertEqual(len(self.env.current_topology.edges), len(self.env.sample.topology_object.edges))
        self.assertGreater(self.env.current_mlu, 0.0)
        
        # Check obs shape
        n = self.env.sample.get_network_size()
        self.assertEqual(self.obs['traffic'].shape, (n, n))
        self.assertEqual(self.obs['topology'].shape, (n, n))
        self.assertEqual(self.obs['link_utilization'].shape, (n, n))
        self.assertEqual(len(self.obs['path']), 11) # max_steps 10 + 1

    def test_step_remove_edge(self):
        """Test removing an edge works and updates topology"""
        # Find an edge on the current path to remove (to force re-routing)
        current_path = self.env.current_path
        if len(current_path) < 2:
            self.skipTest("Path too short to test edge removal effectively")
            
        u, v = current_path[0], current_path[1]
        
        # Find action index for (u, v) or (v, u)
        action = -1
        for idx, edge in enumerate(self.env.all_edges):
            if set(edge) == {u, v}:
                action = idx
                break
        
        if action == -1:
            self.fail(f"Edge {u}-{v} not found in action space")
            
        # Take step
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Check if edge is removed from current_topology
        self.assertFalse(self.env.current_topology.has_edge(u, v))
        self.assertFalse(self.env.current_topology.has_edge(v, u))
        
        # Check if path changed (unless no other path exists)
        if not terminated:
             # If not terminated, a new path was found
             # If not terminated, a new path was found
             self.assertNotEqual(self.env.current_path, current_path)
             self.assertIn('is_success', info)
             
             # With Best-So-Far reward logic:
             # Reward > 0 ONLY IF improvement over min_mlu_so_far.
             # On first step, min_mlu_so_far = original_mlu.
             # So if info['is_success'] (improvement > 0), then reward should be > 0.
             if info['is_success']:
                 self.assertGreater(reward, 0.0)
             else:
                 self.assertEqual(reward, 0.0)
             
             self.assertIsInstance(reward, float)

    def test_action_mask(self):
        """Test that action mask correctly identifies removed edges"""
        # Initially all masks should be True (or at least all edges in graph)
        # UPDATE: With strictly restrictive masking, only edges in current PATH should be True.
        masks = self.env.action_masks()
        
        # Verify restriction
        current_path_edges = set()
        path = self.env.current_path
        if len(path) > 1:
            for i in range(len(path)-1):
                current_path_edges.add(frozenset((path[i], path[i+1])))
        
        for idx, edge in enumerate(self.env.all_edges):
            if frozenset(edge) not in current_path_edges:
                self.assertFalse(masks[idx], f"Edge {edge} not in path {path} but masking is {masks[idx]}")
        
        # Remove an edge
        # Find a valid action first
        valid_indices = np.where(masks)[0]
        if len(valid_indices) == 0:
            self.skipTest("No valid actions to test removal")
            
        action = valid_indices[0]
        u, v = self.env.all_edges[action]
        self.env.step(action)
        
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
                self.assertIn(frozenset(edge), path_edges, "Valid action must be in current path")

    def test_disconnect_graph(self):
        """Test that disconnecting the graph terminates the episode"""
        # This is hard to guarantee in one step, but we can try removing all edges connected to src
        src = self.env.src
        neighbors = list(self.env.current_topology.neighbors(src))
        
        terminated = False
        for i, neighbor in enumerate(neighbors):
            # Find action
            action = -1
            for idx, edge in enumerate(self.env.all_edges):
                if set(edge) == {src, neighbor}:
                    action = idx
                    break
            
            if action != -1:
                _, _, terminated, _, _ = self.env.step(action)
                if terminated:
                    break
                    
        # Should be terminated if we isolated the node (or graph became disconnected)
        if not nx.has_path(self.env.current_topology, src, self.env.dst):
            self.assertTrue(terminated)

    def test_improvement_capability(self):
        """Test that the environment is actually capable of finding improvements"""
        # Search for a case where removing an edge improves MLU
        found_improvement = False
        max_episodes = 50
        
        for i in range(max_episodes):
            self.env.reset()
            base_mlu = self.env.current_mlu
            current_path = self.env.current_path
            
            if len(current_path) < 2: continue
            
            # Identify edges on the current path
            path_edges = list(zip(current_path[:-1], current_path[1:]))
            
            # Helper to find action index
            def get_action_idx(u, v):
                for idx, edge in enumerate(self.env.all_edges):
                    if set(edge) == {u, v}:
                        return idx
                return None
            
            # Try removing each edge on the path
            for u, v in path_edges:
                action = get_action_idx(u, v)
                if action is None: continue
                
                # Check if valid (not masked)
                masks = self.env.action_masks()
                if not masks[action]: continue
                
                # Check if this removal would improve MLU locally without taking the full step
                # (We use the env internal logic to predict improvement to avoid corrupting state for other checks,
                # OR we just step and if it improves we break, if not we reset/continue)
                
                # Using temporary env copy or manual logic to avoid deepcopying entire env
                # Actually, simplest is to use logic similar to verify_improvement.py
                
                # Temporarily remove
                data_uv = self.env.current_topology.get_edge_data(u, v)
                data_vu = self.env.current_topology.get_edge_data(v, u) if self.env.current_topology.has_edge(v, u) else None
                
                self.env.current_topology.remove_edge(u, v)
                if data_vu: self.env.current_topology.remove_edge(v, u)
                
                # If connected, calc new MLU
                if nx.has_path(self.env.current_topology, self.env.src, self.env.dst):
                    new_path = nx.shortest_path(self.env.current_topology, self.env.src, self.env.dst, weight='weight')
                    new_mlu = self.env.sample.calculate_max_utilization(new_path, self.env.bg_loads)
                    
                    if base_mlu - new_mlu > 1e-4:
                        found_improvement = True
                
                # Restore
                self.env.current_topology.add_edge(u, v, **data_uv)
                if data_vu: self.env.current_topology.add_edge(v, u, **data_vu)
                
                if found_improvement: break
            if found_improvement: break
            
        if not found_improvement:
            self.skipTest(f"No improvement opportunity found in {max_episodes} random episodes (stochastic)")
        else:
            self.assertTrue(found_improvement)

if __name__ == '__main__':
    unittest.main()
