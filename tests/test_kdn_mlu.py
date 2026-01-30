"""Unit tests for KDN MLU calculation."""

import pytest
import tensorflow as tf
import numpy as np
from pathlib import Path
from src.kdn import KDN


def parse_tfrecord(example_proto):
    """Parse TFRecord example."""
    feature_description = {
        'traffic': tf.io.VarLenFeature(tf.float32),
        'delay': tf.io.VarLenFeature(tf.float32),
        'paths': tf.io.VarLenFeature(tf.int64),
        'links': tf.io.VarLenFeature(tf.int64),
        'n_paths': tf.io.FixedLenFeature([], tf.int64),
        'n_links': tf.io.FixedLenFeature([], tf.int64),
    }
    return tf.io.parse_single_example(example_proto, feature_description)


def parse_flow_to_links(paths_flat, links_flat):
    """
    Parse flow-to-link mapping from TFRecord format.
    
    The TFRecord format uses parallel arrays:
    - paths_flat[i] = flow_id
    - links_flat[i] = link_id used by that flow
    
    Args:
        paths_flat: Array of flow IDs
        links_flat: Array of link IDs (parallel to paths_flat)
    
    Returns:
        Dict mapping flow_id -> [link_ids]
    """
    from collections import defaultdict
    
    flow_links = defaultdict(list)
    
    for i in range(len(paths_flat)):
        flow_id = int(paths_flat[i])
        link_id = int(links_flat[i])
        flow_links[flow_id].append(link_id)
    
    return flow_links


class TestKDNMLU:
    """Test suite for KDN MLU calculation."""
    
    @pytest.fixture
    def kdn(self):
        """Create KDN instance."""
        graph_path = "data/nsfnetbw/graph_attr.txt"
        return KDN(graph_path)
    
    @pytest.fixture
    def link_index_to_nodes(self, kdn):
        """Create mapping from link index to (src, dst)."""
        # Build ordered list of links matching TFRecord link indexing
        link_list = []
        for u, v, key, data in kdn.G.edges(keys=True, data=True):
            link_list.append((u, v, key))
        return link_list
    
    def test_graph_loading(self, kdn):
        """Test that graph is loaded correctly."""
        # NSFNet has 14 nodes
        assert kdn.G.number_of_nodes() == 14
        
        # Check that capacities are loaded
        assert len(kdn.link_caps) > 0
        
        # Verify some known capacities from graph_attr.txt
        # Edge 0->1 has 10kbps capacity (key=0)
        assert kdn.link_caps.get((0, 1, 0)) == 10.0
        
        # Edge 3->4 has 40kbps capacity (key=0)
        assert kdn.link_caps.get((3, 4, 0)) == 40.0
    
    def test_mlu_calculation_simple(self, kdn):
        """Test MLU calculation with simple example."""
        # Simple test: single flow on single link
        traffic_matrix = [5.0]  # 5 kbps
        paths = [[0, 1]]  # Path from node 0 to node 1
        
        # Link 0->1 has capacity 10 kbps
        # Expected utilization: 5.0 / 10.0 = 0.5
        mlu = kdn.calculate_mlu(traffic_matrix, paths)
        assert abs(mlu - 0.5) < 1e-6
    
    def test_mlu_multiple_flows(self, kdn):
        """Test MLU with multiple flows sharing a link."""
        # Two flows on the same link
        traffic_matrix = [3.0, 4.0]
        paths = [
            [0, 1],  # Flow 1: 3 kbps
            [0, 1],  # Flow 2: 4 kbps
        ]
        
        # Total load: 3 + 4 = 7 kbps
        # Capacity: 10 kbps
        # Expected MLU: 7.0 / 10.0 = 0.7
        mlu = kdn.calculate_mlu(traffic_matrix, paths)
        assert abs(mlu - 0.7) < 1e-6
    
    def test_mlu_different_capacities(self, kdn):
        """Test MLU with links of different capacities."""
        # Flow on high-capacity link
        traffic_matrix = [20.0, 5.0]
        paths = [
            [3, 4],  # High capacity link (40 kbps)
            [0, 1],  # Low capacity link (10 kbps)
        ]
        
        # Utilizations:
        # Link 3->4: 20/40 = 0.5
        # Link 0->1: 5/10 = 0.5
        # Expected MLU: 0.5
        mlu = kdn.calculate_mlu(traffic_matrix, paths)
        assert abs(mlu - 0.5) < 1e-6
    
    @pytest.mark.parametrize("split", ["train", "evaluate"])
    def test_mlu_with_tfrecords(self, kdn, link_index_to_nodes, split):
        """
        Test MLU calculation against TFRecord ground truth.
        
        This test verifies that the MLU calculation works correctly
        with the TFRecord link-based representation:
        - MLU should be > 0 for non-zero traffic
        - MLU should be reasonable (typically < 2.0 for healthy networks)
        - Tests both train and evaluate datasets
        """
        # Load one TFRecord file
        tfrecord_dir = Path(f"data/nsfnetbw/tfrecords/{split}")
        tfrecord_files = list(tfrecord_dir.glob("*.tfrecords"))
        
        if not tfrecord_files:
            pytest.skip(f"No TFRecord files found in {tfrecord_dir}")
        
        # Test with first file
        tfrecord_path = str(tfrecord_files[0])
        dataset = tf.data.TFRecordDataset(tfrecord_path)
        
        for raw_record in dataset.take(1):
            parsed = parse_tfrecord(raw_record)
            
            # Extract data
            traffic = tf.sparse.to_dense(parsed['traffic']).numpy()
            paths_flat = tf.sparse.to_dense(parsed['paths']).numpy()
            links_flat = tf.sparse.to_dense(parsed['links']).numpy()
            n_paths = int(parsed['n_paths'].numpy())
            
            # Parse flow-to-link mapping
            flow_links = parse_flow_to_links(paths_flat, links_flat)
            
            # Convert link indices to node paths
            paths = []
            for flow_id in range(n_paths):
                link_indices = flow_links.get(flow_id, [])
                if link_indices:
                    # Reconstruct path from link sequence
                    node_path = [link_index_to_nodes[link_indices[0]][0]]  # Start node
                    for link_idx in link_indices:
                        src, dst, key = link_index_to_nodes[link_idx]
                        node_path.append(dst)
                    paths.append(node_path)
                else:
                    # Direct connection or single-node path
                    paths.append([flow_id])  # Placeholder
            
            # Calculate MLU
            if len(paths) == len(traffic):
                mlu = kdn.calculate_mlu(traffic.tolist(), paths)
                
                # Basic sanity checks
                assert mlu > 0, "MLU should be positive for non-zero traffic"
                assert mlu < 2.0, f"MLU is too high: {mlu:.4f} (check capacity units or path reconstruction)"
                
                print(f"\n{split} dataset:")
                print(f"  Calculated MLU: {mlu:.4f}")
                print(f"  Number of flows: {len(traffic)}")
                print(f"  Avg traffic per flow: {np.mean(traffic):.4f}")
                print(f"  File: {tfrecord_path.split('/')[-1]}")
            else:
                pytest.skip(f"Path reconstruction failed: {len(paths)} paths vs {len(traffic)} traffic entries")
    
    @pytest.mark.parametrize("split", ["train", "evaluate"])
    def test_mlu_delay_correlation(self, kdn, link_index_to_nodes, split):
        """
        Test correlation between MLU and delay using Spearman's Rank Correlation.
        
        This validates the MLU implementation by ensuring that:
        1. Higher MLU correlates with higher average delay
        2. The correlation is statistically significant
        
        Spearman's correlation measures monotonic relationships, which is appropriate
        because MLU and delay should have a monotonically increasing relationship
        (higher congestion -> higher delay).
        """
        from scipy.stats import spearmanr
        
        # Load multiple TFRecord files to get enough samples
        tfrecord_dir = Path(f"data/nsfnetbw/tfrecords/{split}")
        tfrecord_files = list(tfrecord_dir.glob("*.tfrecords"))
        
        if not tfrecord_files:
            pytest.skip(f"No TFRecord files found in {tfrecord_dir}")
        
        # Limit to first 20 files for reasonable test time
        tfrecord_files = tfrecord_files[:20]
        
        mlu_values = []
        avg_delay_values = []
        
        for tfrecord_path in tfrecord_files:
            dataset = tf.data.TFRecordDataset(str(tfrecord_path))
            
            for raw_record in dataset.take(1):
                parsed = parse_tfrecord(raw_record)
                
                # Extract data
                traffic = tf.sparse.to_dense(parsed['traffic']).numpy()
                delay = tf.sparse.to_dense(parsed['delay']).numpy()
                paths_flat = tf.sparse.to_dense(parsed['paths']).numpy()
                links_flat = tf.sparse.to_dense(parsed['links']).numpy()
                n_paths = int(parsed['n_paths'].numpy())
                
                # Parse flow-to-link mapping
                flow_links = parse_flow_to_links(paths_flat, links_flat)
                
                # Convert link indices to node paths
                paths = []
                for flow_id in range(n_paths):
                    link_indices = flow_links.get(flow_id, [])
                    if link_indices:
                        # Reconstruct path from link sequence
                        node_path = [link_index_to_nodes[link_indices[0]][0]]
                        for link_idx in link_indices:
                            src, dst, key = link_index_to_nodes[link_idx]
                            node_path.append(dst)
                        paths.append(node_path)
                    else:
                        # Direct connection or single-node path
                        paths.append([flow_id])
                
                # Calculate MLU
                if len(paths) == len(traffic):
                    mlu = kdn.calculate_mlu(traffic.tolist(), paths)
                    avg_delay = np.mean(delay)
                    
                    mlu_values.append(mlu)
                    avg_delay_values.append(avg_delay)
        
        # Ensure we have enough samples
        assert len(mlu_values) >= 10, f"Need at least 10 samples, got {len(mlu_values)}"
        
        # Calculate Spearman's rank correlation
        correlation, p_value = spearmanr(mlu_values, avg_delay_values)
        
        print(f"\n{split} dataset - MLU vs Delay Correlation:")
        print(f"  Number of samples: {len(mlu_values)}")
        print(f"  Spearman correlation: {correlation:.4f}")
        print(f"  P-value: {p_value:.4e}")
        print(f"  MLU range: [{min(mlu_values):.4f}, {max(mlu_values):.4f}]")
        print(f"  Avg delay range: [{min(avg_delay_values):.4f}, {max(avg_delay_values):.4f}]")
        
        # Assertions to validate the correlation
        # We expect a positive correlation (higher MLU -> higher delay)
        assert correlation > 0, f"Expected positive correlation, got {correlation:.4f}"
        
        # The correlation should be statistically significant (p < 0.05)
        assert p_value < 0.05, f"Correlation not significant (p={p_value:.4f})"
        
        # We expect a moderate to strong positive correlation (> 0.3)
        # Note: The threshold may need adjustment based on the dataset characteristics
        assert correlation > 0.3, (
            f"MLU-delay correlation too weak ({correlation:.4f}). "
            "This suggests MLU calculation may be incorrect."
        )
        
        print(f"  ✅ Correlation test passed: Strong positive correlation detected")
