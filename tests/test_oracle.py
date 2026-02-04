
import pytest
import os
import sys
import numpy as np
import networkx as nx

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datanet import Datanet

TFRECORDS_DIR = 'data/geant2bw'
TRAFFIC_INTENSITY = 15

@pytest.mark.skipif(not os.path.exists(TFRECORDS_DIR), reason="Dataset not found")
class TestOracle:
    """
    Verifies that the Oracle (Optimal Path) logic produces paths that are consistently
    better than or equal to the Baseline (Shortest Path) in terms of MLU.
    """
    
    @pytest.fixture(scope="class")
    def reader(self):
        # Initialize reader once for the class
        return Datanet(TFRECORDS_DIR, intensity_values=[TRAFFIC_INTENSITY])

    @pytest.fixture(scope="class")
    def samples(self, reader):
        # Load a small batch of samples (e.g., 5) for testing
        loaded_samples = []
        for i, sample in enumerate(reader):
            if i >= 5: 
                break
            loaded_samples.append(sample)
        return loaded_samples

    def test_oracle_vs_shortest_path(self, samples):
        """
        Check that Optimal MLU <= Shortest Path MLU for a set of random pairs.
        """
        total_checks = 0
        valid_comparisons = 0
        
        for sample in samples:
            n = sample.get_network_size()
            
            # Check a subset of pairs to keep test fast
            # We deterministically pick some pairs
            pairs_to_check = [
                (0, n-1), (1, n-2), (int(n/2), int(n/2)+1), 
                (0, int(n/2)), (int(n/2), n-1)
            ]
            
            for src, dst in pairs_to_check:
                if src == dst or src >= n or dst >= n:
                    continue
                
                # 1. Background Loads
                bg_loads = sample.calculate_background_loads(src, dst)
                
                # 2. Shortest Path (Baseline)
                try:
                    sp_path = nx.shortest_path(sample.topology_object, src, dst, weight='weight')
                except nx.NetworkXNoPath:
                    continue
                    
                sp_mlu = sample.calculate_max_utilization(sp_path, bg_loads)
                
                # 3. Optimal Path (Oracle)
                opt_path = sample.search_optimal_path(src, dst, bg_loads, max_steps=100)
                
                if not opt_path:
                    # If oracle fails to find path, something is wrong, unless disconnected
                    # But SP found one, so it must be connected.
                    pytest.fail(f"Oracle failed to find path for {src}->{dst} while SP found one.")
                    
                opt_mlu = sample.calculate_max_utilization(opt_path, bg_loads)
                
                # Assertion: Optimal MLU must be <= SP MLU (with small float tolerance)
                # Note: opt_mlu might be slightly higher due to floating point if they are effectively equal,
                # so we use a tolerance epsilon.
                epsilon = 1e-5
                
                assert opt_mlu <= sp_mlu + epsilon, \
                    f"Oracle MLU ({opt_mlu}) > SP MLU ({sp_mlu}) for sample {sample}, pair {src}->{dst}"
                
                valid_comparisons += 1
                total_checks += 1
        
        assert valid_comparisons > 0, "No valid comparisons were made (dataset might be empty or disconnected)"
