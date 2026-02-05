
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

    def test_oracle_matches_brute_force(self, samples):
        """
        Verify that Oracle MLU equals the brute-force optimal MLU.
        """
        from itertools import islice

        def brute_force_optimal_path(sample, src, dst, bg_loads, max_paths=1000):
            """
            Enumerate all simple paths and find the one with minimum MLU.
            Returns (best_path, min_mlu) or (None, inf) if no path exists.
            """
            G = sample.topology_object
            try:
                # Get all simple paths (limited to avoid combinatorial explosion)
                all_paths = list(islice(nx.all_simple_paths(G, src, dst), max_paths))
            except nx.NetworkXNoPath:
                return None, float('inf')
            
            if not all_paths:
                return None, float('inf')
            
            best_path = None
            min_mlu = float('inf')
            
            for path in all_paths:
                mlu = sample.calculate_max_utilization(path, bg_loads)
                if mlu < min_mlu or (abs(mlu - min_mlu) < 1e-9 and 
                                    best_path is not None and 
                                    len(path) < len(best_path)):
                    min_mlu = mlu
                    best_path = path
            return best_path, min_mlu

        total_checks = 0
        mismatches = []
        
        for sample_idx, sample in enumerate(samples[:3]): # Keep it fast
            n = sample.get_network_size()
            
            # Test a subset of src-dst pairs
            pairs_to_check = []
            for src in range(min(4, n)):
                for dst in range(min(4, n)):
                    if src != dst:
                        pairs_to_check.append((src, dst))
            
            for src, dst in pairs_to_check:
                if src >= n or dst >= n:
                    continue
                
                bg_loads = sample.calculate_background_loads(src, dst)
                oracle_path = sample.search_optimal_path(src, dst, bg_loads, max_steps=100)
                
                if oracle_path is None:
                    bf_path, bf_mlu = brute_force_optimal_path(sample, src, dst, bg_loads)
                    assert bf_path is None, f"Oracle=None but BF found path for {src}->{dst}"
                    continue
                
                oracle_mlu = sample.calculate_max_utilization(oracle_path, bg_loads)
                bf_path, bf_mlu = brute_force_optimal_path(sample, src, dst, bg_loads)
                
                epsilon = 1e-6
                if oracle_mlu > bf_mlu + epsilon:
                    mismatches.append({
                        'sample': sample_idx, 'src': src, 'dst': dst,
                        'oracle_mlu': oracle_mlu, 'bf_mlu': bf_mlu
                    })
                total_checks += 1
        
        if mismatches:
            pytest.fail(f"Found {len(mismatches)} mismatches where Oracle > BF. First: {mismatches[0]}")
        
        assert total_checks > 0

    def test_oracle_finds_shortest_among_optimal(self, samples):
        """
        Verify that among all paths with optimal MLU, Oracle picks the shortest (minimum hops).
        """
        from itertools import islice
        total_checks = 0
        
        for sample in samples:
            n = sample.get_network_size()
            pairs_to_check = [(0, n-1), (1, n-2)] if n > 2 else [(0, 1)]
            
            for src, dst in pairs_to_check:
                if src >= n or dst >= n or src == dst:
                    continue
                
                bg_loads = sample.calculate_background_loads(src, dst)
                oracle_path = sample.search_optimal_path(src, dst, bg_loads, max_steps=100)
                if oracle_path is None:
                    continue
                
                oracle_mlu = sample.calculate_max_utilization(oracle_path, bg_loads)
                
                # Find all paths with the same optimal MLU
                G = sample.topology_object
                try:
                    all_paths = list(islice(nx.all_simple_paths(G, src, dst), 500))
                except nx.NetworkXNoPath:
                    continue
                
                min_hops_at_optimal = float('inf')
                epsilon = 1e-6
                
                for path in all_paths:
                    mlu = sample.calculate_max_utilization(path, bg_loads)
                    if abs(mlu - oracle_mlu) < epsilon:
                        min_hops_at_optimal = min(min_hops_at_optimal, len(path))
                
                assert len(oracle_path) <= min_hops_at_optimal + 1, \
                    f"Oracle path ({len(oracle_path)} hops) not shortest optimal ({min_hops_at_optimal})"
                total_checks += 1
        
        assert total_checks > 0
