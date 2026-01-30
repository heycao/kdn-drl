"""Knowledge Defined Networking (KDN) utilities."""

import networkx as nx
import re
from typing import Dict, List, Tuple


class KDN:
    """Knowledge Defined Networking simulator and utilities."""
    
    def __init__(self, graph_file_path: str):
        """
        Loads topology and capacities from graph file.
        
        Args:
            graph_file_path: Path to graph_attr.txt file
        """
        self.G = nx.MultiDiGraph()
        self._load_graph(graph_file_path)
        
        # Pre-cache capacities for speed
        # Format: self.link_caps[(u, v)] = capacity_value
        self.link_caps = {}
        for u, v, key, data in self.G.edges(keys=True, data=True):
            # Store with key to handle multigraph
            self.link_caps[(u, v, key)] = data['capacity']
    
    def _load_graph(self, graph_file_path: str):
        """
        Parse the custom GML-like graph format.
        
        Format example:
            node [ id 0 label "0" ]
            edge [ source 0 target 1 bandwidth "10kbps" ]
        """
        with open(graph_file_path, 'r') as f:
            data = f.read()
        
        # Extract nodes
        node_pattern = r'node \[\s+id (\d+)'
        node_ids = re.findall(node_pattern, data)
        for node_id in node_ids:
            self.G.add_node(int(node_id))
        
        # Extract edges with bandwidth
        edge_pattern = r'edge \[(.*?)\]'
        edge_blocks = re.findall(edge_pattern, data, re.DOTALL)
        
        for block in edge_blocks:
            src_match = re.search(r'source (\d+)', block)
            dst_match = re.search(r'target (\d+)', block)
            key_match = re.search(r'key (\d+)', block)
            
            # FIX B: Handle decimal bandwidth and multiple units (kbps, Mbps, Gbps)
            bw_match = re.search(r'bandwidth "([\d.]+)(kbps|Mbps|Gbps)"', block, re.IGNORECASE)
            
            if src_match and dst_match and bw_match:
                src = int(src_match.group(1))
                dst = int(dst_match.group(1))
                key = int(key_match.group(1)) if key_match else 0
                
                # Parse bandwidth with unit conversion to kbps
                bw_value = float(bw_match.group(1))
                bw_unit = bw_match.group(2).lower()
                
                # Convert to kbps (base unit)
                if bw_unit == 'kbps':
                    capacity = bw_value
                elif bw_unit == 'mbps':
                    capacity = bw_value * 1000  # 1 Mbps = 1000 kbps
                elif bw_unit == 'gbps':
                    capacity = bw_value * 1_000_000  # 1 Gbps = 1,000,000 kbps
                else:
                    capacity = bw_value  # Default to kbps
                
                self.G.add_edge(src, dst, key=key, capacity=capacity)
    
    def calculate_mlu(self, traffic_matrix: List[float], paths: List[List[int]]) -> float:
        """
        Calculate Maximum Link Utilization (MLU).
        
        MLU is the "high water mark" of your network.
        Formula: For every link l, calculate Load_l = Traffic on l / Capacity_l
        MLU = max(Load_all_links)
        
        Args:
            traffic_matrix: List of traffic intensities for each flow
            paths: List of paths, where each path is a list of node IDs
                   Example: [[0, 1, 2], [0, 3, 4, 5]]
        
        Returns:
            Maximum Link Utilization across all links
        """
        # Initialize link loads
        link_loads = {}
        
        # Accumulate traffic on each link
        for flow_idx, (traffic, path) in enumerate(zip(traffic_matrix, paths)):
            # Iterate through consecutive node pairs in the path
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                
                # FIX A: For MultiGraph, select the HIGHEST CAPACITY link
                # (Assumes smart router uses best available link)
                if self.G.has_edge(u, v):
                    # Find the edge with maximum capacity
                    best_key = None
                    best_capacity = -1
                    
                    for key in self.G[u][v].keys():
                        edge_capacity = self.G[u][v][key]['capacity']
                        if edge_capacity > best_capacity:
                            best_capacity = edge_capacity
                            best_key = key
                    
                    link_id = (u, v, best_key)
                    
                    # Accumulate load
                    if link_id not in link_loads:
                        link_loads[link_id] = 0.0
                    link_loads[link_id] += traffic
        
        # Calculate utilization for each link
        max_utilization = 0.0
        
        for link_id, load in link_loads.items():
            capacity = self.link_caps[link_id]
            
            # Utilization = Load / Capacity
            utilization = load / capacity if capacity > 0 else float('inf')
            max_utilization = max(max_utilization, utilization)
        
        return max_utilization
