import numpy as np
import networkx as nx
from src.env import KDNEnvinronment

class DeflationEnv(KDNEnvinronment):
    """
    Inherits from KDNEnvinronment (Deflation Logic).
    Only implements specific action masking for deflation.
    """
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
            G_temp = self.current_topology.copy()
            
            # Robust removal: Remove ALL edges between u and v
            if G_temp.has_edge(u, v):
                if G_temp.is_multigraph():
                    keys = list(G_temp[u][v].keys())
                    for k in keys:
                        G_temp.remove_edge(u, v, key=k)
                else:
                    G_temp.remove_edge(u, v)
                
            if G_temp.has_edge(v, u):
                if G_temp.is_multigraph():
                    keys = list(G_temp[v][u].keys())
                    for k in keys:
                        G_temp.remove_edge(v, u, key=k)
                else:
                    G_temp.remove_edge(v, u)
                
            is_connected = nx.has_path(G_temp, self.src, self.dst)
            mask.append(is_connected)
            
        return np.array(mask, dtype=bool)
