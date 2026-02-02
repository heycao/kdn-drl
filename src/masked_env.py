import numpy as np
from src.env import KDNEnvinronment


class MaskedKDNEnv(KDNEnvinronment):
    """
    Extended KDN-based Routing Environment with Action Masking.
    
    Provides action_masks() method for sb3-contrib's MaskablePPO.
    Only actions (edges) that start from the current node are valid.
    """
    
    def action_masks(self):
        """
        Return a boolean mask indicating which actions are valid.
        
        An action is valid if the edge starts from the current node.
        
        Returns:
            np.ndarray: Boolean array of shape (num_edges,) where True indicates
                       the action (edge) is valid from the current node.
        """
        mask = np.zeros(len(self.edges), dtype=bool)
        # DEBUG: Check if mask is being called
        # DEBUG: Check if mask is being called
        # print(f"DEBUG: action_masks called for node {self.current_node}.")
        # print(f"DEBUG: Mask has {np.sum(mask)} valid actions. Indices: {np.where(mask)[0]}")
        
        # 1. Identify all neighbors (edges starting from current_node)
        # 2. Filter out neighbors that are already in self.path
        
        has_valid_action = False
        neighbor_indices = []
        
        for i, (u, v, key) in enumerate(self.edges):
            if u == self.current_node:
                neighbor_indices.append(i)
                if v not in self.path:
                    mask[i] = True
                    has_valid_action = True
        
        # If no valid action found (dead end due to visited neighbors),
        # allow all neighbors (which will trigger cycle termination).
        if not has_valid_action and len(neighbor_indices) > 0:
            mask[neighbor_indices] = True
            
        # DEBUG: Check if mask is being called
        # print(f"DEBUG: action_masks called for node {self.current_node}.")
        # print(f"DEBUG: Mask has {np.sum(mask)} valid actions. Indices: {np.where(mask)[0]}")
        

                
        return mask
