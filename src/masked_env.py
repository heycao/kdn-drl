import numpy as np
from src.env import AdaptivePathEnv


class MaskedAdaptivePathEnv(AdaptivePathEnv):
    """
    Extended KDN-based Routing Environment with Action Masking.
    
    Provides action_masks() method for sb3-contrib's MaskablePPO.
    Only actions (edges) that start from the current node are valid.
    """
    
    def action_masks(self):
        """
        Return a boolean mask indicating which actions are valid.
        
        An action is valid if:
        1. The edge starts from the current node
        2. The edge does NOT lead to an already visited node (prevents loops)
        
        Returns:
            np.ndarray: Boolean array of shape (num_edges,) where True indicates
                       the action (edge) is valid from the current node.
        """
        mask = np.zeros(self.num_edges, dtype=bool)
        visited = set(self.current_path)
        
        for i, (u, v, key) in enumerate(self.edges):
            # Edge must start from current node AND not lead to visited node
            if u == self.current_node and v not in visited:
                mask[i] = True
        
        return mask
