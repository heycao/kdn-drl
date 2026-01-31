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
        
        for i, (u, v, key) in enumerate(self.edges):
            if u == self.current_node:
                mask[i] = True
        
        return mask
