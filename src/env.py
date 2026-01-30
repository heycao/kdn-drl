import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import tensorflow as tf
import glob
from src.kdn import KDN

class AdaptivePathEnv(gym.Env):
    """
    KDN-based Routing Environment.
    Action Space: Discrete(num_edges)
    Loads traffic scenarios from TFRecords.
    Reward based on Maximum Link Utilization (MLU) changes.
    """
    def __init__(self, graph_path='data/nsfnetbw/graph_attr.txt', tfrecords_dir=None, traffic_intensity=9, alpha=10.0, max_steps=50):
        super().__init__()
        
        # Load topology via KDN
        self.kdn = KDN(graph_path)
        self.graph = self.kdn.G
        self.num_nodes = len(self.graph.nodes())
        
        # Mapping for edge-based actions
        # Each index corresponds to an edge (u, v, key)
        self.edges = list(self.graph.edges(keys=True))
        self.num_edges = len(self.edges)

        # Action: Index of the edge to use
        self.action_space = spaces.Discrete(self.num_edges)

        # Observation: current node, destination, and full traffic matrix
        self.observation_space = spaces.Dict({
            "current_node": spaces.Box(low=0, high=self.num_nodes, shape=(1,), dtype=int),
            "destination": spaces.Box(low=0, high=self.num_nodes, shape=(1,), dtype=int),
            "traffic": spaces.Box(low=0, high=np.inf, shape=(182,), dtype=np.float32),
        })
        
        # Load TFRecords if provided
        self.tfrecords_dir = tfrecords_dir
        self.traffic_intensity = traffic_intensity
        self.tfrecord_dataset = None
        self.tfrecord_iterator = None
        
        if tfrecords_dir:
            self._init_tfrecords(tfrecords_dir, traffic_intensity)
        
        # Simulation state
        self.current_node = 0
        self.destination = 0
        self.current_traffic = 0.0
        self.traffic_matrix = np.zeros(182, dtype=np.float32)
        self.current_path = []  # Track path during episode
        self.current_mlu = 0.0  # Track running Maximum Link Utilization
        self.current_step = 0  # Track step count
        
        # Episode parameters
        self.alpha = alpha
        self.max_steps = max_steps

    def _init_tfrecords(self, tfrecords_dir, traffic_intensity):
        """Initialize TFRecord dataset and iterator.
        
        Filters files matching the pattern:
        results_nsfnetbw_{traffic_intensity}_Routing_SP_k_*.tfrecords
        """
        pattern = f"{tfrecords_dir}/results_nsfnetbw_{traffic_intensity}_Routing_SP_k_*.tfrecords"
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"No tfrecords found matching pattern: {pattern}")
        
        self.tfrecord_dataset = tf.data.TFRecordDataset(files)
        self.tfrecord_dataset = self.tfrecord_dataset.shuffle(buffer_size=10000)
        self.tfrecord_dataset = self.tfrecord_dataset.repeat()
        self.tfrecord_iterator = iter(self.tfrecord_dataset)
        
        self.feature_description = {
            'traffic': tf.io.FixedLenFeature([182], tf.float32),
        }

    def _sample_traffic(self):
        """Sample a traffic matrix from TFRecords."""
        raw_record = next(self.tfrecord_iterator)
        example = tf.io.parse_single_example(raw_record, self.feature_description)
        return example['traffic'].numpy()

    def _flat_to_pair(self, flat_idx):
        """Convert flat traffic matrix index to (src, dst) pair.
        
        For N nodes, traffic matrix has N*(N-1) entries (no self-loops).
        Example for N=14: indices 0-12 are src=0, dst=1..13
                          indices 13-25 are src=1, dst=0,2..13, etc.
        """
        n = self.num_nodes
        src = flat_idx // (n - 1)
        dst_offset = flat_idx % (n - 1)
        dst = dst_offset if dst_offset < src else dst_offset + 1
        return src, dst

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.tfrecord_iterator:
            # Sample traffic from TFRecords
            traffic_matrix = self._sample_traffic()  # Shape: (182,)
            
            # Select a random source-destination pair from the traffic matrix
            flat_idx = self.np_random.integers(0, len(traffic_matrix))
            src, dst = self._flat_to_pair(flat_idx)
            
            self.current_node = src
            self.destination = dst
            self.current_traffic = traffic_matrix[flat_idx]
            self.traffic_matrix = traffic_matrix
            
            # Initialize path tracking
            self.current_path = [self.current_node]
            self.current_mlu = 0.0  # Reset running MLU
            self.current_step = 0  # Reset step counter
        else:
            raise ValueError(
                "TFRecords directory not provided. "
                "Please initialize the environment with tfrecords_dir parameter: "
                "AdaptivePathEnv(tfrecords_dir='data/nsfnetbw/tfrecords/train')"
            )
        
        return self._get_obs(), {}

    def _get_link_utilization(self, u, v, key):
        """Get utilization of a specific link for current traffic."""
        capacity = self.kdn.link_caps.get((u, v, key), 0.0)
        return self.current_traffic / capacity if capacity > 0 else 1.0

    def step(self, action):
        """
        Takes an action (edge index) and moves the agent if valid.
        
        Reward based on MLU change: R_t = -alpha * (MLU_t - MLU_{t-1})
        """
        # action is an index into self.edges
        u, v, key = self.edges[action]
        
        # Increment step counter
        self.current_step += 1
        
        # Track MLU before action
        mlu_before = self.current_mlu
        
        # Reward components
        step_penalty = -0.1  # Penalty for every step taken
        invalid_penalty = -1.0  # Strong penalty for selecting invalid edge
        terminated = False
        
        # Check if the selected edge is valid (starts from current node)
        if u == self.current_node:
            # Valid action: move to the next node
            self.current_node = v
            self.current_path.append(v)
            
            # Incremental MLU Update: Max(current_mlu, new_link_utilization)
            new_link_util = self._get_link_utilization(u, v, key)
            self.current_mlu = max(self.current_mlu, new_link_util)
            
            # Congestion-based component: delta in Max Link Utilization
            mlu_reward = -self.alpha * (self.current_mlu - mlu_before)
            
            # Total reward for valid action: step_penalty + mlu_reward
            reward = step_penalty + mlu_reward
            
            # Check if reached destination
            if self.current_node == self.destination:
                terminated = True
        else:
            # Invalid action: combined step_penalty and invalid_penalty
            reward = step_penalty + invalid_penalty
        
        # Check if max steps reached
        truncated = self.current_step >= self.max_steps
        info = {
            'mlu': self.current_mlu,
            'path_length': len(self.current_path),
            'is_success': terminated
        }
        
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return {
            "current_node": np.array([self.current_node], dtype=int),
            "destination": np.array([self.destination], dtype=int),
            "traffic": self.traffic_matrix.copy(),
        }
