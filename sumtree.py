import numpy as np

class SumTree:
    """
    Binary heap for O(log n) prioritized sampling.
    Implementation based on typical RL prioritized experience replay.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        # The tree has 2*capacity - 1 nodes
        # Leaves start at index capacity - 1
        self.tree = np.zeros(2 * capacity - 1)
        self.data_pointer = 0
        self.count = 0
        self._max_priority = 1.0  # Initialize with 1.0 to avoid zero priority issues

    def update(self, data_idx: int, priority: float):
        """Update priority at a given data index."""
        # Validate index
        if data_idx < 0 or data_idx >= self.capacity:
            raise IndexError(f"Data index {data_idx} out of range [0, {self.capacity-1}]")

        tree_idx = data_idx + self.capacity - 1
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        
        # Track max priority efficiently
        self._max_priority = max(self._max_priority, priority)

        # Propagate the change up the tree
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v: float):
        """
        Sample a leaf node with cumulative value v.
        Returns: (tree_idx, priority, data_idx)
        """
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1

            # If we reached the leaf layer
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    v -= self.tree[left_child_idx]
                    parent_idx = right_child_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], data_idx

    @property
    def total_priority(self):
        """Return the sum of all priorities."""
        return self.tree[0]

    @property
    def max_priority(self):
        """Return the maximum priority recorded."""
        return self._max_priority

    def add(self, priority: float):
        """Add a new priority (incremental update)."""
        self.update(self.data_pointer, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)
