# Regression Trees Midterm

import numpy as np

# Class to represent a node in the regression tree
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, depth=0):
        # feature to split at
        self.feature = feature
        # threshold for the split
        self.threshold = threshold
        self.left = left
        self.right = right
        # mean target value
        self.value = value
        self.depth = depth

# Class for a regression tree    
class RegressionTree:
    def __init__(self, X, y, max_depth=None, min_leaf_size=None, control='depth'):
        # Max depth of tree
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.control = control
        # build tree from root
        self.root = self._build_tree(np.array(X), np.array(y), depth=0)

    # Function to calculate the sum of squared errors
    def _sum_squared_errors(self, y):
        # If no samples
        if len(y) == 0:
            return 0
        mean = np.mean(y)
        # Sum of squared differences from mean
        sum_squared_errors = np.sum((y - mean) ** 2)
        return sum_squared_errors
    
    # Function to find the best feature to split at (Step 2 in Notes Algorithm)
    def _best_split(self, X, y):
        best_sum_squared_errors = self._sum_squared_errors(y)
        best_feature = None
        best_threshold = None

        # Loop through all features
        for feature in range(X.shape[1]):
            # Sort values by feature
            sorted = X[:, feature].argsort()
            X_sorted = X[sorted]
            y_sorted = y[sorted]

            # Try all features to split at
            for i in range(1, len(y_sorted)):
                # If values don't change just skip it
                if X_sorted[i, feature] == X_sorted[i - 1, feature]:
                    continue
                
                # Comput threshold between the two points
                threshold = (X_sorted[i, feature] + X_sorted[i - 1, feature]) / 2
                left_y = y_sorted[:i]
                right_y = y_sorted[i:]
                # Total error after split
                sum_squared_errors_split = self._sum_squared_errors(left_y) + self._sum_squared_errors(right_y)
                # Update if this is the new best split feature
                if sum_squared_errors_split < best_sum_squared_errors:
                    best_sum_squared_errors = sum_squared_errors_split
                    best_feature = feature
                    best_threshold = threshold
        # best split
        return best_feature, best_threshold
    
    # Function for building the regression tree (step 1 in Notes Algorith)
    def _build_tree(self, X, y, depth):
        # checks if all points have same variablees
        if len(set(map(tuple, X))) == 1:
            return Node(value=np.mean(y), depth=depth)
        # Check if max depth is reached
        if self.control == 'depth' and self.max_depth is not None and depth >= self.max_depth:
            return Node(value=np.mean(y), depth=depth)
        # Check if node size is below leaf size limit
        if self.control == 'leaf_size' and self.min_leaf_size is not None and len(y) <= self.min_leaf_size:
            return Node(value=np.mean(y), depth=depth)
        # Find best split for current node
        feature, threshold = self._best_split(X, y)
        # No error reduction
        if feature is None:
            return Node(value=np.mean(y), depth=depth)
        # Split data based on threshold
        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold
        # Don't split into empty children
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return Node(value=np.mean(y), depth=depth)
        
        # Build left and right subtrees
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        # return node with split
        return Node(feature, threshold, left_child, right_child, depth=depth)
    
    # Function to traverse tree for prediction and path (Step 3 in Notes Algorithm)
    def _traverse(self, node, x, path=None):
        if path is None:
            path = []
        # Check if node is a leaf
        if node.value is not None:
            return node.value, path
        
        # Decide left or right traversal based on feature
        if x[node.feature] <= node.threshold:
            path.append(f"X[{node.feature}] <= {node.threshold}")
            return self._traverse(node.left, x, path)
        
        else:
            path.append(f"X[{node.feature}] > {node.threshold}")
            return self._traverse(node.right, x, path)

    # Function to predict value for input
    def predict(self, x):
        return self._traverse(self.root, np.array(x))[0]

    # Function to return path for input
    def decision_path(self, x):
        _, path = self._traverse(self.root, np.array(x))
        return path