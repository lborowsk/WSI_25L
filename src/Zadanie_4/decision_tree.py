import numpy as np
from collections import Counter

class MyDecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None
    
    def _entropy(self, y):
        """Calculate entropy of a set of labels"""
        counts = np.bincount(y)
        probs = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probs if p > 0])
    
    def _information_gain(self, X_col, y):
        """Calculate information gain for a split based on given column"""
        parent_entropy = self._entropy(y)
        
        # Get unique values and split the dataset
        unique_vals = np.unique(X_col)
        child_entropy = 0
        
        for val in unique_vals:
            mask = X_col == val
            y_child = y[mask]
            child_entropy += (len(y_child) / len(y)) * self._entropy(y_child)
            
        return parent_entropy - child_entropy
    
    def _best_feature(self, X, y, feature_indices):
        """Find the best feature to split on"""
        gains = [self._information_gain(X[:, i], y) for i in feature_indices]
        best_idx = np.argmax(gains)
        return feature_indices[best_idx]
    
    def _build_tree(self, X, y, feature_indices, depth=0):
        """Recursive function to build the tree"""
        # Stopping conditions
        if len(np.unique(y)) == 1:
            return {'class': y[0]}
            
        if len(feature_indices) == 0 or (self.max_depth and depth >= self.max_depth):
            return {'class': Counter(y).most_common(1)[0][0]}
        
        # Select best feature
        best_feat = self._best_feature(X, y, feature_indices)
        remaining_features = [f for f in feature_indices if f != best_feat]
        
        # Build subtrees
        tree = {'feature': best_feat, 'children': {}}
        unique_vals = np.unique(X[:, best_feat])
        
        for val in unique_vals:
            mask = X[:, best_feat] == val
            X_subset = X[mask]
            y_subset = y[mask]
            
            if len(y_subset) == 0:
                tree['children'][val] = {'class': Counter(y).most_common(1)[0][0]}
            else:
                tree['children'][val] = self._build_tree(
                    X_subset, y_subset, remaining_features, depth+1)
        
        return tree
    
    def fit(self, X, y):
        """Train the tree"""
        feature_indices = list(range(X.shape[1]))
        self.tree = self._build_tree(X, y, feature_indices)
    
    def _predict_sample(self, sample, tree):
        """Recursive prediction for a single sample"""
        if 'class' in tree:
            return tree['class']
        
        feat_value = sample[tree['feature']]
        if feat_value in tree['children']:
            return self._predict_sample(sample, tree['children'][feat_value])
        else:
            # If value wasn't seen during training, return most common class
            return self._get_most_common_class(tree)
    
    def _get_most_common_class(self, tree):
        """Helper function to find the most common class in a subtree"""
        if 'class' in tree:
            return tree['class']
        
        classes = []
        for child in tree['children'].values():
            classes.append(self._get_most_common_class(child))
        
        return Counter(classes).most_common(1)[0][0]
    
    def predict(self, X):
        """Predict for multiple samples"""
        return np.array([self._predict_sample(x, self.tree) for x in X])