import numpy as np
from collections import Counter
from decision_tree import MyDecisionTree

class MyRandomForest:
    def __init__(self, n_trees=100, max_features='sqrt', random_state=None):
        """
        Parameters:
        - n_trees: number of trees in the forest
        - max_features: number of features to consider at each split ('sqrt', 'log2', int)
        - random_state: random seed for reproducibility
        """
        self.n_trees = n_trees
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.feature_indices = []
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the random forest on data X (features) and y (labels).
        
        X: numpy array with shape (n_samples, n_features)
        y: numpy array with shape (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Determine number of features to consider at each split
        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            max_features = int(np.log2(n_features))
        else:
            max_features = int(self.max_features)
        
        self.trees = []
        self.feature_indices = []
        
        for _ in range(self.n_trees):
            # 1. Bootstrap sampling (sampling with replacement)
            sample_idx = np.random.choice(n_samples, size=n_samples, replace=True)
            X_sample = X[sample_idx]
            y_sample = y[sample_idx]
            
            # 2. Random feature subset selection
            feat_idx = np.random.choice(n_features, size=max_features, replace=False)
            X_sample_subset = X_sample[:, feat_idx]
            
            # 3. Train the tree
            tree = MyDecisionTree()
            tree.fit(X_sample_subset, y_sample)
            
            # 4. Save the tree and selected features
            self.trees.append(tree)
            self.feature_indices.append(feat_idx)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples X"""
        all_predictions = np.zeros((X.shape[0], len(self.trees)), dtype=int)
        
        for i, (tree, feat_idx) in enumerate(zip(self.trees, self.feature_indices)):
            X_subset = X[:, feat_idx]
            all_predictions[:, i] = tree.predict(X_subset)
        
        # Majority voting
        return np.array([Counter(row).most_common(1)[0][0] for row in all_predictions])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities"""
        all_predictions = np.zeros((X.shape[0], len(self.trees)), dtype=int)
        
        for i, (tree, feat_idx) in enumerate(zip(self.trees, self.feature_indices)):
            X_subset = X[:, feat_idx]
            all_predictions[:, i] = tree.predict(X_subset)
        
        # Calculate probabilities
        proba = np.zeros((X.shape[0], 2))
        for i in range(X.shape[0]):
            counts = np.bincount(all_predictions[i], minlength=2)
            proba[i] = counts / self.n_trees
        
        return proba