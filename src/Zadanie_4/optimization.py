import pandas as pd
from data_from_file import data_from_csv
from random_forest import MyRandomForest
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np

def evaluate_without_roc(model, X_train, y_train, X_test, y_test):
    """Custom evaluation without ROC plotting"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }

def optimize_random_forest(X, y, n_trees_range, n_runs=5, test_size=0.5):
    """Optimize random forest with multiple runs per tree count"""
    results = []
    
    for n_trees in n_trees_range:
        print(f"\nTesting with n_trees = {n_trees}")
        run_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'roc_auc': []}
        
        for run in range(n_runs):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            model = MyRandomForest(n_trees=n_trees)
            metrics = evaluate_without_roc(model, X_train, y_train, X_test, y_test)
            
            for key in metrics:
                run_metrics[key].append(metrics[key])
        
        # Calculate mean for each metric
        results.append({
            'n_trees': n_trees,
            'accuracy': np.mean(run_metrics['accuracy']),
            'precision': np.mean(run_metrics['precision']),
            'recall': np.mean(run_metrics['recall']),
            'roc_auc': np.mean(run_metrics['roc_auc'])
        })
    
    return pd.DataFrame(results)

def plot_optimization_results(results_df):
    """Plot optimization results"""
    plt.figure(figsize=(12, 8))
    
    metrics = ['accuracy', 'precision', 'recall', 'roc_auc']
    colors = ['blue', 'green', 'red', 'purple']
    
    for metric, color in zip(metrics, colors):
        plt.plot(
            results_df['n_trees'],
            results_df[metric],
            '-o',
            label=metric,
            color=color
        )
    
    plt.xlabel('Number of trees')
    plt.ylabel('Metric value')
    plt.title('Random Forest Optimization (averaged over 5 runs)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Load data
    X, y = data_from_csv("./lab-4-dataset.csv")
    
    # Define range of trees to test
    n_trees_range = [10, 50, 100, 200, 500, 600, 700, 800, 900]
    
    # Run optimization
    results_df = optimize_random_forest(X, y, n_trees_range)
    
    # Display averaged results
    print("\nAverage results over 5 runs:")
    print(results_df.to_string(index=False))
    
    # Plot results
    plot_optimization_results(results_df)
    
    # Find and display best configuration
    best_idx = results_df['roc_auc'].idxmax()
    best_result = results_df.loc[best_idx]
    
    print("\nBest configuration:")
    print(f"Number of trees: {best_result['n_trees']}")
    print(f"Accuracy: {best_result['accuracy']:.4f}")
    print(f"Precision: {best_result['precision']:.4f}")
    print(f"Recall: {best_result['recall']:.4f}")
    print(f"ROC AUC: {best_result['roc_auc']:.4f}")