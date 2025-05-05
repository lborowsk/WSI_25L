from data_from_file import data_from_csv
from random_forest import MyRandomForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Train the model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # ROC plot
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr
    }

X, y = data_from_csv("./lab-4-dataset.csv")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
# Testing custom implementation
print("Custom Random Forest implementation:")
my_rf = MyRandomForest(n_trees=100)
my_metrics = evaluate_model(my_rf, X_train, y_train, X_test, y_test)

# Testing scikit-learn implementation for comparison
print("\nScikit-learn Random Forest:")
sklearn_rf = RandomForestClassifier(n_estimators=100, max_features='sqrt')
sklearn_metrics = evaluate_model(sklearn_rf, X_train, y_train, X_test, y_test)

# Display results
print("\nResults comparison:")
print(f"{'Metric':<15} {'Custom impl.':<15} {'Scikit-learn':<15}")
print(f"{'Accuracy':<15} {my_metrics['accuracy']:.4f}{'':<10} {sklearn_metrics['accuracy']:.4f}")
print(f"{'Precision':<15} {my_metrics['precision']:.4f}{'':<10} {sklearn_metrics['precision']:.4f}")
print(f"{'Recall':<15} {my_metrics['recall']:.4f}{'':<10} {sklearn_metrics['recall']:.4f}")
print(f"{'ROC AUC':<15} {my_metrics['roc_auc']:.4f}{'':<10} {sklearn_metrics['roc_auc']:.4f}")