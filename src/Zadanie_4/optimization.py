import numpy as np
import pandas as pd
from array_from_file import data_from_csv
from random_forest import MyRandomForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from evaluation import evaluate_model

# Optymalizacja liczby drzew
n_trees_range = [10, 50, 100, 200, 500]
results = []

X, y =  data_from_csv("Zadanie_4/lab-4-dataset.csv")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

for n_trees in n_trees_range:
    print(f"\nTestowanie z n_trees = {n_trees}")
    model = MyRandomForest(n_trees=n_trees)
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
    results.append({
        'n_trees': n_trees,
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'roc_auc': metrics['roc_auc']
    })

# Wizualizacja wyników optymalizacji
results_df = pd.DataFrame(results)
plt.figure(figsize=(12, 8))
for metric in ['accuracy', 'precision', 'recall', 'roc_auc']:
    plt.plot(results_df['n_trees'], results_df[metric], label=metric)
plt.xlabel('Liczba drzew')
plt.ylabel('Wartość metryki')
plt.title('Wpływ liczby drzew na jakość modelu')
plt.legend()
plt.grid()
plt.show()

# Wybór najlepszej liczby drzew
best_result = results_df.loc[results_df['roc_auc'].idxmax()]
print(f"\nNajlepsze wyniki osiągnięto dla n_trees = {best_result['n_trees']}:")
print(f"Accuracy: {best_result['accuracy']:.4f}")
print(f"Precision: {best_result['precision']:.4f}")
print(f"Recall: {best_result['recall']:.4f}")
print(f"ROC AUC: {best_result['roc_auc']:.4f}")