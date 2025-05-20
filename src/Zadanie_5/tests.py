import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

from MLP import MLP
from MLP_Better import MLP_Better





df = pd.read_csv('wsi5-25L_dataset.csv')
X = df.drop(columns=['quality']).values
y = df['quality'].values

X = X.T

# 4. One-hot encoding etykiet
encoder = OneHotEncoder(sparse_output=False)
Y = encoder.fit_transform(y.reshape(-1, 1)).T



def getting_random_vectors():
    X_train, X_temp, y_train_idx, y_temp_idx = train_test_split(
        X.T, y, test_size=0.4, stratify=y)
    X_val, X_test, y_val_idx, y_test_idx = train_test_split(
        X_temp, y_temp_idx, test_size=0.5, stratify=y_temp_idx)

    Y_train = encoder.transform(y_train_idx.reshape(-1, 1)).T
    Y_val = encoder.transform(y_val_idx.reshape(-1, 1)).T
    Y_test = encoder.transform(y_test_idx.reshape(-1, 1)).T

    X_train, X_val, X_test = X_train.T, X_val.T, X_test.T

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, y_train_idx, y_val_idx, y_test_idx

accuracies = []
accuracies_better = []

def train_mlp():
    X_train, Y_train, X_val, Y_val, X_test, Y_test, y_train_idx, y_val_idx, y_test_idx = getting_random_vectors()
    mlp = MLP([X_train.shape[0], 32, 16, Y_train.shape[0]])
    val_costs, test_costs =  mlp.fit(
        X_train, Y_train, X_val, Y_val,
        epochs=30_000, alpha=0.01,
        print_every=1000
    )
    return val_costs, test_costs

def train_mlp_better():
    X_train, Y_train, X_val, Y_val, X_test, Y_test, y_train_idx, y_val_idx, y_test_idx = getting_random_vectors()
    mlp_better = MLP_Better([X_train.shape[0], 32, 16, Y_train.shape[0]])
    val_costs, test_costs = mlp_better.fit(
        X_train, Y_train, X_val, Y_val,
        epochs=30_000, alpha=0.01,
        lambd=0.01, keep_prob=0.8,
        print_every=1000
    )
    return val_costs, test_costs


def plot_costs(val_costs, test_costs, title=""):


    epochs_range = range(1, len(val_costs) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, test_costs, label='Koszt dla zbioru testowego')
    plt.plot(epochs_range, val_costs, label='Koszt dla zbioru walidacyjnego')
    plt.xlabel('Epoka')
    plt.ylabel('Koszt')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{title}.png')
    plt.close()

val_costs, test_costs = train_mlp()
plot_costs(val_costs, test_costs, title="Zwykły perceptron — Koszt danych treningowych i walidacyjnych")

val_costs_better, test_costs_better = train_mlp_better()
plot_costs(val_costs_better, test_costs_better, title = "Ulepszony perceptron — Koszt danych treningowych i walidacyjnych")



for i in range(5):
    X_train, Y_train, X_val, Y_val, X_test, Y_test, y_train_idx, y_val_idx, y_test_idx = getting_random_vectors()
    mlp = MLP([X_train.shape[0], 32, 16, Y_train.shape[0]])
    mlp.fit(
        X_train, Y_train, X_val, Y_val,
        epochs=30_000, alpha=0.01,
        print_every=1000
    )
    preds = mlp.predict(X_test)
    accuracy = np.sum(preds == y_test_idx) / len(y_test_idx)
    print(f"Test {i + 1}/5 (zwykłe); Dokładność: {accuracy * 100:.2f}%")
    accuracies.append(accuracy)

    mlp_better = MLP_Better([X_train.shape[0], 32, 16, Y_train.shape[0]])
    mlp_better.fit(
        X_train, Y_train, X_val, Y_val,
        epochs=30_000, alpha=0.01,
        lambd=0.01, keep_prob=0.8,
        print_every=1000
    )
    preds = mlp_better.predict(X_test)
    accuracy_better = np.sum(preds == y_test_idx) / len(y_test_idx)
    print(f"Test {i + 1}/5 (ulepszone); Dokładność: {accuracy_better * 100:.2f}%")
    accuracies_better.append(accuracy_better)

print(f"Średnia dokładność (zwykłe): {np.mean(accuracies) * 100:.2f}%")
print(f"Średnia dokładność (ulepszone): {np.mean(accuracies_better) * 100:.2f}%")

