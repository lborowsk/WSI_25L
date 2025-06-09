import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from naive_bayes_classifier import NaiveBayesClassifier

# --- 1. Wczytywanie i przygotowanie danych ---
# Zbiór danych Spambase. Ostatnia kolumna to etykieta (1 = spam, 0 = nie spam).
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
data = pd.read_csv(url, header=None).to_numpy()

X = data[:, :-1]
y = data[:, -1]

# Wstępny, jednorazowy podział na zbiór treningowy (80%) i ostateczny testowy (20%)
# Używamy podziału стратификованого, aby zachować proporcje klas w obu zbiorach.
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Całkowity rozmiar zbioru: {X.shape[0]} próbek")
print(f"Rozmiar zbioru treningowego (do walidacji): {X_train_full.shape[0]} próbek")
print(f"Rozmiar zbioru testowego (końcowa ocena): {X_test.shape[0]} próbek")

# --- 2. Eksperyment A: Wielokrotny losowy podział ---
print("\n--- Rozpoczynam eksperyment A: Wielokrotny losowy podział ---")
n_splits_random = 10
sss = StratifiedShuffleSplit(n_splits=n_splits_random, test_size=0.25, random_state=42)
random_split_scores = []

for i, (train_index, val_index) in enumerate(sss.split(X_train_full, y_train_full)):
    X_train, X_val = X_train_full[train_index], X_train_full[val_index]
    y_train, y_val = y_train_full[train_index], y_train_full[val_index]
    
    model = NaiveBayesClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    score = f1_score(y_val, y_pred)
    random_split_scores.append(score)
    print(f"Losowy podział {i+1}/{n_splits_random}, F1-Score: {score:.4f}")

# --- 3. Eksperyment B: k-krotna walidacja krzyżowa ---
print("\n--- Rozpoczynam eksperyment B: k-krotna walidacja krzyżowa ---")
n_splits_kfold = 10
skf = StratifiedKFold(n_splits=n_splits_kfold, shuffle=True, random_state=42)
kfold_scores = []

for i, (train_index, val_index) in enumerate(skf.split(X_train_full, y_train_full)):
    X_train, X_val = X_train_full[train_index], X_train_full[val_index]
    y_train, y_val = y_train_full[train_index], y_train_full[val_index]
    
    model = NaiveBayesClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    score = f1_score(y_val, y_pred)
    kfold_scores.append(score)
    print(f"Walidacja krzyżowa (fold {i+1}/{n_splits_kfold}), F1-Score: {score:.4f}")

# --- 4. Porównanie i wizualizacja metod walidacji ---
mean_random = np.mean(random_split_scores)
std_random = np.std(random_split_scores)
mean_kfold = np.mean(kfold_scores)
std_kfold = np.std(kfold_scores)

print("\n--- Porównanie metod walidacji (F1-Score) ---")
print(f"Losowe podziały: Średnia = {mean_random:.4f}, Odch. std. = {std_random:.4f}")
print(f"Walidacja krzyżowa: Średnia = {mean_kfold:.4f}, Odch. std. = {std_kfold:.4f}")

# Wykres porównawczy
fig, ax = plt.subplots(figsize=(10, 6))
methods = ['Losowe podziały', 'Walidacja krzyżowa']
means = [mean_random, mean_kfold]
stds = [std_random, std_kfold]

ax.bar(methods, means, yerr=stds, capsize=5, color=['skyblue', 'lightgreen'], alpha=0.8)
ax.set_ylabel('Średni F1-Score')
ax.set_title('Porównanie stabilności metod walidacji')
ax.set_ylim(bottom=min(means) - 2*max(stds), top=max(means) + 2*max(stds))

for i, v in enumerate(means):
    ax.text(i, v + 0.01, f"{v:.3f}", ha='center', va='bottom')

plt.tight_layout()
plt.savefig("porownanie_walidacji.png")
print("\nZapisano wykres do pliku 'porownanie_walidacji.png'")


# --- 5. Ostateczna ocena na zbiorze testowym ---
print("\n--- Ostateczna ocena na zbiorze testowym ---")
final_model = NaiveBayesClassifier()
final_model.fit(X_train_full, y_train_full)
y_pred_final = final_model.predict(X_test)

print("\nRaport klasyfikacji dla ostatecznego modelu:")
print(classification_report(y_test, y_pred_final, target_names=['Not Spam', 'Spam']))

print("Macierz pomyłek:")
print(confusion_matrix(y_test, y_pred_final))