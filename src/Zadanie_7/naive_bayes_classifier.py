import numpy as np

class NaiveBayesClassifier:
    """
    Naiwny klasyfikator Bayesa dla danych ciągłych, z założeniem
    rozkładu normalnego (Gaussa) dla każdej cechy.
    """
    def fit(self, X, y):
        """
        "Trenuje" model poprzez obliczenie średnich, odchyleń standardowych
        i prawdopodobieństw a priori dla każdej klasy.

        Args:
            X (np.array): Dane treningowe (cechy).
            y (np.array): Etykiety treningowe.
        """
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # Inicjalizacja struktur do przechowywania statystyk
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        """
        Przewiduje etykiety dla nowych danych.

        Args:
            X (np.array): Dane do predykcji.

        Returns:
            list: Lista przewidzianych etykiet.
        """
        y_pred = [self._predict_sample(x) for x in X]
        return np.array(y_pred)

    def _predict_sample(self, x):
        """
        Klasyfikuje pojedynczą próbkę danych.
        """
        posteriors = []

        # Oblicz prawdopodobieństwo a posteriori dla każdej klasy
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            
            # Oblicz logarytm gęstości prawdopodobieństwa (likelihood)
            # Używamy logarytmów, aby uniknąć problemów z bardzo małymi liczbami (underflow)
            class_conditional_likelihood = np.sum(self._pdf(idx, x))
            
            posterior = prior + class_conditional_likelihood
            posteriors.append(posterior)

        # Zwróć klasę z najwyższym prawdopodobieństwem a posteriori
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        """
        Oblicza logarytm gęstości prawdopodobieństwa (PDF) rozkładu normalnego.
        """
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        
        # Dodajemy małą stałą (epsilon) do wariancji, aby uniknąć dzielenia przez zero
        epsilon = 1e-9
        numerator = np.exp(- (x - mean)**2 / (2 * (var + epsilon)))
        denominator = np.sqrt(2 * np.pi * (var + epsilon))
        
        # Zwracamy logarytm PDF
        return np.log(numerator / denominator)

