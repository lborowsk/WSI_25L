# q_learning_agent.py

import numpy as np

class QLearningAgent:
    """
    Ogólny agent Q-Learning.
    """
    def __init__(
        self,
        observation_space_size: int,
        action_space_size: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.999,
        min_epsilon: float = 0.01,
    ):
        """
        Inicjalizuje agenta Q-Learning.

        Args:
            observation_space_size (int): Liczba stanów w środowisku.
            action_space_size (int): Liczba możliwych akcji.
            learning_rate (float): Współczynnik uczenia (w notatkach oznaczony jako β).
            discount_factor (float): Współczynnik dyskontowania (gamma, γ).
            epsilon (float): Początkowa wartość współczynnika eksploracji (ε).
            epsilon_decay (float): Mnożnik, przez który epsilon jest zmniejszany po każdym epizodzie.
            min_epsilon (float): Minimalna wartość dla epsilon.
        """
        self.action_space_size = action_space_size
        # Inicjalizacja tablicy Q-table zerami
        self.q_table = np.zeros((observation_space_size, action_space_size))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def choose_action(self, state: int) -> int:
        """
        Wybiera akcję, korzystając ze strategii ε-greedy (epsilon-zachłannej).
        """
        # Z prawdopodobieństwem epsilon, eksploruj (wybierz losową akcję)
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space_size)
        # W przeciwnym razie, eksploatuj (wybierz najlepszą znaną akcję)
        else:
            return np.argmax(self.q_table[state, :])

    def learn(self, state: int, action: int, reward: float, next_state: int):
        """
        Aktualizuje tablicę Q, używając reguły aktualizacji Q-Learning.
        """
        # Znajdź maksymalną wartość Q dla następnego stanu
        max_next_q = np.max(self.q_table[next_state, :])

        # Główna formuła Q-Learning
        current_q = self.q_table[state, action]
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state, action] = new_q

    def decay_epsilon(self):
        """
        Zmniejsza współczynnik eksploracji (epsilon) w czasie.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)