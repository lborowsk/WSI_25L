# sarsa_agent.py

import numpy as np

class SarsaAgent:
    """
    Ogólny agent SARSA.
    """
    def __init__(
        self,
        observation_space_size: int,
        action_space_size: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.9995,
        min_epsilon: float = 0.01,
    ):
        """
        Inicjalizuje agenta SARSA.

        Args:
            observation_space_size (int): Liczba stanów w środowisku.
            action_space_size (int): Liczba możliwych akcji.
            learning_rate (float): Współczynnik uczenia (β).
            discount_factor (float): Współczynnik dyskontowania (γ).
            epsilon (float): Początkowa wartość współczynnika eksploracji (ε).
            epsilon_decay (float): Mnożnik, przez który epsilon jest zmniejszany.
            min_epsilon (float): Minimalna wartość dla epsilon.
        """
        self.action_space_size = action_space_size
        self.q_table = np.zeros((observation_space_size, action_space_size))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def choose_action(self, state: int) -> int:
        """
        Wybiera akcję, korzystając ze strategii ε-greedy.
        """
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space_size)
        else:
            return np.argmax(self.q_table[state, :])

    def learn(self, state: int, action: int, reward: float, next_state: int, next_action: int):
        """
        Aktualizuje tablicę Q, używając reguły aktualizacji SARSA.
        Kluczowa różnica: używa wartości Q dla faktycznie wybranej następnej akcji (next_action),
        a nie maksymalnej możliwej wartości Q w następnym stanie.
        """
        current_q = self.q_table[state, action]
        next_q = self.q_table[next_state, next_action]
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_q - current_q)
        self.q_table[state, action] = new_q

    def decay_epsilon(self):
        """
        Zmniejsza współczynnik eksploracji (epsilon).
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)