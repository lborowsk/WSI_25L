# main_experiment.py

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from q_learning_agent import QLearningAgent
from action_runner import train, run_random_agent

def visualize_agent_path(agent, env_name):
    """
    Wizualizuje ścieżkę nauczonego agenta w jednym epizodzie.
    """
    print("\n--- Wizualizacja nauczonej ścieżki ---")
    env = gym.make(env_name, render_mode="human")
    state, _ = env.reset()
    
    # Wyłączamy eksplorację na czas wizualizacji
    agent.epsilon = 0
    
    terminated = False
    truncated = False
    
    while not terminated and not truncated:
        action = agent.choose_action(state)
        next_state, _, terminated, truncated, _ = env.step(action)
        state = next_state
    
    env.close()

# --- GŁÓWNY BLOK EKSPERYMENTU ---
if __name__ == "__main__":
    ENV_NAME = 'CliffWalking-v0'
    env = gym.make(ENV_NAME)
    
    obs_space_size = env.observation_space.n
    act_space_size = env.action_space.n
    env.close()

    # --- HIPERPARAMETRY ---
    NUM_RUNS = 10
    EPISODES_PER_RUN = 1000
    LEARNING_RATE = 0.1
    DISCOUNT_FACTOR = 0.99
    
    # Listy do przechowywania wyników ze wszystkich przebiegów
    all_q_rewards = []
    all_random_rewards = []

    print(f"Rozpoczynanie {NUM_RUNS} przebiegów po {EPISODES_PER_RUN} epizodów każdy...")

    for run in range(NUM_RUNS):
        print(f"\n--- Przebieg {run + 1}/{NUM_RUNS} ---")
        
        env = gym.make(ENV_NAME)
        q_agent = QLearningAgent(obs_space_size, act_space_size, LEARNING_RATE, DISCOUNT_FACTOR)
        
        print("Trenowanie Agenta Q-Learning...")
        q_rewards = train(q_agent, env, EPISODES_PER_RUN)
        all_q_rewards.append(q_rewards)
        
        print("Uruchamianie Agenta Losowego...")
        random_rewards = run_random_agent(env, EPISODES_PER_RUN)
        all_random_rewards.append(random_rewards)

        env.close()

    # --- ANALIZA I WIZUALIZACJA WYNIKÓW ---
    
    q_rewards_np = np.array(all_q_rewards)
    random_rewards_np = np.array(all_random_rewards)

    mean_q = np.mean(q_rewards_np, axis=0)
    std_q = np.std(q_rewards_np, axis=0)
    
    mean_random = np.mean(random_rewards_np, axis=0)
    
    last_100_q_mean = np.mean(q_rewards_np[:, -100:])
    last_100_random_mean = np.mean(random_rewards_np[:, -100:])
    
    print("\n\n--- Wyniki Końcowe (średnia z ostatnich 100 epizodów na przestrzeni wszystkich przebiegów) ---")
    print(f"Agent Q-Learning: {last_100_q_mean:.2f}")
    print(f"Agent Losowy: {last_100_random_mean:.2f}")

    # --- Tworzenie wykresu ---
    plt.figure(figsize=(12, 8))
    
    plt.plot(mean_q, label='Q-Learning (Średnia)', color='blue')
    plt.fill_between(range(EPISODES_PER_RUN), mean_q - std_q, mean_q + std_q, color='blue', alpha=0.2, label='Q-Learning (Odch. Std.)')

    plt.plot(mean_random, label='Agent Losowy (Średnia)', color='red', linestyle='--')
    
    plt.title(f'Porównanie wyników agentów w środowisku CliffWalking\n({NUM_RUNS} przebiegów)')
    plt.xlabel('Epizody')
    plt.ylabel('Suma nagród')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('wyniki_porownawcze_q_vs_random.png')
    print("\nZapisano wykres do pliku 'wyniki_porownawcze_q_vs_random.png'")
    
    visualize_agent_path(q_agent, ENV_NAME)