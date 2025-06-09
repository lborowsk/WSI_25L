import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from q_learning_agent import QLearningAgent
from sarsa_agent import SarsaAgent
from action_runner import train, train_sarsa

def visualize_agent_path(agent, agent_name, env_name):
    """
    Wizualizuje ścieżkę nauczonego agenta w jednym epizodzie.
    """
    print(f"\n--- Wizualizacja ścieżki agenta: {agent_name} ---")
    env = gym.make(env_name, render_mode="human")
    state, _ = env.reset()
    
    agent.epsilon = 0  # Wyłączamy eksplorację
    
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
    
    all_q_rewards = []
    all_sarsa_rewards = []

    print(f"Rozpoczynanie {NUM_RUNS} przebiegów po {EPISODES_PER_RUN} epizodów każdy...")

    for run in range(NUM_RUNS):
        print(f"\n--- Przebieg {run + 1}/{NUM_RUNS} ---")
        
        env = gym.make(ENV_NAME)
        q_agent = QLearningAgent(obs_space_size, act_space_size, LEARNING_RATE, DISCOUNT_FACTOR)
        sarsa_agent = SarsaAgent(obs_space_size, act_space_size, LEARNING_RATE, DISCOUNT_FACTOR)
        
        print("Trenowanie Agenta Q-Learning...")
        q_rewards = train(q_agent, env, EPISODES_PER_RUN)
        all_q_rewards.append(q_rewards)
        
        print("Trenowanie Agenta SARSA...")
        sarsa_rewards = train_sarsa(sarsa_agent, env, EPISODES_PER_RUN)
        all_sarsa_rewards.append(sarsa_rewards)

        env.close()

    # --- ANALIZA I WIZUALIZACJA WYNIKÓW ---
    
    q_rewards_np = np.array(all_q_rewards)
    sarsa_rewards_np = np.array(all_sarsa_rewards)

    mean_q = np.mean(q_rewards_np, axis=0)
    std_q = np.std(q_rewards_np, axis=0)
    
    mean_sarsa = np.mean(sarsa_rewards_np, axis=0)
    std_sarsa = np.std(sarsa_rewards_np, axis=0)
    
    last_100_q_mean = np.mean(q_rewards_np[:, -100:])
    last_100_sarsa_mean = np.mean(sarsa_rewards_np[:, -100:])
    
    print("\n\n--- Wyniki Końcowe (średnia z ostatnich 100 epizodów) ---")
    print(f"Agent Q-Learning: {last_100_q_mean:.2f}")
    print(f"Agent SARSA: {last_100_sarsa_mean:.2f}")

    # --- Tworzenie wykresu ---
    plt.figure(figsize=(12, 8))
    
    plt.plot(mean_q, label='Q-Learning (Średnia)', color='blue')
    plt.fill_between(range(EPISODES_PER_RUN), mean_q - std_q, mean_q + std_q, color='blue', alpha=0.2)

    plt.plot(mean_sarsa, label='SARSA (Średnia)', color='green')
    plt.fill_between(range(EPISODES_PER_RUN), mean_sarsa - std_sarsa, mean_sarsa + std_sarsa, color='green', alpha=0.2)
    
    plt.title(f'Porównanie Q-Learning vs SARSA w środowisku CliffWalking\n({NUM_RUNS} przebiegów)')
    plt.xlabel('Epizody')
    plt.ylabel('Suma nagród')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('wyniki_sarsa_vs_q.png')
    print("\nZapisano wykres do pliku 'wyniki_sarsa_vs_q.png'")
    
    # --- Wizualizacja ścieżek obu agentów ---
    visualize_agent_path(q_agent, "Q-Learning", ENV_NAME)
    visualize_agent_path(sarsa_agent, "SARSA", ENV_NAME)