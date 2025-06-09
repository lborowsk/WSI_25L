# action_runner.py

import numpy as np

def train(agent, env, episodes):
    """
    Trenuje agenta w danym środowisku.
    """
    rewards_per_episode = []
    for episode in range(episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0

        while not terminated and not truncated:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Agent uczy się na podstawie swojego doświadczenia
            agent.learn(state, action, reward, next_state)

            state = next_state
            total_reward += reward

        # Zmniejsz epsilon po każdym epizodzie, aby z czasem ograniczyć eksplorację
        agent.decay_epsilon()
        rewards_per_episode.append(total_reward)
        
        # Logowanie postępów co 100 epizodów
        if (episode + 1) % 100 == 0:
            print(f"Epizod {episode + 1}/{episodes} | Całkowita nagroda: {total_reward}")
            
    return rewards_per_episode

def train_sarsa(agent, env, episodes):
    """
    Trenuje agenta SARSA. Logika jest tu inna i musi podążać za cyklem (S, A, R, S', A').
    """
    rewards_per_episode = []
    for episode in range(episodes):
        state, _ = env.reset()
        action = agent.choose_action(state)
        
        terminated = False
        truncated = False
        total_reward = 0

        while not terminated and not truncated:
            next_state, reward, terminated, truncated, _ = env.step(action)

            next_action = agent.choose_action(next_state)

            agent.learn(state, action, reward, next_state, next_action)
        
            state = next_state
            action = next_action
            
            total_reward += reward

        agent.decay_epsilon()
        rewards_per_episode.append(total_reward)
        
        if (episode + 1) % 100 == 0:
            print(f"SARSA - Epizod {episode + 1}/{episodes} | Całkowita nagroda: {total_reward}")
            
    return rewards_per_episode

def run_random_agent(env, episodes):
    """
    Uruchamia agenta podejmującego wyłącznie losowe akcje w celu porównania.
    """
    rewards_per_episode = []
    for episode in range(episodes):
        state, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        
        while not terminated and not truncated:
            action = env.action_space.sample() 
            next_state, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward

        rewards_per_episode.append(total_reward)

        if (episode + 1) % 100 == 0:
            print(f"Agent losowy - Epizod {episode + 1}/{episodes} | Całkowita nagroda: {total_reward}")

    return rewards_per_episode