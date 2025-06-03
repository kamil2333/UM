import numpy as np
import gymnasium as gym
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


env = gym.make("CartPole-v1") 

num_bins_pos = 10
num_bins_vel = 10
num_bins_ang = 10
num_bins_ang_vel = 10

pos_bins = np.linspace(-2.4, 2.4, num_bins_pos + 1)[1:-1]
vel_bins = np.linspace(-4.0, 4.0, num_bins_vel + 1)[1:-1]
ang_bins = np.linspace(-0.2095, 0.2095, num_bins_ang + 1)[1:-1]
ang_vel_bins = np.linspace(-4.0, 4.0, num_bins_ang_vel + 1)[1:-1]

all_bins = [pos_bins, vel_bins, ang_bins, ang_vel_bins]
n_states_per_dim = [len(b) + 1 for b in all_bins]
n_states_total = np.prod(n_states_per_dim)

def get_discrete_state(observation):
    observation = np.clip(observation,
                          [-4.8, -float('inf'), -0.418, -float('inf')],
                          [4.8, float('inf'), 0.418, float('inf')])
    
    pos_bin = np.digitize(observation[0], pos_bins)
    vel_bin = np.digitize(observation[1], vel_bins)
    ang_bin = np.digitize(observation[2], ang_bins)
    ang_vel_bin = np.digitize(observation[3], ang_vel_bins)
    
    state_idx = np.ravel_multi_index((pos_bin, vel_bin, ang_bin, ang_vel_bin), dims=n_states_per_dim)
    return state_idx

n_actions = env.action_space.n
Q = np.zeros((n_states_total, n_actions))

initial_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.00005
alpha = 0.1
gamma = 0.99
episodes = 50000

rewards = []

print(f"Całkowita liczba dyskretnych stanów: {n_states_total}")
print(f"Rozmiar tabeli Q: {Q.shape}")
print(f"Rozpoczynam trening {episodes} epizodów BEZ renderowania. Proszę czekać...")

for episode in range(episodes):
    observation, info = env.reset()
    state = get_discrete_state(observation)
    done = False
    total_reward = 0

    epsilon = min_epsilon + (initial_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    
    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        next_observation, reward, terminated, truncated, info = env.step(action)
        next_state = get_discrete_state(next_observation)

        done = terminated or truncated

        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state, :]) - Q[state, action]
        )

        state = next_state
        total_reward += reward

    rewards.append(total_reward)

    if episode % 1000 == 0:
        print(f"Epizod: {episode}/{episodes}, Epsilon: {epsilon:.4f}, Ostatnia nagroda: {total_reward}")

env.close()

print("\nTrening zakończony.")

print("\n--- Testowanie nauczonego agenta (z renderowaniem) ---")
test_env = gym.make("CartPole-v1", render_mode="human")
test_episodes = 5
for i in range(test_episodes):
    observation, info = test_env.reset()
    state = get_discrete_state(observation)
    done = False
    total_test_reward = 0
    print(f"Test Epizod {i+1}/{test_episodes}:")
    while not done:
        test_env.render()
        time.sleep(0.02)

        action = np.argmax(Q[state, :])

        next_observation, reward, terminated, truncated, info = test_env.step(action)
        next_state = get_discrete_state(next_observation)
        done = terminated or truncated

        state = next_state
        total_test_reward += reward
    print(f"  Całkowita nagroda: {total_test_reward}")
test_env.close()

print("\nGenerowanie wykresu wyników treningu...")
avg_rewards = []
chunk_size = 1000
for i in range(0, episodes, chunk_size):
    chunk = rewards[i:i+chunk_size]
    if chunk:
        avg_rewards.append(np.mean(chunk))
    else:
        break

plt.plot(avg_rewards)
plt.xlabel(f"Kolejne {chunk_size} epizodów (z {episodes} łącznie)")
plt.ylabel("Średnia nagroda (liczba kroków przeżycia)")
plt.title("Uczenie agenta Q-learning (CartPole)")
plt.grid(True)
plt.show()

print("\nProgram zakończony.")