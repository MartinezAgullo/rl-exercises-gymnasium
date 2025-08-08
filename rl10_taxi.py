"""
Solving Gymansium's Taxi-v3
Using adaptative epsilon greedy strategy
"""

import os

import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from gymnasium.wrappers import RecordVideo

env = gym.make(
    "Taxi-v3", is_rainy=False, fickle_passenger=False, render_mode="rgb_array"
)
OUTPUT_DIR = "./docs/rl10"
os.makedirs(OUTPUT_DIR, exist_ok=True)
env = RecordVideo(env, video_folder=OUTPUT_DIR, episode_trigger=lambda x: x % 100 == 0)


plt.style.use("ggplot")


NUM_EPISODES = 300
steps_total = []
rewards_total = []
epsilon_total = []

# Initialize Q-table
number_of_states = env.observation_space.n
number_of_actions = env.action_space.n
Q = torch.zeros([number_of_states, number_of_actions])

# Hyperparameters
GAMMA = 0.9
LEARNING_RATE = 1
EPSILON_INITIAL = 0.9
EPSILON_FINAL = 0.001
EPSILON_DECAY = 0.999


epsilon = EPSILON_INITIAL  # pylint: disable=invalid-name

for i_episode in range(NUM_EPISODES):

    state, _ = env.reset()
    step = 0  # pylint: disable=invalid-name

    while True:

        step += 1

        random_values = Q[state] + torch.rand(1, number_of_actions) / 1000

        # Greedy epsilon strategy
        random_for_egreedy = torch.rand(1).item()
        if random_for_egreedy < epsilon:
            action = env.action_space.sample()
        else:
            action = torch.argmax(random_values).item()

        # Update epsilon if necessary
        if epsilon > EPSILON_FINAL:
            epsilon = EPSILON_DECAY * epsilon  # pylint: disable=invalid-name

        new_state, reward, terminated, truncated, info = env.step(action)

        # Q-value (LEARNING_RATE = 1 because deterministic)
        Q[state, action] = (1 - LEARNING_RATE) * Q[state, action] + LEARNING_RATE * (
            reward + GAMMA * torch.max(Q[new_state])
        )

        env.render()

        state = new_state

        if terminated or truncated:
            steps_total.append(step)
            rewards_total.append(reward)
            epsilon_total.append(epsilon)
            break


# Print statistics
print(f"\nTraning completed after {NUM_EPISODES} episodes.")
print(
    "Hyperparameters: \n"
    f"\t Discount factor: {GAMMA}\n"
    f"\t Learning rate: {LEARNING_RATE} (becuase deterministic)\n"
    "\t Epsilon for exploration:\n"
    f"\t \t Initial {EPSILON_INITIAL}\n"
    f"\t \t Final {EPSILON_FINAL}\n"
    f"\t \t Decay multipliyer {EPSILON_DECAY}\n"
)


N_EPISODES_FOR_STATS = 100
successful_episodes = sum(1 for r in rewards_total if r == 20)
successful_last_N = sum(1 for r in rewards_total[-N_EPISODES_FOR_STATS:] if r == 20)

print(
    f"Percent of episodes finished successfully: {100*successful_episodes / NUM_EPISODES}%"
)
print(
    f"Percent of episodes finished successfully (last {N_EPISODES_FOR_STATS} episodes): "
    f"{100*successful_last_N / N_EPISODES_FOR_STATS}%"
)

print(f"Average number of steps: {sum(steps_total) / NUM_EPISODES:.2f}")
print(
    f"Average number of steps (last {N_EPISODES_FOR_STATS} episodes): "
    f"{sum(steps_total[-N_EPISODES_FOR_STATS:]) / N_EPISODES_FOR_STATS:.2f}"
)


plt.figure(figsize=(12, 5))
plt.title("Rewards per Episode")
plt.bar(
    torch.arange(len(rewards_total)), rewards_total, alpha=0.6, color="green", width=5
)
plt.xlabel("Episode")
plt.ylabel("Reward")
reward_plot_path = os.path.join(OUTPUT_DIR, "rl10_rewards_per_episode.png")
plt.savefig(reward_plot_path, dpi=300)
plt.close()

plt.figure(figsize=(12, 5))
plt.title("Steps / Episode length")
plt.bar(torch.arange(len(steps_total)), steps_total, alpha=0.6, color="red", width=5)
plt.xlabel("Episode")
plt.ylabel("Steps")
steps_plot_path = os.path.join(OUTPUT_DIR, "rl10_steps_per_episode.png")
plt.savefig(steps_plot_path, dpi=300)
plt.close()


plt.figure(figsize=(12, 5))
plt.title("Epsilon greedy")
plt.bar(
    torch.arange(len(epsilon_total)), epsilon_total, alpha=0.6, color="blue", width=5
)
plt.xlabel("Episode")
plt.ylabel("Epsilon")
steps_plot_path = os.path.join(OUTPUT_DIR, "rl10_epsilon_per_episode_with_decay.png")
plt.savefig(steps_plot_path, dpi=300)
plt.close()
