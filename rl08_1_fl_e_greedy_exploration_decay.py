"""
Deterministic FrozenLake.

Implementation of epsilon-greedy exploration strategy.
The epsilon is adjusted at every episode so that the
exploration is favoured at the begining
"""

import os

import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from gymnasium.envs.registration import register

# Deterministic
register(
    id="FrozenLakeDeterministic-v1",
    entry_point="gymnasium.envs.toy_text.frozen_lake:FrozenLakeEnv",
    kwargs={"map_name": "4x4", "is_slippery": False},
)

env = gym.make("FrozenLakeDeterministic-v1")
OUTPUT_DIR = "./docs/rl08"
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.style.use("ggplot")


NUM_EPISODES = 1000
steps_total = []
rewards_total = []

# Initialize Q-table
number_of_states = env.observation_space.n
number_of_actions = env.action_space.n
Q = torch.zeros([number_of_states, number_of_actions])

# Hyperparameters
# The LR is set to 1 so that the Q-value is that
# of the Bellman eq for deterministic environments.
GAMMA = 0.9
LEARNING_RATE = 1
EPSILON_INITIAL = 0.7
EPSILON_FINAL = 0.1
EPSILON_DECAY = 0.999

epsilon = EPSILON_INITIAL  # pylint: disable=invalid-name
for i_episode in range(NUM_EPISODES):

    state, _ = env.reset()
    step = 0  # pylint: disable=invalid-name

    while True:

        step += 1

        random_values = Q[state] + torch.rand(1, number_of_actions) / 1000

        # Generate random number to compare with epsilon
        random_for_egreedy = torch.rand(1).item()

        # Exploration strategy
        if random_for_egreedy < epsilon:
            # Explore new actions - Random movement
            action = env.action_space.sample()
        else:
            # Chose best action (to our knoledge)
            action = torch.argmax(random_values).item()

        # Update epsilon if necessary
        if epsilon > EPSILON_FINAL:
            epsilon = EPSILON_DECAY * epsilon  # pylint: disable=invalid-name

        new_state, reward, terminated, truncated, info = env.step(action)

        # Q-value
        Q[state, action] = (1 - LEARNING_RATE) * Q[state, action] + LEARNING_RATE * (
            reward + GAMMA * torch.max(Q[new_state])
        )

        state = new_state

        if terminated:
            steps_total.append(step)
            rewards_total.append(reward)
            break

        if truncated:
            steps_total.append(step)
            rewards_total.append(reward)
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
print(
    f"Percent of episodes finished successfully: {100*sum(rewards_total) / NUM_EPISODES}%"
)
print(
    f"Percent of episodes finished successfully (last {N_EPISODES_FOR_STATS} episodes): "
    f"{100*sum(rewards_total[-N_EPISODES_FOR_STATS:]) / N_EPISODES_FOR_STATS}%"
)

print(f"Average number of steps: {sum(steps_total) / NUM_EPISODES:.2f}")
print(
    f"Average number of steps (last 100 episodes): {sum(steps_total[-100:]) / 100:.2f}"
)


plt.figure(figsize=(12, 5))
plt.title("Rewards per Episode")
plt.bar(
    torch.arange(len(rewards_total)), rewards_total, alpha=0.6, color="green", width=5
)
plt.xlabel("Episode")
plt.ylabel("Reward")
reward_plot_path = os.path.join(OUTPUT_DIR, "rl08b_rewards_per_episode_with_decay.png")
plt.savefig(reward_plot_path, dpi=300)
plt.close()
print(f"Saved: {reward_plot_path}")


plt.figure(figsize=(12, 5))
plt.title("Steps / Episode length")
plt.bar(torch.arange(len(steps_total)), steps_total, alpha=0.6, color="red", width=5)
plt.xlabel("Episode")
plt.ylabel("Steps")
steps_plot_path = os.path.join(OUTPUT_DIR, "rl08b_steps_per_episode_with_decay.png")
plt.savefig(steps_plot_path, dpi=300)
plt.close()
