"""
FrozenLake.

Implementation of the Bellman equation with time
difference correction for the stochastic scenario
"""

import os

import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from gymnasium.envs.registration import register

# Stochastic
register(
    id="FrozenLakeStochastic-v1",
    entry_point="gymnasium.envs.toy_text.frozen_lake:FrozenLakeEnv",
    kwargs={"map_name": "4x4", "is_slippery": True},
)

env = gym.make("FrozenLakeStochastic-v1")
OUTPUT_DIR = "./docs/rl07"
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
GAMMA = 0.9  # Discount factor
LEARNING_RATE = 0.8


for i_episode in range(NUM_EPISODES):

    state, _ = env.reset()
    step = 0  # pylint: disable=invalid-name

    while True:

        step += 1

        random_values = Q[state] + torch.rand(1, number_of_actions) / 1000

        action = torch.argmax(random_values).item()

        new_state, reward, terminated, truncated, info = env.step(action)

        # Algorithm for stochastic environment
        Q[state, action] = (1 - LEARNING_RATE) * Q[state, action] + LEARNING_RATE * (
            reward + GAMMA * torch.max(Q[new_state])
        )

        # env.render()
        # print_q_table(Q)

        state = new_state

        if terminated:
            row = new_state // env.unwrapped.ncol
            col = new_state % env.unwrapped.ncol
            cell = env.unwrapped.desc[row][col].decode("utf-8")

            # if cell == "H":
            #     print(
            #         f"  Episode {i_episode + 1} terminated after {step} steps: "
            #         f"Fell into a hole at ({row}, {col})."
            #     )
            # elif cell == "G":
            #     print(
            #         f"  Episode {i_episode + 1} terminated after {step} steps: "
            #         f"Reached the goal at ({row}, {col})."
            #     )

            steps_total.append(step)
            rewards_total.append(reward)
            break

        if truncated:
            steps_total.append(step)
            rewards_total.append(reward)
            # print(
            #     f"  Episode {i_episode + 1} truncated after {step} steps: Time limit reached."
            # )
            break


# Print statistics
print(f"\nTraning completed after {NUM_EPISODES} episodes.")
print(
    "Hyperparameters: \n"
    f"\t Discount factor: {GAMMA}\n"
    f"\t Learning rate: {LEARNING_RATE}"
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
reward_plot_path = os.path.join(OUTPUT_DIR, "rl07_rewards_per_episode.png")
plt.savefig(reward_plot_path, dpi=300)
plt.close()
print(f"Saved: {reward_plot_path}")


plt.figure(figsize=(12, 5))
plt.title("Steps / Episode length")
plt.bar(torch.arange(len(steps_total)), steps_total, alpha=0.6, color="red", width=5)
plt.xlabel("Episode")
plt.ylabel("Steps")
steps_plot_path = os.path.join(OUTPUT_DIR, "rl07_steps_per_episode.png")
plt.savefig(steps_plot_path, dpi=300)
plt.close()
