"""
Random policy for FrozenLake using Gymnasium.

With register, is possible to switch easily between deterministic and
stochastic environments.
Stochastic version (default) -> is_slippery=True
Deterministic version -> is_slippery=False
"""

import os

import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from gymnasium.envs.registration import register

# Register a deterministic version
register(
    id="FrozenLakeDeterministic-v1",
    entry_point="gymnasium.envs.toy_text.frozen_lake:FrozenLakeEnv",
    kwargs={"map_name": "4x4", "is_slippery": False},
)

# Register a stochastic version
register(
    id="FrozenLakeStochastic-v1",
    entry_point="gymnasium.envs.toy_text.frozen_lake:FrozenLakeEnv",
    kwargs={"map_name": "4x4", "is_slippery": True},
)


env = gym.make("FrozenLakeDeterministic-v1")  # or "FrozenLakeStochastic-v1"
OUTPUT_DIR = "./docs/rl04"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_EPISODES = 1000
steps_total = []
rewards_total = []

for i_episode in range(NUM_EPISODES):

    state = env.reset()

    step = 0  # pylint: disable=invalid-name

    while True:

        step += 1

        action = env.action_space.sample()

        new_state, reward, terminated, truncated, info = env.step(action)

        env.render()
        print(new_state)
        print(info)

        if terminated:
            row = new_state // env.unwrapped.ncol
            col = new_state % env.unwrapped.ncol
            cell = env.unwrapped.desc[row][col].decode("utf-8")

            if cell == "H":
                print(
                    f"Episode {i_episode + 1} terminated after {step} steps: "
                    f"Fell into a hole at ({row}, {col})."
                )
            elif cell == "G":
                print(
                    f"Episode {i_episode + 1} terminated after {step} steps: "
                    f"Reached the goal at ({row}, {col})."
                )

            steps_total.append(step)
            rewards_total.append(reward)
            break

        if truncated:
            steps_total.append(step)
            rewards_total.append(reward)
            print(
                f"Episode {i_episode + 1} truncated after {step} steps: Time limit reached."
            )
            break


# Print statistics
print(f"Percent of episodes finished successfully: {sum(rewards_total) / NUM_EPISODES}")
print(
    "Percent of episodes finished successfully (last 100 episodes): "
    f"{sum(rewards_total[-100:]) / 100}"
)

print(f"Average number of steps: {sum(steps_total) / NUM_EPISODES:.2f}")
print(
    f"Average number of steps (last 100 episodes): {sum(steps_total[-100:]) / 100:.2f}"
)


# ----- Figure 1: Rewards -----
plt.figure(figsize=(12, 5))
plt.title("Rewards per Episode")
plt.bar(
    torch.arange(len(rewards_total)), rewards_total, alpha=0.6, color="green", width=5
)
plt.xlabel("Episode")
plt.ylabel("Reward")
reward_plot_path = os.path.join(OUTPUT_DIR, "rl04_rewards_per_episode.png")
plt.savefig(reward_plot_path, dpi=300)
# plt.show()
plt.close()
print(f"Saved: {reward_plot_path}")


# ----- Figure 2: Steps -----
plt.figure(figsize=(12, 5))
plt.title("Steps / Episode length")
plt.bar(torch.arange(len(steps_total)), steps_total, alpha=0.6, color="red", width=5)
plt.xlabel("Episode")
plt.ylabel("Steps")
steps_plot_path = os.path.join(OUTPUT_DIR, "rl04_steps_per_episode.png")
plt.savefig(steps_plot_path, dpi=300)
# plt.show()
plt.close()
