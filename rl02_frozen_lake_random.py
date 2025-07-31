"""
Random policy for FrozenLake using Gymnasium.
"""

import os

import gymnasium as gym
import matplotlib.pyplot as plt
import torch

# env = gym.make('FrozenLake-v1', render_mode="human", desc=None, map_name="4x4", is_slippery=True)
env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=True)

OUTPUT_DIR = "./docs/rl02"
NUM_EPISODES = 1000
steps_total = []
rewards_total = []

for i_episode in range(NUM_EPISODES):

    state = env.reset()

    step = 0  # pylint: disable=invalid-name

    while True:

        step += 1

        # Action space:
        #      0: Move left
        #      1: Move adown
        #      2: Move right
        #      3: Move up
        action = env.action_space.sample()

        new_state, reward, terminated, truncated, info = env.step(action)

        # Tile letters:
        #       “S” for Start tile
        #       “G” for Goal tile
        #       “F” for frozen tile
        #       “H” for a tile with a hole

        # print(new_state)
        # print(info)

        # time.sleep(0.4)

        # env.render()

        if terminated:
            # Get the character of the new position
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
    f"Percent of episodes finished successfully (last 100 episodes): {sum(rewards_total[-100:]) / 100}"
)

print(f"Average number of steps: {sum(steps_total) / NUM_EPISODES:.2f}")
print(
    f"Average number of steps (last 100 episodes): {sum(steps_total[-100:]) / 100:.2f}"
)


# Ensure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----- Figure 1: Rewards -----
plt.figure(figsize=(12, 5))
plt.title("Rewards per Episode")
plt.bar(
    torch.arange(len(rewards_total)), rewards_total, alpha=0.6, color="green", width=5
)
plt.xlabel("Episode")
plt.ylabel("Reward")
reward_plot_path = os.path.join(OUTPUT_DIR, "rl02_rewards_per_episode.png")
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
steps_plot_path = os.path.join(OUTPUT_DIR, "rl02_steps_per_episode.png")
plt.savefig(steps_plot_path, dpi=300)
# plt.show()
plt.close()
