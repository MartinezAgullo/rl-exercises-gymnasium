"""
Random policy for CartPole using Gymnasium.


"""

import os
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import torch

env = gym.make("CartPole-v1")
# env = gym.make("CartPole-v1", render_mode="human")


OUTPUT_DIR = "./docs/rl12"


NUM_EPISODES = 1000
SEED_VALUE = 14
torch.manual_seed(SEED_VALUE)
random.seed(SEED_VALUE)

steps_total = []
rewards_total = []

for i_episode in range(NUM_EPISODES):

    state, _ = env.reset()
    step = 0  # pylint: disable=invalid-name

    while True:

        step += 1

        action = env.action_space.sample()

        new_state, reward, terminated, truncated, info = env.step(action)

        if terminated:
            cart_position = new_state[0]
            pole_angle = new_state[2]
            steps_total.append(step)
            rewards_total.append(reward)
            if abs(cart_position) > 2.4:
                print(
                    f"Episode {i_episode+1} terminated after {step} steps: "
                    f"Cart position exceeded limit ({cart_position:.2f})."
                )
            elif abs(pole_angle) > 0.2095:
                print(
                    f"Episode {i_episode+1} terminated after {step} steps: "
                    f"Pole angle exceeded limit ({pole_angle:.2f} rad)."
                )
            break

        if truncated:
            steps_total.append(step)
            rewards_total.append(reward)
            print(f"Episode {i_episode+1} truncated: Max steps reached (time limit).")
            break

        state = new_state


os.makedirs("./docs/rl12", exist_ok=True)

plt.figure(figsize=(12, 5))
plt.title("Rewards per Episode")
plt.bar(
    torch.arange(len(rewards_total)), rewards_total, alpha=0.6, color="green", width=5
)
plt.xlabel("Episode")
plt.ylabel("Reward")
reward_plot_path = os.path.join(OUTPUT_DIR, "rl12_rewards_per_episode_radnom_move.png")
plt.savefig(reward_plot_path, dpi=300)
plt.close()

plt.figure(figsize=(12, 5))
plt.title("Steps / Episode length")
plt.bar(torch.arange(len(steps_total)), steps_total, alpha=0.6, color="red", width=5)
plt.xlabel("Episode")
plt.ylabel("Steps")
steps_plot_path = os.path.join(OUTPUT_DIR, "rl12_steps_per_episode_radnom_move.png")
plt.savefig(steps_plot_path, dpi=300)
plt.close()

env.close()
env.env.close()
