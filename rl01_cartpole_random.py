"""
Random policy for CartPole using Gymnasium.

This script runs a random policy on the CartPole-v1 environment and plots the
number of steps per episode.
"""

import os

import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1", render_mode="human")
NUM_EPISODES = 1000
steps_total = []

for i_episode in range(NUM_EPISODES):

    state = env.reset()
    step = 0  # pylint: disable=invalid-name

    # for step in range(100):
    while True:

        step += 1

        # Action space:
        #      0: Push cart to the left
        #      1: Push cart to the right
        action = env.action_space.sample()

        new_state, reward, terminated, truncated, info = env.step(action)

        # print(new_state)
        # print(info)

        env.render()

        if terminated:
            steps_total.append(step)
            print(f"Episode {i_episode+1} terminated: Pole or cart limit exceeded.")
            break
        if truncated:
            steps_total.append(step)
            print(f"Episode {i_episode+1} truncated: Max steps reached (time limit).")
            break


os.makedirs("./docs/rl01", exist_ok=True)

print(f"Average number of steps: {sum(steps_total)/NUM_EPISODES:.2f}")
plt.plot(steps_total)
plt.title("Average Number of Steps per Episode")
plt.xlabel("Episode")
plt.savefig("./docs/rl01/rl01_avg_number_of_steps.png", dpi=300)
plt.close()
plt.show()

print("Plot saved to ./docs/rl01/rl01_avg_number_of_steps.png")
