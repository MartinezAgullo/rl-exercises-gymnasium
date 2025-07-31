"""
Random policy for CartPole using Gymnasium.
Use RecordVideo to store the graphics.
"""

import os

import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo

env = gym.make("CartPole-v1", render_mode="rgb_array")

OUTPUT_DIR = "./docs/rl03"
env = RecordVideo(env, video_folder=OUTPUT_DIR, episode_trigger=lambda x: x % 100 == 0)

NUM_EPISODES = 1000
steps_total = []

for i_episode in range(NUM_EPISODES):

    state = env.reset()
    step = 0  # pylint: disable=invalid-name

    while True:

        step += 1

        action = env.action_space.sample()

        new_state, reward, terminated, truncated, info = env.step(action)

        # env.render()

        if terminated:
            cart_position = new_state[0]  # Cart Position allowed in ±2.4
            pole_angle = new_state[2]  # Pole Angle allowed in [-12, +12]º
            steps_total.append(step)
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
            print(f"Episode {i_episode+1} truncated: Max steps reached (time limit).")
            break


os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Average number of steps: {sum(steps_total)/NUM_EPISODES:.2f}")
plt.plot(steps_total)
plt.title("Average Number of Steps per Episode")
plt.xlabel("Episode")
steps_plot_path = os.path.join(OUTPUT_DIR, "rl03_avg_number_of_steps.png")
plt.savefig(steps_plot_path, dpi=300)
plt.close()


print(f"Plot saved to {steps_plot_path}")
