"""
CartPole using Gymnasium using two epsilon-greedy decay strategies.

The first uses an exponential decay based on total steps, while the
second uses a multiplicative decay per episode.

[Not learning implemented, just the egreedy strategies]

"""

import math
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import torch

# --- Environment Setup --
env = gym.make("CartPole-v1")
NUM_EPISODES = 1000

# --- Plotting directory --
OUTPUT_DIR = "./docs/rl13"
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.style.use("ggplot")

# --- Epsilon-Greedy Strategy 1 (Exponential Decay) ---
EGREEDY_INITIAL_EXP = 0.9
EGREEDY_FINAL_EXP = 0.02
EGREEDY_DECAY_EXP = 500


def calculate_epsilon_exp(steps_done: int) -> float:
    """
    Calculates epsilon using an exponential decay formula.

    Args:
        steps_done: The total number of steps taken in the environment.

    Returns:
        The calculated epsilon value.
    """
    epsilon = EGREEDY_FINAL_EXP + (EGREEDY_INITIAL_EXP - EGREEDY_FINAL_EXP) * math.exp(
        -1.0 * steps_done / EGREEDY_DECAY_EXP
    )
    return epsilon


# --- Epsilon-Greedy Strategy 2 (Multiplicative Decay) ---
egreedy_mult = EGREEDY_INITIAL_EXP  # pylint: disable=invalid-name
EGREEDY_FINAL_MULT = EGREEDY_FINAL_EXP
EGREEDY_DECAY_MULT = 0.999


# --- Main Loop and Data Collection ---
steps_total = []
rewards_total = []
egreedy_exp_total = []
egreedy_mult_total = []
total_steps = 0  # pylint: disable=invalid-name

for i_episode in range(NUM_EPISODES):

    state, _ = env.reset()
    step = 0  # pylint: disable=invalid-name

    while True:

        step += 1
        total_steps += 1

        # Epsilon 1: Calculate and store epsilon for exponential decay
        epsilon_exp = calculate_epsilon_exp(total_steps)
        egreedy_exp_total.append(epsilon_exp)

        # Epsilon 2: Update and store epsilon for multiplicative decay
        if egreedy_mult > EGREEDY_FINAL_MULT:
            egreedy_mult *= EGREEDY_DECAY_MULT
            egreedy_mult_total.append(egreedy_mult)
        else:
            egreedy_mult_total.append(egreedy_mult)

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


# --- Plotting Results ---
plt.figure(figsize=(12, 5))
plt.title("Rewards per Episode")
plt.bar(
    torch.arange(len(rewards_total)), rewards_total, alpha=0.6, color="green", width=5
)
plt.xlabel("Episode")
plt.ylabel("Reward")
reward_plot_path = os.path.join(OUTPUT_DIR, "rl13_rewards_per_episode.png")
plt.savefig(reward_plot_path, dpi=300)
plt.close()

plt.figure(figsize=(12, 5))
plt.title("Steps / Episode length")
plt.bar(torch.arange(len(steps_total)), steps_total, alpha=0.6, color="red", width=5)
plt.xlabel("Episode")
plt.ylabel("Steps")
steps_plot_path = os.path.join(OUTPUT_DIR, "rl13_steps_per_episod.png")
plt.savefig(steps_plot_path, dpi=300)
plt.close()

plt.figure(figsize=(12, 5))
plt.title("Epsilon-Greedy Decay (Exponential)")
plt.bar(
    torch.arange(len(egreedy_exp_total)),
    egreedy_exp_total,
    alpha=0.6,
    color="yellow",
    width=5,
)
plt.xlabel("Steps")
plt.ylabel("Epsilon Value")
exp_plot_path = os.path.join(OUTPUT_DIR, "rl13_exp_decay.png")
plt.savefig(exp_plot_path, dpi=300)
plt.close()

plt.figure(figsize=(12, 5))
plt.title("Epsilon-Greedy Decay (Multiplicative)")
plt.bar(
    torch.arange(len(egreedy_mult_total)),
    egreedy_mult_total,
    alpha=0.6,
    color="blue",
    width=5,
)
plt.xlabel("Steps")
plt.ylabel("Epsilon Value")
mult_plot_path = os.path.join(OUTPUT_DIR, "rl13_mult_decay.png")
plt.savefig(mult_plot_path, dpi=300)
plt.close()

env.close()
env.env.close()
