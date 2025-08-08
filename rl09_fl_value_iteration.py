"""
Stochastic FrozenLake.

Implementation of the value iteration
"""

import os

import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from gymnasium.envs.registration import register

# Stochastic frozen lake
register(
    id="FrozenLakeStochastic-v1",
    entry_point="gymnasium.envs.toy_text.frozen_lake:FrozenLakeEnv",
    kwargs={"map_name": "4x4", "is_slippery": True},
)
env = gym.make("FrozenLakeStochastic-v1")

OUTPUT_DIR = "./docs/rl09"
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.style.use("ggplot")


NUM_EPISODES = 1000
steps_total = []
rewards_total = []

number_of_states = env.observation_space.n
number_of_actions = env.action_space.n

# V values - size 16 (as number of fields[states] in our lake)
V = torch.zeros([number_of_states])


# Hyperparameters
GAMMA = 0.9
MAX_ITERATIONS = 1500


def next_step_evaluation(state, v_values):
    """
    Evaluates the best possible move from a given state using the Bellman expectation equation.

    Calculates the expected value for each possible action from a state by considering
    the rewards and discounted future values of all potential next states.

    Args:
        state: The current state (an integer index).
        v_values: A PyTorch tensor containing the current V-value estimates for all states.

    Returns:
        A tuple containing:
        - The maximum expected V-value.
        - The index of the best action.

    """
    v_temp = torch.zeros(number_of_actions)

    for action_possible in range(number_of_actions):

        for prob, next_state, reward_possible, _ in env.unwrapped.P[state][
            action_possible
        ]:
            # env.unwrapped.P[state][action_possible] holds the
            # transition probabilities and rewards for the environment.
            v_temp[action_possible] += prob * (
                reward_possible + GAMMA * v_values[next_state]
            )

    max_value, indice = torch.max(v_temp, 0)

    return max_value, indice


def value_iteration():
    """
    Performs V-value iteration to find the optimal value function (V-values).

    Iteratively updates the V-value of each state to be the maximum expected return
    until the V-values converge. This implementation uses a fixed number of
    iterations.

    Returns:
        A PyTorch tensor containing the final, optimal V-values for each state.
    """
    v_values = torch.zeros(number_of_states)

    for _ in range(MAX_ITERATIONS):
        # for each step we search for best possible move
        for state in range(number_of_states):
            max_value, _ = next_step_evaluation(state, v_values)
            v_values[state] = max_value.item()

    return v_values


# W0621: Redefining name 'v_policy' from outer scope
def build_policy(v_values):
    """
    Builds the optimal policy based on a the converged V-values.
    The policy are the instructions that state which are best moves from each single state.

    For each state, this build_policy() determines the action that yields the highest
    expected return according to the provided V-values.

    Args:
        v_values: A PyTorch tensor containing the optimal V-values for each state.

    Returns:
        A PyTorch tensor representing the optimal policy, where each index
        corresponds to a state and its value is the best action to take.

    """
    v_policy = torch.zeros(number_of_states)  # pylint: disable=redefined-outer-name

    for state in range(number_of_states):
        _, indice = next_step_evaluation(state, v_values)
        v_policy[state] = indice.item()

    return v_policy


# 2 main steps to build policy for our agent
v = value_iteration()
v_policy = build_policy(v)

# v(state): V-value for each state
# v_policy(state): shows the best action for each state.

for i_episode in range(NUM_EPISODES):

    current_state, _ = env.reset()
    step = 0  # pylint: disable=invalid-name

    while True:

        step += 1

        action = v_policy[current_state].item()

        new_state, reward, terminated, truncated, info = env.step(action)

        current_state = new_state

        if terminated or truncated:
            steps_total.append(step)
            rewards_total.append(reward)
            break

print(f"V-values: {v}")

# Print statistics
print(f"\nTraning completed after {NUM_EPISODES} episodes.")
print("Hyperparameters: \n" f"\t Discount factor: {GAMMA}\n")

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
reward_plot_path = os.path.join(OUTPUT_DIR, "rl09_rewards_per_episode.png")
plt.savefig(reward_plot_path, dpi=300)
plt.close()

plt.figure(figsize=(12, 5))
plt.title("Steps / Episode length")
plt.bar(torch.arange(len(steps_total)), steps_total, alpha=0.6, color="red", width=5)
plt.xlabel("Episode")
plt.ylabel("Steps")
steps_plot_path = os.path.join(OUTPUT_DIR, "rl09_steps_per_episode.png")
plt.savefig(steps_plot_path, dpi=300)
plt.close()
