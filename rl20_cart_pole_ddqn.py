"""
CartPole using Gymnasium
using Double DQN
"""

import math
import os
import random
import time
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn, optim

# --- Environment Setup --
env = gym.make("CartPole-v1")
NUM_EPISODES = 1000
REPORT_INTERVAL = 10

# --- Plotting directory --
OUTPUT_DIR = "./docs/rl20"
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.style.use("ggplot")

# --- HYPERPARAMETERS ---
HIDDEN_LAYER = 64
LEARNING_RATE = 0.02
NUM_EPISODES = 500
GAMMA = 0.99

REPLAY_MEM_SIZE = 50000
BATCH_SIZE = 32

UPDATE_TARGET_FREQUENCY = 100
CLIP_ERROR = False
DOUBLE_DQN = True

EPSILON_INITIAL = 0.9
EPSILON_FINAL = 0.02
EPSILON_DECAY = 500

# --- GPU ussage ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- Experience reply ---
class ExperienceReplay:
    """
    A fixed-size buffer that stores experience tuples and
    allows sampling for training reinforcement learning agents.
    """

    def __init__(self, capacity: int) -> None:
        """
        Initialize the replay memory.

        Args:
            capacity (int): Maximum number of experiences to store.
        """
        self.capacity = capacity
        self.memory = []  # type: ignore[var-annotated]
        self.position = 0

    # pylint: disable=too-many-arguments
    def push(
        self, state: Any, action: int, next_state: Any, reward: float, is_done: bool
    ) -> None:
        """
        Store a transition in memory.

        Args:
            state: Current state.
            action: Action taken.
            next_state: Resulting state.
            reward: Reward received.
            done: Whether the episode ended.
        """
        transition = (state, action, next_state, reward, is_done)

        if self.position >= len(self.memory):
            # new entry
            self.memory.append(transition)
        else:
            # override
            self.memory[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        """
        Sample a rndm batch of experiences from memory.

        Args:
            batch_size (int): Number of experiences to sample.

        Returns:
            A tuple of lists: (states, actions, next_states, rewards, dones)
        """
        # zip(*) :: Argument unpacking (*) to reverse the zip process
        return zip(*random.sample(self.memory, batch_size))

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return len(self.memory)


# --- NN and QNet Agent definitions ---
number_of_states = env.observation_space.shape[0]
number_of_outputs = env.action_space.n

print(
    "Building NN taking:\n"
    f"  - Input =  Number of states = {number_of_states}\n"
    f"  - Output =  Number of actions = {number_of_outputs}"
)


class NeuralNetwork(nn.Module):
    """A simple neural network with two layers."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(number_of_states, HIDDEN_LAYER)
        self.linear2 = nn.Linear(HIDDEN_LAYER, number_of_outputs)

        self.activation = nn.Tanh()

    def forward(self, x_in):
        """Forward pass of the network."""
        output1 = self.linear1(x_in)
        output1 = self.activation(output1)
        output2 = self.linear2(output1)
        return output2


class QNetAgent:
    """The Q-network agent."""

    def __init__(self):
        self.nn = NeuralNetwork().to(DEVICE)
        self.target_nn = NeuralNetwork().to(DEVICE)

        self.loss_func = nn.MSELoss()

        self.optimizer = optim.Adam(params=self.nn.parameters(), lr=LEARNING_RATE)

        self.update_target_counter = 0  # pylint: disable=invalid-name

    def select_action(self, agnt_state, agnt_epsilon):
        """
        Use the Agent's NN to select the best possible action.
        An adatative epsilon greedy is used for explroation.
        """
        random_for_egreedy = torch.rand(1)[0]

        if random_for_egreedy > agnt_epsilon:
            # Best possible movement
            with torch.no_grad():
                agnt_state = Tensor(agnt_state).to(DEVICE)
                action_from_nn = self.nn(agnt_state)
                action = torch.max(action_from_nn, 0)[1]
                action = action.item()
        else:
            # Explroation: Random movement
            action = env.action_space.sample()

        return action

    def optimize(self):
        """
        Use Experience Replay and Bellman equation to train the NN.
        """

        if len(memory) < BATCH_SIZE:
            return None

        # Sample batch from memory
        agnt_state, action, next_state, agnt_reward, is_done = memory.sample(BATCH_SIZE)

        # Convert to tensors
        agnt_state = torch.Tensor(np.array(agnt_state)).to(DEVICE)
        next_state = torch.Tensor(np.array(next_state)).to(DEVICE)
        agnt_reward = torch.Tensor(np.array(agnt_reward)).to(DEVICE)
        action = torch.LongTensor(np.array(action)).to(DEVICE)
        is_done = torch.Tensor(np.array(is_done)).to(DEVICE)

        if DOUBLE_DQN:
            # With indexes we get the idixes of which actions are the best
            new_state_indexes = self.nn(next_state).detach()
            max_new_state_indexes = torch.max(new_state_indexes, 1)[1]
            # The [1] retuns the indexes and the [0] the values

            # Calculate the values
            new_state_values = self.target_nn(next_state).detach()
            max_new_state_values = new_state_values.gather(
                1, max_new_state_indexes.unsqueeze(1)
            ).squeeze(1)

        else:
            # Using target NN for Q-values
            new_state_values = self.target_nn(next_state).detach()
            max_new_state_values = torch.max(new_state_values, 1)[0]

        target_q_value = agnt_reward + (1 - is_done) * GAMMA * max_new_state_values

        predicted_q_value = (
            self.nn(agnt_state).gather(1, action.unsqueeze(1)).squeeze(1)
        )

        loss = self.loss_func(predicted_q_value, target_q_value)

        self.optimizer.zero_grad()
        loss.backward()

        if CLIP_ERROR:
            for param in self.nn.parameters():
                param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        if self.update_target_counter % UPDATE_TARGET_FREQUENCY == 0:
            self.target_nn.load_state_dict(self.nn.state_dict())

        self.update_target_counter += 1

        return loss.item()


# --- Calcualte next epsilon ---
def calculate_epsilon_exp(steps_done: int) -> float:
    """
    Calculates epsilon using an exponential decay formula.
    """
    egreedy = EPSILON_FINAL + (EPSILON_INITIAL - EPSILON_FINAL) * math.exp(
        -1.0 * steps_done / EPSILON_DECAY
    )
    return egreedy


# --- Lists for plots ---
steps_total = []
rewards_total = []
epsilon_total = []
losses = []
total_steps = 0  # pylint: disable=invalid-name
solved_after = 0  # pylint: disable=invalid-name
solved = False  # pylint: disable=invalid-name
SCORE_TO_SOLVE = 195


start_time = time.time()

# --- Main loop ---
qnet_agent = QNetAgent()
memory = ExperienceReplay(REPLAY_MEM_SIZE)

epsilon = EPSILON_INITIAL  # pylint: disable=invalid-name

for i_episode in range(NUM_EPISODES):
    current_state, _ = env.reset()
    episode_reward_return = 0.0  # pylint: disable=invalid-name
    step = 0  # pylint: disable=invalid-name

    while True:

        step += 1
        total_steps += 1

        step_action = qnet_agent.select_action(current_state, epsilon)
        epsilon = calculate_epsilon_exp(total_steps)

        new_state, received_reward, terminated, truncated, _ = env.step(step_action)

        episode_reward_return += received_reward

        done = terminated or truncated

        memory.push(current_state, step_action, new_state, received_reward, done)
        step_loss = qnet_agent.optimize()

        current_state = new_state

        # --- Termination conditons (terminated/truncated) ---
        if done:
            cart_position = new_state[0]
            pole_angle = new_state[2]
            steps_total.append(step)
            rewards_total.append(episode_reward_return)
            epsilon_total.append(epsilon)
            if step_loss is not None:
                losses.append(step_loss)

            mean_reward_100 = sum(steps_total[-100:]) / 100

            if mean_reward_100 > SCORE_TO_SOLVE and not solved:
                print(f"SOLVED! After {i_episode} episodes")
                solved_after = i_episode
                solved = True  # pylint: disable=invalid-name

            if i_episode % REPORT_INTERVAL == 0:
                # pylint: disable=line-too-long
                print(
                    f"""
            ────────────────────────────────────────
            Episode {i_episode+1:,} — truncated (time limit)

            Avg. reward   : last {REPORT_INTERVAL}: {sum(steps_total[-REPORT_INTERVAL:])/REPORT_INTERVAL:.2f}
                            last 100: {mean_reward_100:.2f}
                            all: {sum(steps_total)/len(steps_total):.2f}
            Epsilon       : {epsilon:.3f}
            Frames total  : {total_steps:,}
            Elapsed time  : {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}
            ────────────────────────────────────────
            """.strip()
                )
                # pylint: enable=line-too-long

            break


# --- Plotting ---

# Plot for loss vs episode
plt.figure(figsize=(12, 5))
plt.plot(range(len(losses)), losses, alpha=0.6, color="blue")
plt.title("Loss vs. Episode")
plt.xlabel("Episode")
plt.ylabel("Loss")
loss_plot_path = os.path.join(OUTPUT_DIR, "rl20_loss_vs_episode.png")
plt.savefig(loss_plot_path, dpi=300)
plt.close()

# Plot for steps per episode
plt.figure(figsize=(12, 5))
plt.plot(range(len(steps_total)), steps_total, alpha=0.6, color="red")
plt.title("Steps per Episode")
plt.xlabel("Episode")
plt.ylabel("Steps")
steps_plot_path = os.path.join(OUTPUT_DIR, "rl20_steps_per_episode.png")
plt.savefig(steps_plot_path, dpi=300)
plt.close()

# Plot for rewards per episode
plt.figure(figsize=(12, 5))
plt.plot(range(len(rewards_total)), rewards_total, alpha=0.6, color="green")
plt.title("Rewards per Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
rewards_plot_path = os.path.join(OUTPUT_DIR, "rl20_rewards_per_episode.png")
plt.savefig(rewards_plot_path, dpi=300)
plt.close()

# Plot for epsilon per episode
plt.figure(figsize=(12, 5))
plt.plot(range(len(epsilon_total)), epsilon_total, alpha=0.6, color="purple")
plt.title("Epsilon per Episode")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
epsilon_plot_path = os.path.join(OUTPUT_DIR, "rl20_epsilon_per_episode.png")
plt.savefig(epsilon_plot_path, dpi=300)
plt.close()
