"""
CartPole using Gymnasium
"""

import math
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn, optim

# --- Environment Setup --
env = gym.make("CartPole-v1")
NUM_EPISODES = 1000

# --- Plotting directory --
OUTPUT_DIR = "./docs/rl14"
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.style.use("ggplot")

# --- HYPERPARAMETERS ---
LEARNING_RATE = 0.01
NUM_EPISODES = 1000
GAMMA = 0.85

EPSILON_INITIAL = 0.9
EPSILON_FINAL = 0.02
EPSILON_DECAY = 500

# --- GPU ussage ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- NN and QNet Agent definitions ---
number_of_states = env.observation_space.shape[0]
number_of_outputs = env.action_space.n

print(
    "Building NN taking:\n"
    f"  - Input =  Number of states = {number_of_states}\n"
    f"  - Output =  Number of actions = {number_of_states}"
)


class NeuralNetwork(nn.Module):
    """A simple neural network with one linear layer."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(number_of_states, number_of_outputs)

    def forward(self, x_in):
        """Forward pass of the network."""
        output = self.linear1(x_in)
        return output


class QNetAgent:
    """The Q-network agent."""

    def __init__(self):
        self.nn = NeuralNetwork().to(DEVICE)

        self.loss_func = nn.MSELoss()
        # self.loss_func = nn.SmoothL1Loss()

        self.optimizer = optim.Adam(params=self.nn.parameters(), lr=LEARNING_RATE)
        # self.optimizer = optim.RMSprop(params=self.nn.parameters(), lr=LEARNING_RATE)

    def select_action(self, agnt_state, agnt_epsilon):
        """
        Use the Agent's NN to select the best possible action.
        An adatative epsilon greedy is used for explroation.
        """
        random_for_egreedy = torch.rand(1)[0]

        if random_for_egreedy > agnt_epsilon:
            # Best possible movement
            with torch.no_grad():
                # no_grad :: Won't perform backpropagation
                agnt_state = Tensor(agnt_state).to(DEVICE)
                action_from_nn = self.nn(agnt_state)
                action = torch.max(action_from_nn, 0)[1]
                action = action.item()
        else:
            # Explroation: Random movement
            action = env.action_space.sample()

        return action

    def optimize(
        self, agnt_state, action, next_state, agnt_reward, is_done
    ):  # pylint: disable=too-many-arguments
        """
        Implements Bellman equation and uses it to tune the NN

        target_value :: Uses Bellamn equation to determine best
                        possible action. Similar to real Q-value.

        predicted_value :: Stimation by the NN of the Q-Value.
                           What the agent thinks is the best
                           action.

        """
        agnt_state = Tensor(agnt_state).to(DEVICE)
        next_state = Tensor(next_state).to(DEVICE)
        agnt_reward = Tensor([agnt_reward]).to(DEVICE)

        # Target value behaves like Q-value
        if is_done:
            # If done, deterministic Bellman equation is
            # just reward because there is not next state
            target_value = agnt_reward
        else:
            next_state_values = self.nn(next_state).detach()
            # detach() means that no gradients will be calculated
            # to update NN parameters
            max_next_state_values = torch.max(next_state_values)

            # Bellman for determinsitic
            target_value = agnt_reward + GAMMA * max_next_state_values

        # predicted_value <-- Agent's NN current stimate of Q-value
        predicted_value = self.nn(agnt_state)[action]

        # Compare predicted (agent's) and target (Bellman)
        loss = self.loss_func(predicted_value, target_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
# total_steps controls the epsilon greedy

# --- Main loop ---
qnet_agent = QNetAgent()

epsilon = EPSILON_INITIAL  # pylint: disable=invalid-name

for i_episode in range(NUM_EPISODES):
    state, _ = env.reset()
    step = 0  # pylint: disable=invalid-name

    while True:

        step += 1
        total_steps += 1

        step_action = qnet_agent.select_action(state, epsilon)
        epsilon = calculate_epsilon_exp(total_steps)

        new_state, reward, terminated, truncated, _ = env.step(step_action)

        done = terminated or truncated

        # qnet_agent.optimize(state, new_state, reward, done)
        step_loss = qnet_agent.optimize(state, step_action, new_state, reward, done)

        state = new_state

        # --- Termination conditons (terminated/truncated) ---
        if terminated:
            cart_position = new_state[0]
            pole_angle = new_state[2]
            steps_total.append(step)
            rewards_total.append(reward)
            epsilon_total.append(epsilon)
            losses.append(step_loss)

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
            cart_position = new_state[0]
            pole_angle = new_state[2]
            steps_total.append(step)
            rewards_total.append(reward)
            epsilon_total.append(epsilon)
            losses.append(step_loss)
            print(f"Episode {i_episode+1} truncated: Max steps reached (time limit).")
            break


# --- Plotting ---

# Plot for loss vs episode
plt.figure(figsize=(12, 5))
plt.plot(range(len(losses)), losses, alpha=0.6, color="blue")
plt.title("Loss vs. Episode")
plt.xlabel("Episode")
plt.ylabel("Loss")
loss_plot_path = os.path.join(OUTPUT_DIR, "rl14_loss_vs_episode.png")
plt.savefig(loss_plot_path, dpi=300)
plt.close()

# Plot for steps per episode
plt.figure(figsize=(12, 5))
plt.plot(range(len(steps_total)), steps_total, alpha=0.6, color="red")
plt.title("Steps per Episode")
plt.xlabel("Episode")
plt.ylabel("Steps")
steps_plot_path = os.path.join(OUTPUT_DIR, "rl14_steps_per_episode.png")
plt.savefig(steps_plot_path, dpi=300)
plt.close()

# Plot for rewards per episode
plt.figure(figsize=(12, 5))
plt.plot(range(len(rewards_total)), rewards_total, alpha=0.6, color="green")
plt.title("Rewards per Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
rewards_plot_path = os.path.join(OUTPUT_DIR, "rl14_rewards_per_episode.png")
plt.savefig(rewards_plot_path, dpi=300)
plt.close()

# Plot for epsilon per episode
plt.figure(figsize=(12, 5))
plt.plot(range(len(epsilon_total)), epsilon_total, alpha=0.6, color="purple")
plt.title("Epsilon per Episode")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
epsilon_plot_path = os.path.join(OUTPUT_DIR, "rl14_epsilon_per_episode.png")
plt.savefig(epsilon_plot_path, dpi=300)
plt.close()
