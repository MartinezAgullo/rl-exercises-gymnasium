"""
Review fof NNs basics

Prediction of lienar behaviour
"""

import os

import matplotlib.pyplot as plt
import torch
from torch import nn, optim

# If GPU is to be used, define the device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = "./docs/rl11"
os.makedirs(OUTPUT_DIR, exist_ok=True)


WEIGHT = 2
BIAS = 0.3

x = torch.arange(100, dtype=torch.float).unsqueeze(1).to(DEVICE)
# x =  tensor([ 0.,  1.,  2.,  3.,  4., ..., 98., 99.])
# unsqueeze: Returns a new tensor with a dimension of
#            size one inserted at the specified position.

y = WEIGHT * x + BIAS

# --- HYPERPARAMETERS ---
LEARNING_RATE = 0.01
NUM_EPISODES = 1000

# List to store loss values for plotting
losses = []


class NeuralNetwork(nn.Module):
    """A simple neural network with one linear layer."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1, 1)

    def forward(self, x_in):
        """Forward pass of the network."""
        output = self.linear1(x_in)
        return output


model = NeuralNetwork().to(DEVICE)

loss_func = nn.MSELoss()
# loss_func = nn.SmoothL1Loss()

optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
# optimizer = optim.RMSprop(params=model.parameters(), lr=LEARNING_RATE)

for i_episode in range(NUM_EPISODES):

    predicted_value = model(x)

    loss = loss_func(predicted_value, y)

    losses.append(loss.item())  # For plotting

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i_episode % 50 == 0:
        print(f"Episode {i_episode}, loss {loss.item():.4f}")

# Plot for the regression line
plt.figure(figsize=(12, 5))
plt.plot(
    x.data.cpu().numpy(),
    y.data.cpu().numpy(),
    alpha=0.6,
    color="green",
    label="Original Data",
)
plt.plot(
    x.data.cpu().numpy(),
    predicted_value.data.cpu().numpy(),
    alpha=0.6,
    color="red",
    label="Predicted Value",
)
plt.legend()
plt.title("Neural Network Regression")
plt.xlabel("X")
plt.ylabel("Y")
regression_plot_path = os.path.join(OUTPUT_DIR, "rl11_regression.png")
plt.savefig(regression_plot_path, dpi=300)
plt.close()


# Plot for loss vs episode
plt.figure(figsize=(12, 5))
plt.plot(range(NUM_EPISODES), losses, alpha=0.6, color="blue")
plt.title("Loss vs. Episode")
plt.xlabel("Episode")
plt.ylabel("Loss")
loss_plot_path = os.path.join(OUTPUT_DIR, "rl11_loss_vs_episode.png")
plt.savefig(loss_plot_path, dpi=300)
plt.close()
