# RL Exercises with Gymnasium

A collection of Reinforcement Learning exercises based on a Udemy course, updated and adapted to use the modern Gymnasium API.
Includes practical implementations of classic control environments and examples for experimenting with RL algorithms.

## Exercises

### rl01_cartpole_random.py
[Cart Pole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) environment with random actions to test the Gymnasium API.

### rl02_frozen_lake_random.py
[Frozen Lake](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) environment with random actions.

### rl03_cartpole_videos.py
The same [Cart Pole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) as in [rl01_cartpole_random.py](https://github.com/MartinezAgullo/rl-exercises-gymnasium/blob/main/rl01_cartpole_random.py), but with video recording.

### rl04_frozen_lake_stochastic_deterministic.py
Includes configuration to switch between stochastic and deterministic versions of the environment.

### rl05_frozen_lake_deterministic_bellman.py
Implementation of the Bellman equation in the Frozen Lake exercise

$$
Q(s, a) = r + \gamma \cdot \max_{a'} Q(s', a')
$$

- ***Q(s,a)***: The **action-value function** or Q-value, representing the expected cumulative reward of taking action *a* in state *s*, and then following the optimal policy.

- ***$\gamma$***: The **discount factor** ( 0≤γ≤10 \leq \gamma \leq 1 ), which determines the importance of future rewards.

    -   γ=0\gamma = 0: Focus only on immediate rewards.

    -   γ≈1\gamma \approx 1: Future rewards are considered almost as important as immediate rewards.

- ***r***: The **immediate reward** received after performing action *a* in state *s*.

<div style="display: flex; justify-content: center; align-items: flex-start; gap: 20px;">
    <figure style="margin: 0;">
        <img src="https://github.com/MartinezAgullo/rl-exercises-gymnasium/blob/main/docs/rl05/rl05_steps_per_episode_before_learning.png" alt="Steps per episode before learning" style="width: 100%; max-width: 400px; display: block;">
        <figcaption style="text-align: center; font-size: 0.9em; color: #555;">
            Figure 1: Steps per episode before learning.
        </figcaption>
    </figure>
    <figure style="margin: 0;">
        <img src="https://github.com/MartinezAgullo/rl-exercises-gymnasium/blob/main/docs/rl05/rl05_steps_per_episode.png" alt="Steps per episode after learning" style="width: 100%; max-width: 400px; display: block;">
        <figcaption style="text-align: center; font-size: 0.9em; color: #555;">
            Figure 2: Steps per episode after learning.
        </figcaption>
    </figure>
</div>
