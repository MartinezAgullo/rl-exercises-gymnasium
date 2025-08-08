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

- ***Q(s,a)***: The **action-value function** or Q-value, representing the expected cumulative reward of taking action *a* in state *s*, and then following the optimal policy. The parameters *s'*, *a'* are the next state and action respectively.

- ***γ***: The **discount factor** ( 0≤γ≤1), which determines the importance of future rewards.

    -   γ=0: Focus only on immediate rewards.

    -   γ≈1: Future rewards are considered almost as important as immediate rewards.

- ***r***: The **immediate reward** received after performing action *a* in state *s*.

Note how, now that we are learning, we became efficient when it comes to decide the path to the goal.
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


The learning is proved successful:
```console
Percent of episodes finished successfully (last 200 episodes): 100.0%
```

### rl06_frozen_lake_stochastic_bellman.py
Applying the simple, deterministic Bellman equation you've used for the Frozen Lake problem, which is $Q(s,a)=r+γ⋅max_{a'} Q(s', a')$  doesn't work in a stochastic environment because it incorrectly assumes that an action will always lead to a single, predictable next state. Instead, there's a probability distribution over possible next states.

```console
Percent of episodes finished successfully (last 200 episodes): 2.5%
```



### rl07_frozen_lake_stochastic_q_learning.py
To properly account for this randomness in what is *s'* going to be after doing *a*, the Bellman equation must be modified to use an expectation.  For this purpose the temporal difference (TD) or TD error is used:

$$
TD(s, a) = [r + \gamma \cdot \max_{a'} Q(s', a')] - [Q(s, a)]
$$

and the new Q-value for the *(s, a)* pair at a time *t* is:

$$
Q_{t}(s, a) = Q_{t-1}(s, a) + α \cdot TD
$$

and hence

$$
Q(s, a) = Q(s, a) + α \cdot [r + \gamma \cdot \max_{a'} Q(s', a') - Q(s, a)]
$$

rewritten:

$$
Q(s, a) = (1 - α)Q(s, a) + α \cdot[r + \gamma \cdot \max_{a'} Q(s', a')]
$$

where:
- *** $Q_{t-1}(s, a)$ ***: Old Q-value for the current *s, a* pair.

- ***α***: Learning rate (0<α≤1). Controls balance between learning from new experiences and using experience.
    - α = 0: The agent never learns and its Q-values are never updated.
            Therefore:
            $Q(s, a) = Q(s, a)$

    - α = 1: The agent only takes into account the most recent experience and the old Q-value is completely replaced by the new one.
            Therefore:
            $Q(s, a) = r + \gamma \cdot \max_{a'} Q(s', a')$ (Bellman Equation)

```console
Traning completed after 10000 episodes.
Hyperparameters:
         Discount factor: 1
         Learning rate: 0.5
Percent of episodes finished successfully: 45.61%
Percent of episodes finished successfully (last 100 episodes): 87.0%
```
### rl08_fl_e_greedy_exploration.py
Include an ɛ-greedy exploration strategy.
    <figure style="margin: 0;">
        <img src="https://github.com/MartinezAgullo/rl-exercises-gymnasium/blob/main/docs/rl08/rl08_steps_per_episode.png" alt="Steps per episode before learning" style="width: 100%; max-width: 400px; display: block;">
        <figcaption style="text-align: center; font-size: 0.9em; color: #555;">
            Figure 3: Steps per episode with exploration strategy. Note how, while optimal is known, from time a different path is explored.
        </figcaption>
    </figure>

```console
Traning completed after 1000 episodes.
Hyperparameters:
         Discount factor: 0.9
         Learning rate: 1 (becuase deterministic)
         Epsilon for exploration: 0.1
Percent of episodes finished successfully: 85.5%
Percent of episodes finished successfully (last 100 episodes): 91.0%
Average number of steps: 6.59
Average number of steps (last 100 episodes): 6.47
```
