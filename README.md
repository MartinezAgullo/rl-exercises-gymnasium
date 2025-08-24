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
- $Q_{t-1}(s, a)$: Old Q-value for the current *s, a* pair.

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
### rl08_0_fl_e_greedy_exploration.py
Include an ɛ-greedy exploration strategy.
    <figure style="margin: 0;">
        <img src="https://github.com/MartinezAgullo/rl-exercises-gymnasium/blob/main/docs/rl08/rl08_steps_per_episode.png" alt="Steps per episode before learning" style="width: 100%; max-width: 400px; display: block;">
        <figcaption style="text-align: center; font-size: 0.9em; color: #555;">
            Figure 3: Steps per episode with exploration strategy. Note how, while optimal is known, from time to time a different path is explored.
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

### rl08_1_fl_e_greedy_exploration.py
The ɛ greedy is varying through the episodes so that the exploration is favoured in the first episodes.
    <figure style="margin: 0;">
        <img src="https://github.com/MartinezAgullo/rl-exercises-gymnasium/blob/main/docs/rl08/rl08b_steps_per_episode_with_decay.png" alt="Steps per episode before learning" style="width: 100%; max-width: 400px; display: block;">
        <figcaption style="text-align: center; font-size: 0.9em; color: #555;">
            Figure 4: Steps per episode with exploration strategy using adaptative ɛ.
        </figcaption>
    </figure>
    <figure style="margin: 0;">
        <img src="https://github.com/MartinezAgullo/rl-exercises-gymnasium/blob/main/docs/rl08/rl08b_epsilon_per_episode_with_decay.png" alt="Steps per episode after learning" style="width: 100%; max-width: 400px; display: block;">
        <figcaption style="text-align: center; font-size: 0.9em; color: #555;">
            Figure 5: ɛ per episode using adaptative ɛ.
        </figcaption>
    </figure>

### rl09_fl_value_iteration.py
As an alternative, instead of Q-values, the V-values can be used.

The V-value at a state *s* for a given policy $\pi$ is:

$$
V(s) := \sum_{s'} P_{\pi(s)}(s, s') (R_{\pi(s)}(s, s') + \gamma V(s'))
$$

It calculates the expected return from state *s* by summing the rewards and discounted future values of all possible next states (*s′*) weighted by their probabilities $P_{\pi(s)}(s, s')$ under policy $\pi$. Here $R_{\pi(s)}(s, s')$ is the inmediate reward received by the agent when trasitoning from *s* to *s'*. The $P_{\pi(s)}(s, s')$ represents the probability of transitioning from state *s* to state *s′*, given that the agent is in state *s* and takes the action specified by its policy $\pi$.



The optimal policy $\pi(s)$ is given by:

$$
\pi(s) := \arg \max_{a} [ \sum_{s'} P_{a}(s, s') [R_{a}(s, s') + \gamma V(s')] ]
$$

It states that the best action to take in state *s* is the one that maximizes the expected value of the resulting state, essentially choosing the action that leads to the highest potential reward.

The value iteration algorithm is an iterative method to find the optimal value function by repeatedly updating the value of each  *V(s)* to equal the maximum expected value achievable from that state. This process continues until the values converge to the optimal values.

$$
V_{i+1}(s) := \max_{a} [ \sum_{s'} P_{a}(s, s') (R_{a}(s, s') + \gamma V_i(s')) ]
$$

Since a (optimal) policy is found with this method, there is no learning after taking steps.
    <figure style="margin: 0;">
        <img src="https://github.com/MartinezAgullo/rl-exercises-gymnasium/blob/main/docs/rl09/rl09_steps_per_episode.png" alt="Steps per episode before learning" style="width: 100%; max-width: 400px; display: block;">
        <figcaption style="text-align: center; font-size: 0.9em; color: #555;">
            Figure 6: Steps per episode after finding the optimal policy with value interation.
        </figcaption>
    </figure>

### rl10_taxi.py
Apply the adaptative ɛ greedy strategy in the [Taxi](https://gymnasium.farama.org/environments/toy_text/taxi/) environment. In this scenario, negative rewards are possible:
    <figure style="margin: 0;">
        <img src="https://github.com/MartinezAgullo/rl-exercises-gymnasium/blob/main/docs/rl10/rl10_rewards_per_episode.png" alt="Steps per episode before learning" style="width: 100%; max-width: 400px; display: block;">
        <figcaption style="text-align: center; font-size: 0.9em; color: #555;">
            Figure 7: Reward per episode.
        </figcaption>
    </figure>

<div style="text-align: center;">
<a href="https://github.com/MartinezAgullo/rl-exercises-gymnasium/blob/main/docs/rl10/rl10_1_taxi_stochastic_episode_200.mp4">
    <img src="https://github.com/MartinezAgullo/rl-exercises-gymnasium/blob/main/docs/rl10/rl10_video_thumbnail.png" alt="Video of Taxi-v3 environment" style="width: 70%; max-width: 400px; display: block;">
    <br>
    <figcaption style="text-align: center; font-size: 0.9em; color: #555;">
            Figure 8: Animation of our taxi agent.
    </figcaption>
</a>
</div>

<!-- ```console
Traning completed after 300 episodes.
Hyperparameters:
         Discount factor: 0.9
         Learning rate: 1 (becuase deterministic)
         Epsilon for exploration:
                 Initial 0.9
                 Final 0.001
                 Decay multipliyer 0.999

Percent of episodes finished successfully: 87.66%
Percent of episodes finished successfully (last 100 episodes): 100.0%
Average number of steps: 63.25
Average number of steps (last 100 episodes): 19.71
``` -->


### rl11_nn_review.py
**Scaling Up Reinforcement Learning**
Q-tables are inefficient for environments with a very large number of states, a situation known as the "curse of dimensionality." In these scenarios, it's highly probable the agent will encounter states it has never seen before, making a complete table lookup impossible.-

To address this, we use function approximation:

- **Generalize**: Instead of relying on lookups on a complete Q-table, we use a single, generalized function to estimate the Q-value for any given state. This function can provide a value even for states the agent has never visited.

- **Function Approximation**: This approach allows the agent to learn the underlying rules of the environment and apply them to similar situations, whether they've been seen before or not. By identifying patterns, the function can make educated guesses about the value of unseen states.

NNs are an excellent solution for function approximation. They can learn to map states to their corresponding Q-values, allowing for effective reinforcement learning in complex, large-scale environments

### rl12_cart_pole_random_new.py
Same as [rl01_cartpole_random.py](https://github.com/MartinezAgullo/rl-exercises-gymnasium/blob/main/rl01_cartpole_random.py)

### rl13_cart_pole_egreedy.py
Implement the addaptative $\epsilon$-greedy strategy in the [rl12_cart_pole_random_new.py](https://github.com/MartinezAgullo/rl-exercises-gymnasium/blob/main/rl12_cart_pole_random_new.py) environment using:

- Exponential decay formula: Using a $\tau$ decay rate

$$ \epsilon = \epsilon_{final} + (\epsilon_{initial} - \epsilon_{final}) \cdot e^{-\tfrac{steps_{done}}{\tau}} $$

- Multiplicative decay: Using a decay $\gamma$ rate

$$ \epsilon_{t+1} = \epsilon_t \cdot \gamma $$

### rl14_cart_pole_nn.py
[CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/)-v1 environment using a Q-Learning algorithm with a single-layer neural network and an addaptative $\epsilon$-greedy strategy.

### rl15_cart_pole_nn_log.py
Improved logging.

### rl16_cart_pole_nn_2_layers.py
Adding additional hiden layer like in [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602). As expected, the [rewards now](https://github.com/MartinezAgullo/rl-exercises-gymnasium/blob/main/docs/rl16/rl16_steps_per_episode.png) are much better than the [previous](https://github.com/MartinezAgullo/rl-exercises-gymnasium/blob/main/docs/rl15/rl15_steps_per_episode.png) ones. Nevertheless, some inestabilies are observed. These inestabilies could be done to the fact that since the agent is moving forward, the next state will be very similar to the current one. Training a NN on a stream of highly correlated data can lead to instability and poor performance, as the network gets stuck in local minima. This will be latter soved in [rl18_cart_pole_experience_replay.py](https://github.com/MartinezAgullo/rl-exercises-gymnasium/blob/main/rl18_cart_pole_experience_replay.py).

### rl17_cart_pole_nn_2_layers.py
The same [Cart Pole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) as in [rl16_cart_pole_nn_2_layers.py](https://github.com/MartinezAgullo/rl-exercises-gymnasium/blob/main/rl16_cart_pole_nn_2_layers.py) but with better hyperparameters.

### rl18_cart_pole_experience_replay.py
To alleviate the problems of correlated data and non-stationary distributions, we use an ([experience replay mechanism](https://apps.dtic.mil/sti/tr/pdf/ADA261434.pdf)) which randomly samples previous transitions, and thereby smooths the training distribution over many past behaviors.  Instead of training on experiences in the order they occur, the agent stores them in a data structure called a replay buffer.



### rl19_cart_pole_target_NN.py
The Deep Q-Network (DQN) uses a separate, identical NN  called the target network to calculate the target Q-values. This target network's weights are periodically synchronized with the main Learning NN weights but are held fixed during the training of the main network.

The loss function ($L_i$) for the DQN traning is:

$$ L_i(\theta_i) = \mathbb{E}_{(s,a,r,s') \sim U(D)} \left[ \left( r + \gamma \max_{a'} Q(s',a';\theta_i^-) - Q(s,a;\theta_i) \right)^2 \right] $$

Where $r + \gamma \max_{a'} Q(s',a';\theta_i^-)$ is the target Q value and $Q(s,a;\theta_i)$ is the current Q-value estimate for the action $a$ taken in state $s$. The $\theta$ are NN weights. Remember that the agent is the Learning NN.

###

[DRL with Double Q-learning](https://arxiv.org/abs/1509.06461) (DDQN) is an improvement on the standard DQN algorithm, designed to reduce overestimation bias in the predicted Q-values.
<!-- It achieves this by decoupling the selection of the best action from the evaluation of its value. -->
The standard DQN target value is:
$$Y_{\text{DQN}} = r + \gamma \cdot \max_{a'} Q(s', a'; \theta)$$

and for the DDQN is:
$$Y_{\text{DoubleDQN}} = r + \gamma \cdot Q(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-)$$
