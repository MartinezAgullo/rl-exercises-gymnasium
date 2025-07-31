import gymnasium as gym

import matplotlib.pyplot as plt

env = gym.make('CartPole-v1', render_mode="human")
num_episodes = 1000
steps_total = []

for i_episode in range(num_episodes):
    
    state = env.reset()
    step = 0

    #for step in range(100):
    while True:
        
        step += 1
        
        # The action space is:
        #      0: Push cart to the left
        #      1: Push cart to the right
        action = env.action_space.sample()
        
        new_state, reward, terminated, truncated, info = env.step(action)
        
        #print(new_state)
        #print(info)
        
        env.render()
        
        if terminated:
            steps_total.append(step)
            print(f"Episode {i_episode+1} terminated: Pole or cart limit exceeded.")
            break
        if truncated:
            steps_total.append(step)
            print(f"Episode {i_episode+1} truncated: Max steps reached (time limit).")
            break

            
        
        
print(f"Average number of steps: {sum(steps_total)/num_episodes:.2f}")
plt.plot(steps_total)
plt.show()