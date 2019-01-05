import gym
import numpy as np

from basicfetch import register
from reward_functions import reward_function_dict

register()
env = gym.make('FetchBasic-v0')
env.unwrapped.reward_func = reward_function_dict['cosanglelimit']['up']
action = np.zeros(env.action_space.shape)
np.set_printoptions(1)
while True:
    env.reset()
    done = False
    rewards = []
    while not done:
        action += np.random.normal(loc=0, scale=0.1, size=env.action_space.shape)
        action = np.clip(action, -1.0, 1.0)
        obs, reward, done, info = env.step(action)
        # print("Reward:", reward)
        # env.render(mode='human')
        rewards.append(reward)
    print(sum(rewards))