import readline

import gym
import numpy as np

from basicfetch import register
from reward_functions import reward_function_dict

# noinspection PyStatementEffect
readline

register()
env = gym.make('FetchBasic-v0').unwrapped  # unwrap past TimeLimit
env.unwrapped.reward_func = reward_function_dict['goal']['up']
np.set_printoptions(2)
print(env.get_ctrl_names())
while True:
    command = input()
    try:
        action = np.fromstring(command, sep=' ')
        if len(action) != env.action_space.shape[0] + 1:
            raise Exception("Wrong length")
    except Exception as e:
        print(e)
        continue
    action, reps = action[:-1], int(action[-1])
    print(action, reps)

    for _ in range(reps):
        obs, reward, done, info = env.step(action)
        print(reward)
        env.render(mode='human')
        print('Grippper position:', env.sim.data.get_site_xpos('grip'))
        print('Reward:', reward)

        # left side of table: x 1.073 to 1.435, y 0.4
        # right side of table: x 1.044 to 1.424, y 1.09
        # back side of table: x 1.023, y 0.424 to 1.101
        # front side of table: x 1.424, y 0.402 to 1.093
        pos = env.unwrapped.sim.data.get_site_xpos('grip')
        e = 0.05
        if pos[0] > 1.044 - e and pos[0] < 1.435 + e and pos[1] > 0.4 - e and pos[1] < 0.4 + e:
            print('left')
        if pos[0] > 1.044 - e and pos[0] < 1.435 + e and pos[1] > 1.09 - e and pos[1] < 1.09 + e:
            print('right')
        if pos[0] > 1.023 - e and pos[0] < 1.023 + e and pos[1] > 0.402 - e and pos[1] < 1.101 + e:
            print('back')
        if pos[0] > 1.5 - e and pos[0] < 1.5 + e and pos[1] > 0.402 - e and pos[1] < 1.101 + e:
            print('front')
