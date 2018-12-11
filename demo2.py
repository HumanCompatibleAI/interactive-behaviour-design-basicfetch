import select
import sys

import gym
import numpy as np
from gym.envs.robotics.rotations import quat2euler

from basicfetch import register


def read_timeout(timeout=(1 / 30)):
    i, o, e = select.select([sys.stdin], [], [], timeout)
    if i:
        return sys.stdin.readline().strip()
    else:
        return None

register()
env = gym.make('FetchBasicLevelDense-v0').unwrapped  # unwrap past TimeLimit
np.set_printoptions(1)
print(env.get_ctrl_names())
while True:
    command = read_timeout()
    action = np.zeros(env.action_space.shape)
    if command is None:
        obs, reward, done, info = env.step(action)
        env.render(mode='human')
    else:
        try:
            action_idx, val, reps = command.split(' ')
            action_idx = int(action_idx)
            val = float(val)
            reps = int(reps)
            action[action_idx] = val
            for _ in range(reps):
                obs, reward, done, info = env.step(action)
                env.render(mode='human')
                print('Grippper position:', env.sim.data.get_site_xpos('grip'))
        except Exception as e:
            print(e)
