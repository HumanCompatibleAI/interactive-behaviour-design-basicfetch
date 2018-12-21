import gym
import numpy as np
from matplotlib.pyplot import subplot, hist, show, title, xlim

from reward_functions import reward_function_dict
from basicfetch import register
register()
env = gym.make('FetchBasic-v0')
env.unwrapped.reward_func = reward_function_dict['dummy']
initial_positions = []
for n in range(50):
    print(n)
    env.reset()
    gripper_position = env.unwrapped.sim.data.get_site_xpos('grip')
    initial_positions.append(np.copy(gripper_position))
initial_positions = np.array(initial_positions)
print(initial_positions)
subplot(1, 3, 1)
hist(initial_positions[:, 0])
xlim([-1, 2])
title('x')
subplot(1, 3, 2)
hist(initial_positions[:, 1])
xlim([-1, 2])
title('y')
subplot(1, 3, 3)
hist(initial_positions[:, 2])
xlim([-1, 2])
title('z')
show()
