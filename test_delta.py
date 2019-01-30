import gym
import numpy as np

from basicfetch import register

register()
env = gym.make('FetchBasicDelta-v0').unwrapped  # unwrap past TimeLimit

np.set_printoptions(precision=3, floatmode='fixed', linewidth=999, suppress=True, sign=' ')

for random_initial_pos in [False, True]:
    env.unwrapped.random_initial_gripper_position = random_initial_pos
    for action_n in range(env.action_space.shape[0]):
        action = np.zeros(env.action_space.shape)
        for action_val in [-1, 0, +1]:
            env.reset()
            print(f"Action {action_n} = {action_val}")
            action[action_n] = action_val
            for _ in range(100):
                env.step(action)
                env.render()
                ctrl = env.unwrapped.sim.data.ctrl
                print(env.unwrapped.actuator_ctrl_to_normalized_action(ctrl))
            input()
