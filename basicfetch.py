import numpy as np
import os
from gym import Wrapper
from gym.envs import register as gym_register
from gym.envs.robotics import utils
from gym.envs.robotics.robot_env import RobotEnv
from gym.wrappers import FlattenDictWrapper


class FetchEnvBasic(RobotEnv):
    def __init__(self, target, reward):
        self.target_type = target
        self.reward = reward
        model_path = os.path.join(os.path.dirname(__file__), 'mujoco-py/xmls/fetch/main.xml')
        RobotEnv.__init__(self, model_path=model_path, n_substeps=20, n_actions=8, initial_qpos=None)

    def get_ctrl_names(self):
        return self.sim.model.actuator_names

    def _set_action(self, action):
        assert action.shape == (8,)
        action = np.concatenate([action, [action[-1]]])
        ctrlrange = self.sim.model.actuator_ctrlrange
        actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.
        actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.
        self.sim.data.ctrl[:] = actuation_center + action * actuation_range
        self.sim.data.ctrl[:] = np.clip(self.sim.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1])

    def _is_success(self, achieved_goal, desired_goal):
        return False

    def compute_reward(self, achieved_goal, desired_goal, info):
        if self.target_type == 'level':
            assert achieved_goal.shape == (4,), achieved_goal.shape
            assert desired_goal.shape == (4,), desired_goal.shape
            print(achieved_goal, desired_goal)
            diff = np.linalg.norm(achieved_goal - desired_goal)
            return -diff
        elif self.target_type in ['up', 'right']:
            assert isinstance(achieved_goal, np.float64), type(desired_goal)
            assert isinstance(desired_goal, np.float64), type(desired_goal)
            if self.reward == 'sparse':
                return float(achieved_goal > desired_goal)
            elif self.reward == 'dense':
                capped_distance = max(desired_goal - achieved_goal, 0)
                return -capped_distance
            elif self.reward == 'max':
                return achieved_goal
            else:
                raise Exception(f"Unknown reward type '{self.reward}'")
        else:
            raise Exception(f"Unknown reward target '{self.target_type}'")

    def _sample_goal(self):
        if self.target_type == 'level':
            return np.array([1., 0., 1., 0.])
        elif self.target_type in ['up', 'right']:
            # np.float64 because when training with baselines something wants to call copy()
            return np.float64(0.8)
        else:
            raise Exception(f"Unknown reward target '{self.target_type}'")

    def _get_obs(self):
        # gripper position
        gripper_pos = self.sim.data.get_site_xpos('grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        gripper_vel = self.sim.data.get_site_xvelp('grip') * dt

        # joint positions and velocities
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        # grippers fingers distance and velocity
        fingers_distance = robot_qpos[-2:]
        fingers_vel = robot_qvel[-2:] * dt

        obs = np.concatenate([gripper_pos, gripper_vel, fingers_distance, fingers_vel])

        if self.target_type == 'up':
            achieved = gripper_pos[2]
        elif self.target_type == 'right':
            achieved = gripper_pos[1]
        elif self.target_type == 'level':
            achieved = self.sim.data.get_body_xquat('gripper_link')
        else:
            raise Exception(f"Unknown reward target '{self.reward}'")
        return {
            'observation': obs.copy(),
            'achieved_goal': achieved,
            'desired_goal': self.goal
        }


class StartWithExplore(Wrapper):
    def reset(self):
        self.env.reset()
        for _ in range(30):
            obs, reward, done, info = self.env.step(self.env.action_space.sample())
        return obs

    def step(self, action):
        return self.env.step(action)


def make_env(target, reward):
    env = FetchEnvBasic(target, reward)
    env = FlattenDictWrapper(env, ['observation'])
    env = StartWithExplore(env)
    return env


def register():
    for reward in ['sparse', 'dense', 'max']:
        for target in ['up', 'right', 'level']:
            gym_register(
                id=f'FetchBasic{target.capitalize()}{reward.capitalize()}-v0',
                entry_point=make_env,
                max_episode_steps=250,
                kwargs={'target': target, 'reward': reward}
            )
