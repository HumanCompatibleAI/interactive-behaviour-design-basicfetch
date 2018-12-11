import os

import numpy as np
from gym.envs import register as gym_register
from gym.envs.robotics import utils
from gym.envs.robotics.robot_env import RobotEnv


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
        assert isinstance(achieved_goal, np.float64), type(desired_goal)
        assert isinstance(desired_goal, float), type(desired_goal)
        if self.reward == 'sparse':
            return float(achieved_goal > desired_goal)
        elif self.reward == 'dense':
            capped_distance = max(desired_goal - achieved_goal, 0)
            return -capped_distance
        else:
            raise Exception(f"Unknown reward type '{self.reward}'")

    def _sample_goal(self):
        return 0.8

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
        else:
            raise Exception(f"Unknown target '{self.reward}'")
        return {
            'observation': obs.copy(),
            'achieved_goal': achieved,
            'desired_goal': self.goal
        }


def register():
    for reward in ['sparse', 'dense']:
        for target in ['up', 'right']:
            gym_register(
                id=f'FetchBasic{target.capitalize()}{reward.capitalize()}-v0',
                entry_point=FetchEnvBasic,
                kwargs={'target': target, 'reward': reward}
            )
