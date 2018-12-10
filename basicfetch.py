import os

import numpy as np
from gym.envs import register as gym_register
from gym.envs.robotics import utils
from gym.envs.robotics.robot_env import RobotEnv


class FetchEnvBasic(RobotEnv):
    def __init__(self):
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
        return 0

    def _sample_goal(self):
        return

    def _get_obs(self):
        # gripper unit position
        grip_pos = self.sim.data.get_site_xpos('grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('grip') * dt

        # joint positions and velocities
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        # grippers distance and velocity
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt

        obs = np.concatenate([grip_pos, gripper_state, grip_velp, gripper_vel])

        return {
            'observation': obs.copy(),
            'achieved_goal': np.zeros(0),
            'desired_goal': np.zeros(0)
        }


def register():
    gym_register(
        id='FetchBasic-v0',
        entry_point=FetchEnvBasic
    )
