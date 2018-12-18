import numpy as np
import os
from gym import Wrapper
from gym.envs import register as gym_register
from gym.envs.robotics import utils
from gym.envs.robotics.robot_env import RobotEnv
from gym.utils import EzPickle
from gym.wrappers import FlattenDictWrapper


class FetchEnvBasic(RobotEnv, EzPickle):
    def __init__(self):
        self.reward_type = None
        model_path = os.path.join(os.path.dirname(__file__), 'mujoco-py/xmls/fetch/main.xml')
        RobotEnv.__init__(self, model_path=model_path, n_substeps=20, n_actions=8, initial_qpos=None)
        EzPickle.__init__(self)

    def get_ctrl_names(self):
        return self.sim.model.actuator_names

    def _env_setup(self, initial_qpos):
        initial_qpos = {
            'slide0': 0.405,
            'slide1': 0.48,
            'slide2': 0.0,
        }
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)

        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        object_qpos[:3] = [1.3, 0.75, 0.4]
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        # utils.reset_mocap_welds(self.sim)
        # self.sim.forward()
        #
        # # Move end effector into position.
        # gripper_target = np.array([-0.498, 0.005, -0.431]) + self.sim.data.get_site_xpos('grip')
        # gripper_rotation = np.array([1., 0., 1., 0.])
        # self.sim.data.set_mocap_pos('mocap', gripper_target)
        # self.sim.data.set_mocap_quat('mocap', gripper_rotation)
        # for _ in range(10):
        #     self.sim.step()

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
        quat = self.sim.data.get_body_xquat('gripper_link')
        if all(np.isclose(quat, [np.sqrt(0.5), 0, np.sqrt(0.5), 0], atol=0.1)):
            level_reward = 1
        else:
            level_reward = 0

        if self.reward_type == 'left':
            r_vec = [1, 0, 0]
        elif self.reward_type == 'right':
            r_vec = [-1, 0, 0]
        elif self.reward_type == 'forward':
            r_vec = [0, 1, 0]
        elif self.reward_type == 'backward':
            r_vec = [0, -1, 0]
        elif self.reward_type == 'up':
            r_vec = [0, 0, 1]
        elif self.reward_type == 'down':
            r_vec = [0, 0, -1]
        else:
            raise Exception("Unknown reward type", self.reward_type)

        pos = self.sim.data.get_site_xpos('grip')

        pos_reward = np.dot(pos, r_vec)

        # assuming pos_reward has a scale of about 1, and level_reward also 0/1, so should be balanced
        return level_reward + pos_reward

    def _sample_goal(self):
        return np.array((0, 0, 0))

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

        return {
            'observation': obs.copy(),
            'achieved_goal': gripper_pos,
            'desired_goal': self.goal
        }


class StartWithExplore(Wrapper):
    def reset(self):
        self.env.reset()
        action = np.zeros(self.env.action_space.shape)
        for _ in range(100):
            action += np.random.normal(loc=0, scale=0.1, size=self.action_space.shape)
            action = np.clip(action, -1.0, 1.0)
            obs, reward, done, info = self.env.step(action)
        return obs

    def step(self, action):
        return self.env.step(action)


def make_env():
    env = FetchEnvBasic()
    env = FlattenDictWrapper(env, ['observation'])
    env = StartWithExplore(env)
    return env


def register():
    gym_register(
        id=f'FetchBasic-v0',
        entry_point=make_env,
        max_episode_steps=250
    )
