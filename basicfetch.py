import numpy as np
import os
from gym import Wrapper
from gym.envs import register as gym_register
from gym.envs.robotics.robot_env import RobotEnv
from gym.utils import EzPickle
from gym.wrappers import FlattenDictWrapper

from reward_functions import reward_function_dict


class FetchEnvBasic(RobotEnv, EzPickle):
    def __init__(self, delta_control):
        model_path = os.path.join(os.path.dirname(__file__), 'mujoco-py/xmls/fetch/main.xml')
        RobotEnv.__init__(self,
                          model_path=model_path,
                          n_substeps=20,  # copied from FetchEnv
                          n_actions=8,    # 8 actuators
                          initial_qpos=None)
        EzPickle.__init__(self)
        self.reward_func = reward_function_dict['dummy']
        self.delta_control = delta_control
        self._action = np.zeros(self.action_space.shape)

    def get_ctrl_names(self):
        return self.sim.model.actuator_names

    def _env_setup(self, initial_qpos):
        # Position the robot base
        base_initial_qpos = {
            'slide0': 0.405,
            'slide1': 0.48,
            'slide2': 0.0,
        }
        for name, value in base_initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)

        # Position the small box
        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        object_qpos[:3] = [1.3, 0.75, 0.4]
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        # Move end-effector into position
        # utils.reset_mocap_welds(self.sim)
        # self.sim.forward()
        # gripper_target = np.array([-0.498, 0.005, -0.431]) + self.sim.data.get_site_xpos('grip')
        # gripper_rotation = np.array([1., 0., 1., 0.])
        # self.sim.data.set_mocap_pos('mocap', gripper_target)
        # self.sim.data.set_mocap_quat('mocap', gripper_rotation)
        # for _ in range(10):
        #     self.sim.step()

    def _set_action(self, action):
        assert action.shape == (8,)
        if self.delta_control:
            action_delta = action / 40
            action_delta[-1] *= 40  # move the gripper fingers faster
            self._action += action_delta
            self._action = np.clip(self._action, self.action_space.low, self.action_space.high)
            action = self._action
        action = np.concatenate([action, [action[-1]]])  # control the fingers together
        ctrlrange = self.sim.model.actuator_ctrlrange
        actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.
        actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.
        self.sim.data.ctrl[:] = actuation_center + action * actuation_range
        self.sim.data.ctrl[:] = np.clip(self.sim.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1])

    def _is_success(self, achieved_goal, desired_goal):
        return False

    def _sample_goal(self):
        return np.array((0, 0, 0))

    def compute_reward(self, achieved_goal, desired_goal, info):
        quat = self.sim.data.get_body_xquat('gripper_link')
        pos = self.sim.data.get_site_xpos('grip')
        return self.reward_func(quat, pos)

    def _get_obs(self):
        gripper_pos = self.sim.data.get_site_xpos('grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        gripper_vel = self.sim.data.get_site_xvelp('grip') * dt
        obs = np.concatenate([gripper_pos, gripper_vel])
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


def make_env(delta=False):
    env = FetchEnvBasic(delta)
    env = FlattenDictWrapper(env, ['observation'])
    env = StartWithExplore(env)
    return env


def register():
    gym_register(
        id=f'FetchBasic-v0',
        entry_point=make_env,
        max_episode_steps=10
    )
    gym_register(
        id=f'FetchBasicDelta-v0',
        entry_point=lambda: make_env(delta=True),
        max_episode_steps=10
    )
