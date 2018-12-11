import argparse
import multiprocessing
import os
import sys

import gym
from gym.envs.registration import register
from gym.wrappers import Monitor

import baselines
import basicfetch
from baselines import logger
from baselines.run import main as baselines_run_main

basicfetch.register()

parser = argparse.ArgumentParser()
parser.add_argument('dir')
parser.add_argument('--n_envs', type=int, default=8)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
os.environ["OPENAI_LOGDIR"] = args.dir
os.environ["OPENAI_LOG_FORMAT"] = 'stdout,log,csv,tensorboard'

first_env_semaphore = multiprocessing.Semaphore()


def make_env():
    env = gym.make('FetchBasicDenseUp-v0')
    env._max_episode_steps = 250
    if first_env_semaphore.acquire(timeout=0):
        env = Monitor(env, video_callable=lambda n: n % 20 == 0, directory=logger.get_dir())
    return env


env_name = 'FetchBasicDenseUpMonitor-v0'

register(
    id=env_name,
    entry_point=make_env,
)

baselines.run._game_envs['robotics'].add(env_name)
sys.argv = f"--alg=ppo2 --env={env_name} --num_env {args.n_envs} --nsteps 128 --num_timesteps 50e6 --seed {args.seed}".split(
    " ")
baselines_run_main()
