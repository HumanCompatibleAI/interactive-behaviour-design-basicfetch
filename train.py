import argparse
import multiprocessing
import subprocess

import os
import sys

import gym
from gym.envs.registration import register
from gym.wrappers import Monitor

import baselines
import basicfetch
from baselines import logger
from baselines.run import main as baselines_run_main


def get_git_rev():
    try:
        cmd = 'git rev-parse --short HEAD'
        git_rev = subprocess.check_output(cmd.split(' '), stderr=subprocess.PIPE).decode().rstrip()
        return git_rev
    except subprocess.CalledProcessError:
        return 'unkrev'


def make_env():
    env = gym.make(f'FetchBasic{args.env_type}-v0')
    if first_env_semaphore.acquire(timeout=0):
        env = Monitor(env, video_callable=lambda n: n % 5 == 0, directory=logger.get_dir())
    return env


basicfetch.register()

parser = argparse.ArgumentParser()
parser.add_argument('dir')
parser.add_argument('env_type')
parser.add_argument('--n_envs', type=int, default=8)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
args.dir += '_' + get_git_rev()
os.environ["OPENAI_LOGDIR"] = args.dir
os.environ["OPENAI_LOG_FORMAT"] = 'stdout,log,csv,tensorboard'

first_env_semaphore = multiprocessing.Semaphore()

env_name = 'FetchBasicUpDenseMonitor-v0'

register(
    id=env_name,
    entry_point=make_env,
)

baselines.run._game_envs['robotics'].add(env_name)
arg_str = f"--alg=ppo2 --env={env_name} --num_env {args.n_envs} --nsteps 128 --num_timesteps 1e4 --seed {args.seed} "
arg_str += f"--save_path {os.path.join(args.dir, 'saved_model')}"
sys.argv = arg_str.split(" ")
baselines_run_main()
