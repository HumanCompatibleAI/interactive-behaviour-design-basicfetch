import argparse
import sys

import gym
from baselines.run import main as baselines_run_main
from gym.envs.registration import register

import baselines
import basicfetch


def make_env():
    env = gym.make(f'FetchBasicUpDense-v0')
    return env


basicfetch.register()

parser = argparse.ArgumentParser()
parser.add_argument('model')
args = parser.parse_args()

register(
    id='Env-v0',
    entry_point=make_env,
)

baselines.run._game_envs['robotics'].add('Env-v0')
arg_str = f"--alg=ppo2 --env=Env-v0 --num_env 1 --num_timesteps 0 --load_path {args.model} --play"
sys.argv = arg_str.split(" ")
baselines_run_main()
