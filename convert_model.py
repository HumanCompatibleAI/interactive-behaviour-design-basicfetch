import argparse
import datetime
import os
import sys

import gym
import tensorflow as tf
from gym.envs.registration import register

import baselines
import basicfetch
from baselines.common.tf_util import get_session
from baselines.run import main as baselines_run_main

parser = argparse.ArgumentParser()
parser.add_argument('in_model')
parser.add_argument('ckpt_dir')
parser.add_argument('name')
args = parser.parse_args()

basicfetch.register()
env_name = 'FetchBasicUpDense-v0'
baselines.run._game_envs['robotics'].add(env_name)
arg_str = f"--alg=ppo2 --env={env_name} --num_env 1 --num_timesteps 0 --load_path {args.in_model}"
sys.argv = arg_str.split(" ")
baselines_run_main()

sess = get_session()
saver = tf.train.Saver(var_list=tf.trainable_variables())
now = str(datetime.datetime.now())
path = os.path.join(args.ckpt_dir, 'policy-{}-{}.ckpt'.format(args.name, now))
saved_path = saver.save(sess, path)
print("Saved model to '{}'".format(saved_path))
