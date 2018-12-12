#!/usr/bin/env python3
import argparse
import os
import subprocess

from tmuxprocess import TmuxProcess

parser = argparse.ArgumentParser()
parser.add_argument('runs_dir')
args = parser.parse_args()


def start_tmux_sess_with_cmd(sess_name, cmd):
    cmd += '; echo; read -p "Press enter to exit..."'
    cmd = ['tmux', 'new-sess', '-d', '-s', sess_name, '-n', f'{sess_name}-main', cmd]
    subprocess.run(cmd)


def run_in_tmux_sess(sess_name, cmd, window_name):
    cmd += '; echo; read -p "Press enter to exit..."'
    tmux_cmd = ['tmux', 'new-window', '-ad', '-t', f'{sess_name}-main', '-n', window_name, cmd]
    subprocess.run(tmux_cmd)


names = ['up', 'down', 'left', 'right', 'forward', 'backward']
vecs = ['0 0 1', '0 0 -1', '1 0 0', '-1 0 0', '0 1 0', '0 -1 0']
procs = []
start_tmux_sess_with_cmd('train_subpolicies', 'echo hi')
for name, vec in zip(names, vecs):
    dir = os.path.join(args.runs_dir, 'FetchBasic' + name.capitalize())
    cmd = f"python train.py '{dir}' '{vec}'"
    run_in_tmux_sess('train_subpolicies', cmd, name)
