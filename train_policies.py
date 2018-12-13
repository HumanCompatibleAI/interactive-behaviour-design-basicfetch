#!/usr/bin/env python3

import argparse
import os
import subprocess

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


names = ['left', 'right', 'front', 'back']
procs = []
start_tmux_sess_with_cmd('train_subpolicies', 'echo hi')
for name in names:
    dir = os.path.join(args.runs_dir, 'FetchBasic' + name.capitalize())
    cmd = f"python train.py '{dir}' '{name}'"
    run_in_tmux_sess('train_subpolicies', cmd, name)
