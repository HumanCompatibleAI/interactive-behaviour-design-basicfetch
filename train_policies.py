#!/usr/bin/env python3
import argparse
import os
import subprocess
from multiprocessing import Process

from tmuxprocess import TmuxProcess

parser = argparse.ArgumentParser()
parser.add_argument('runs_dir')
args = parser.parse_args()

names = ['up', 'down', 'left', 'right', 'forward', 'backward']
vecs = ['0 0 1', '0 0 -1', '1 0 0', '-1 0 0', '0 1 0', '0 -1 0']
procs = []
for name, vec in zip(names, vecs):
    dir = os.path.join(args.runs_dir, 'FetchBasic' + name.capitalize())
    p = TmuxProcess(target=subprocess.run, args=[['python', 'train.py', dir, vec]])
    p.start()
    procs.append(p)
subprocess.run(f'tmux attach -t {procs[0].tmux_sess}'.split(' '))
