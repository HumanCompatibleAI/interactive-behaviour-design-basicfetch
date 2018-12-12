#!/usr/bin/env python3
import argparse
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('runs_dir')
args = parser.parse_args()

names = ['up', 'down', 'left', 'right', 'forward', 'backward']
vecs = ['0 0 1', '0 0 -1', '1 0 0', '-1 0 0', '0 1 0', '0 -1 0']
for name, vec in zip(names, vecs):
    dir = os.path.join(args.runs_dir, 'FetchBasic' + name.capitalize())
    subprocess.run(['python', 'train.py', dir, vec])
