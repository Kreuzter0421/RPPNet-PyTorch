import argparse
import subprocess
import os
import datetime
import pickle
import numpy as np
from pprint import pprint

# Parser
parser = argparse.ArgumentParser(description="This is a sample program.")
parser.add_argument('-n', '--new', action='store_true')
parser.add_argument("-m","--mode", type=str, default="None", help="* train_RPP | train_Note | inference_RPP | inference_Note ")
parser.add_argument("-i","--input", type=str, default="auto", help="* specify data path")
parser.add_argument("-o","--output", type=str, default="auto", help="* specify save path")
args = parser.parse_args()

mode_available = ['train_RPP','train_Note','inference_RPP','inference_Note']
if args.mode!='None' and args.mode not in mode_available:
    print('* Your Mode is not available!')
    print('* Please choose one of the following modes: {}'.format(mode_available))
    exit()
else:
    print("\n######## CURRENT MODE IS {} #########\n".format(args.mode.upper()))

# utils
def sort_key(filename):
    month, day, hour, minute = map(int, filename.split('-'))
    return month, day, hour, minute

# New Dir
if args.new:
    print('* Creating New Experiment ...')

    # make new experiment folder
    date = datetime.datetime.now()
    date = f'{date.month}-{date.day}-{date.hour}-{date.minute}'
    workdir = os.path.join('Exp_Record',date)
    if not os.path.exists(workdir):
        os.makedirs(workdir)
    else:
        print('Experiment already exists!')
        exit()

    dir_list = ['midi','result_RPP','result_Note','pkl','midi_inference']
    for dir in dir_list:
        os.makedirs(os.path.join(workdir,dir))

    print(f'* Success!\n Current Experiment Dir is {workdir}')

# Mode
if args.mode == 'train_RPP':
    print('* Training RPP Model ...')

    workspace_dir = "workspace/RPP_level/workspace"
    subprocess.call(["python", "train.py"], cwd=workspace_dir)

elif args.mode == 'train_Note':
    print('* Training Note Model ...')

    workspace_dir = "workspace/Note_level/workspace"
    subprocess.call(["python", "train.py"],cwd=workspace_dir)

elif args.mode == 'inference_RPP':
    # default path
    record_dir = os.path.join(os.getcwd(),'Exp_Record')
    dir_list = [f for f in os.listdir(record_dir) if '-' in f]
    dir_list = sorted(dir_list, key=sort_key)
    exp_dir = os.path.join(record_dir,dir_list[-1])
    print(exp_dir)
    workspace_dir = "workspace/RPP_level/workspace"
    input = args.input
    output = args.output
    cmd = ["python", "inference.py","-e",f"{exp_dir}","-m","feature","-i",f"{input}","-o",f"{output}"]
    subprocess.call(cmd, cwd=workspace_dir)

elif args.mode == 'inference_Note':

    # default path
    record_dir = os.path.join(os.getcwd(),'Exp_Record')
    dir_list = [f for f in os.listdir(record_dir) if '-' in f]
    dir_list = sorted(dir_list, key=sort_key)
    exp_dir = os.path.join(record_dir,dir_list[-1])
    print(exp_dir)
    workspace_dir = "workspace/Note_level/workspace"
    subprocess.call(["python", "inference.py", "-e", f"{exp_dir}","-i",f"{args.input}","-o",f"{args.output}"], cwd=workspace_dir)











