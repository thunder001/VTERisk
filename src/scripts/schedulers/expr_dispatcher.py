import argparse
import subprocess
import os
import pickle
import pprint as pp
import csv
import json
import sys
import pdb
import time
from os.path import dirname, realpath
import random
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
import disrisknet.utils.parsing as parsing
import disrisknet.models.factory as model_factory
import disrisknet.learn.state_keeper as state
from disrisknet.utils.parsing import dict2args


CONFIG_NOT_FOUND_MSG = "ERROR! {}: {} config file not found."
RESULTS_PATH_APPEAR_ERR = 'ALERT! Existing results for the same config {}.'

parser = argparse.ArgumentParser(description='DiskRiskNet Grid Search Dispatcher.')
parser.add_argument("--experiment-config-path", required=True, type=str, help="Path of experiment config")
parser.add_argument('--force-rerun', action='store_true', default=False, help='whether to rerun experiments with the same result file location')
parser.add_argument('--force-evaluate', action='store_true', default=False, help='run evaluation on an already run grid search')
args = parser.parse_args()

print("args: {}".format(args))
def launch_experiment(gpu, flag_string):
    # print(flag_string)
    log_dir = get_log_dir(flag_string)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    log_name = parsing.md5(flag_string)
    # print(log_name)
    log_stem = os.path.join(log_dir, log_name)
    # print(log_stem)
    log_path = '{}.txt'.format(log_stem)
    # print(log_path)
    results_path = '{}.results'.format(log_stem)

    if args.force_evaluate:
        flag_string = flag_string.replace("--test ", "")
        flag_string = flag_string.replace("--train ", "")
        original_args = dict2args(pickle.load(open(results_path, 'rb')))
        snapshot = state.get_identifier(original_args)
        flag_string += " --snapshot {}".format(snapshot)

    # experiment_string = "CUDA_VISIBLE_DEVICES={} /opt/conda/bin/python scripts/main.py {} --results_path='{}'".format(
    #     gpu, flag_string, results_path)

    experiment_string = "python scripts/main.py {} --log_name={} --log_dir={}".format(flag_string, log_name, log_dir)

    # print("\nFlag string:\n")
    # print(flag_string)
    # experiment_string = "python scripts/main.py {} --results_path={}".format(flag_string, results_path) # for CPU only
    # print("\nExperiment Sting:\n")
    print(experiment_string)

    pipe_str = ">>" if args.force_evaluate or ("--resume" in flag_string and not args.force_rerun) else ">"
    shell_cmd = "{} {} {} 2>&1".format(experiment_string, pipe_str, log_path)

    if os.path.exists(log_path):
        print(RESULTS_PATH_APPEAR_ERR.format(log_name))
        if not args.force_rerun and not args.force_evaluate:
            sys.exit(1)
        else:
            print("[WARNING] Overiding the alert {RESULTS_PATH_APPEAR_ERR}...")

    # pp.pprint("[CMD] Launched exp: {}".format(shell_cmd))
    os.system(shell_cmd)

def get_log_dir(flag_string):
    flag_ls = flag_string.split(' ')
    log_dir_idx = flag_ls.index('--log_dir') + 1
    log_dir = flag_ls[log_dir_idx]
    return log_dir

if __name__ == "__main__":
    if not os.path.exists(args.experiment_config_path):
        print(CONFIG_NOT_FOUND_MSG.format("experiment", args.experiment_config_path))
        sys.exit(1)

    # with open(args.experiment_config_path) as f:
    #     job_list = f.read().splitlines()
    config = json.load(open(args.experiment_config_path, 'r'))
    job_list, _ = parsing.parse_dispatcher_config(config)

    # print(job_list)

    # [print(params) for params in job_list]
    [launch_experiment(gpu=2, flag_string=str(params)) for params in job_list]
    # launch_experiment(gpu=2, flag_string=job_list[0])

    sys.exit(0)
