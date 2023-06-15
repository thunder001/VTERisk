import json
import argparse
from collections import Counter
from tqdm import tqdm
import pickle
import numpy as np
import pickle
import pandas as pd
import matplotlib

from performance_table import get_performance_ci

matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from os.path import dirname, realpath
import sys
import random
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
from performance_table import get_boot_metric_clf
import pancnet.datasets.factory as dataset_factory
import pancnet.models.factory as model_factory
import pancnet.learn.train as train
import pancnet.learn.state_keeper as state
from pancnet.utils.time_logger import time_logger
from pancnet.utils.parsing import md5
from plot_model_performance import plot_curve
from pancnet.utils.eval import include_exam_and_determine_label
import datetime

def get_probs_golds(test_preds, month='60', age_mapper=None, age_subgroup=None, sex_mapper=None, sex_subgroup=None):

    probs_for_eval, golds_for_eval = [], []
    if age_subgroup: min_, max_ = list(map(int, age_subgroup.split()))

    for prob_arr, censor_time, gold, date, pid in tqdm(zip(test_preds["probs"], 
                                                test_preds["censor_times"], 
                                                test_preds["golds"],
                                                test_preds["dates"],
                                                test_preds["pids"])):
        index = args.timepoints.index(month)
        include, label = include_exam_and_determine_label(index, censor_time, gold,  args.indipendent_eval)
        if age_mapper and age_subgroup:
            age_at_assessment = (parse_date(date) - age_mapper[pid]).days // 365
            is_above_min = age_at_assessment > min_
            is_below_max = age_at_assessment < max_
            include = include and is_above_min and is_below_max
        if include:        
            probs_for_eval.append(prob_arr[index])
            golds_for_eval.append(label)

    #incidence = len(set([p for p,g in zip(test_preds['pids'],test_preds['golds']) if g]))/len(set(test_preds['pids']))
    incidence = len(set([p for p,g in zip(test_preds['pids'],test_preds['golds']) if g]))/len(set(test_preds['pids']))
    incidence = round(incidence, 4)
    return probs_for_eval, golds_for_eval, incidence

def parse_date(date_str):
    if date_str == 'NA' or type(date_str) is float:
        return datetime.datetime(9999,1,1,0,0)
    else:
        if len(date_str) == 10:
            format_str = '%Y-%m-%d'
        elif len(date_str) == 19:
            format_str = '%Y-%m-%dT%H:%M:%S'
        elif len(date_str) == 4:
            format_str = '%Y'
        else:
            raise Exception("Format for {} not recognized!".format(date_str))
    return datetime.datetime.strptime(date_str, format_str)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PancNet Grid Search Results Collector.')
    parser.add_argument("--subgroup_config_path", required=True, type=str, help="Path of the master experiment config")
    parser.add_argument('--best_month', type=str, default='60', help='Specify maximum number of years to collect.')
    parser.add_argument('--indipendent_eval', action='store_true', default=False, help='choose between indipendent or cumulative evalution')
    parser.add_argument('--timepoints', type=str, default='3-6-12-36-60-120', help='choose between indipendent or cumulative evalution')
    parser.add_argument('--fig_suffix', type=str, default='', help='Specify maximum number of years to collect.')
    parser.add_argument('--n_samples', type=int, default=100, help='every N element in the array take the sample')

    args = parser.parse_args()

    #cpr_df = pd.read_csv('/users/secureome/home/projects/registries/2018/cpr/t_person.tsv', sep='\t', dtype='str')
    #pid2gender = dict(zip(cpr_df.v_pnr_enc, cpr_df.C_KON))
    #pid2bday = dict(zip(cpr_df.v_pnr_enc, map(parse_date, cpr_df.D_FODDATO)))
    #enc_pid2bday = {md5(k):v for k,v in pid2bday.items()}
    #enc_pid2gender = {md5(k):v for k,v in pid2gender.items()}

    # cpr_df = pd.read_csv('/users/secureome/home/projects/registries/2018/cpr/t_person.tsv', sep='\t', dtype='str')
    mgb_df = pd.read_csv('data/dob_dod.csv', sep=',', dtype=str)
    pid2bday = dict(zip(mgb_df.person_id, map(parse_date, mgb_df.year_of_birth)))
    enc_pid2bday = {md5(k):v for k,v in pid2bday.items()}

    best_exp_ids_config = json.load(open(args.subgroup_config_path, 'r'))
    args.timepoints = args.timepoints.split('-')

    fig = plt.figure(figsize=[18,12])
    fig.suptitle('Model performance sub-groups', fontsize=16)
    gs = GridSpec(1,2)
    ax_prc=fig.add_subplot(gs[0])
    ax_roc=fig.add_subplot(gs[1])
    ax_prc.set(xlim=(-0.01, 1.01), title='PRC')
    ax_roc.set(xlim=(-0.01, 1.01), title='ROC')
    ax_roc.plot([0, 1], [0, 1], linestyle='--', color='grey', label='random guess (AUROC: 0.500)')

    for(exp_id, log_dir, model_name) in (zip(best_exp_ids_config['exp_id'], 
                                            best_exp_ids_config['log_dir'], 
                                            best_exp_ids_config['model_name'])):

        test_preds_path = os.path.join(log_dir, "{}.results.test_preds".format(exp_id))
        test_preds = pickle.load(open(test_preds_path, 'rb'))

        for age_range in ["0 50", "50 100", "-1 120"]:
            probs_for_eval, golds_for_eval, incidence = get_probs_golds(test_preds, month=args.best_month, 
                                                                        age_mapper=enc_pid2bday,
                                                                        age_subgroup=age_range)
            
            probs_for_eval = np.array(probs_for_eval)[::args.n_samples]
            golds_for_eval = np.array(golds_for_eval)[::args.n_samples]
            
            #
            # curves, incidence_ci, auroc_ci, fpr_ci, tpr_ci, auprc_ci, precision_ci, recall_ci, odds_ratio_ci, threshold_ci
            curves_records, incidence_ci, auroc_ci, _, _, auprc_ci, _, _, odds_ratio_ci, _ = get_performance_ci(probs_for_eval, 
                golds_for_eval, age_range, None, None, None, n_boot=2)
            
            _, curves, *_ = curves_records
            curves = json.loads(curves)

            plot_curve(curves["fpr"], 
                    curves["tpr"], 
                    auc_handle=auroc_ci[1],
                    curve_type='roc', 
                    ax=ax_roc, 
                    model_name=age_range)
            
            plot_curve(curves["recall"], 
                curves["precision"], 
                auc_handle=auprc_ci[1], 
                curve_type='prc', 
                ax=ax_prc,
                model_name=age_range)
        
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("figures/age_subgroup{}.png".format(args.fig_suffix), bbox_inches='tight')
        plt.savefig("figures/age_subgroup{}.svg".format(args.fig_suffix), bbox_inches='tight', format='svg')
