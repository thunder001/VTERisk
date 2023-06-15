## Takes as input a metric and the output folders of the grid to comprare
# 
import argparse
import os
import pickle as pkl
import csv
import json
import sys
import subprocess
from os.path import dirname, realpath
import random
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def rule_for_best(df, metric, months=None):
    df["score"] = df[[metric.format(m) for m in months]].mean(axis=1)
    df = df.sort_values(by='score', ascending=False)
    return df

def get_best_from_summary(res):
    metric = "dev_{}month_auroc_i"# for the exclusion interval we look at auroc just to get the NAN
    exclusion_months = []

    for m in args.timepoints:
        if res[metric.format(m)].isna().sum()!=0:
            exclusion_months.append(m)
    exclusion_months = sorted(exclusion_months, reverse=True)
    exclusion_months.append(0)

    #Find indexes of exclusion depending on the nan from summary table
    exclusion_index = []
    for i,m in enumerate(exclusion_months):
        if m==0:
            index = res.index
        else:
            index=res[res[metric.format(m)].isna()].index
        if exclusion_index:
            prev_idx = pd.Index([])
            for el in exclusion_index:
                prev_idx = prev_idx.union(el)
            index = index[~index.isin(prev_idx)]
        exclusion_index.append(index)

    exclusion_dict = dict(zip(exclusion_months, exclusion_index))
    metric = args.metric.replace("test", "dev")
    best_cols = [metric.format(m) for m in args.timepoints]

    best_exp_for_exclusion = {}
    for i in sorted(exclusion_dict, reverse=True):
        subdf = res.loc[exclusion_dict[i], best_cols]
        best_exp_for_exclusion[i] = {}
        for m in args.timepoints:
            subdfbest = rule_for_best(subdf, metric, months=[m])
            best_exp_id = subdfbest.index[0]
            score = subdfbest.score[0]
            print ("## best exp id for exlcusion interval {} predicting month {} using metric {} is exp_id {} @ {}".format(
                i, m, metric, best_exp_id, score
            ))
            best_exp_for_exclusion[i][m] = best_exp_id 
    return exclusion_months, best_exp_for_exclusion


def get_stats_from_output(path, log, model_name):
    figure_path = os.path.join(path, 'figures')
    summary_file = [el for el in os.listdir(figure_path) if 'summary_df' in el]
    if any(summary_file):
        summary_df = pd.read_csv(os.path.join(figure_path, summary_file[0]), index_col=0)
    else:
        print('No summary found at{}. Run summary collector first. Run collect grid first! '.format(figure_path))
        sys.exit(1)

    print ("Extract best model: {}".format(model_name))
    exclusion_months , best_exp_for_exclusion = get_best_from_summary(summary_df)

    stats_pkls = {}
    for exclusion_index, best_exp_id in best_exp_for_exclusion.items():
        stats_pkls[exclusion_index] = {}
        for m in best_exp_id.keys():
            stats_path = os.path.join(log, best_exp_id[m] + ".results.{}_stats".format(args.dataset))

            stats_pkl = pkl.load(open(stats_path, 'rb'))

            stats_pkls[exclusion_index][m] = stats_pkl
    return stats_pkls

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PancNet Grid Search Results Collector.')
    parser.add_argument('--config_figures', type=str, default='configs/figures/summary_performances.json', help='Specify maximum number of years to collect.')
    parser.add_argument('--timepoints', type=str, default=None, help='Specify maximum number of years to collect.')
    parser.add_argument('--metric', type=str, default=None, help='Specify maximum number of years to collect.')
    parser.add_argument('--name_prefix', type=str, default='', help='Prefix to name fig to save.')

    args = parser.parse_args()
    sys.stdout = open("figures/{}.readme.txt".format(args.metric.replace('}', '').replace('{', '')), 'w+')
    sys.stderr = open("figures/{}.readme.txt".format(args.metric.replace('}', '').replace('{', '')), 'w+')
    if args.name_prefix:
        args.name_prefix +='_'
    args.dataset =  args.metric.split('_')[0]
    config_figures = json.load(open(args.config_figures))
    args.timepoints = list(map(int, args.timepoints.split('-')))
    print ("### Loading Summary DF")
    if 'prc' in args.metric:
        x_handle = 'recalls'
        y_handle = 'precisions'
    elif 'roc' in args.metric:
        x_handle = 'fpr'
        y_handle = 'tpr'
    else:
        raise NotImplementedError

    stats = [(model_name, get_stats_from_output(res, log, model_name)) for res, log, model_name in zip(config_figures['results_folders'], config_figures['logs_folders'], config_figures['model_name'])]
    exclusion_intervals = [[k for k in sorted(el)] for _, el in stats]
    assert all([exclusion_intervals[0] == el for el in exclusion_intervals]), ("The experiments set have not been tested for the same exclusion intervals")

    fig, ax = plt.subplots(len(args.timepoints), len(exclusion_intervals[0]), figsize=[15,15])
    
    for idx, m in enumerate(args.timepoints):
        for subidx, ei in enumerate(exclusion_intervals[0]):
            p = []
            r = []    
            model = []

            for grid_stat in stats:
                try:
                    model_name, stat_pkl = grid_stat
                    stat = stat_pkl[ei][m]
                    assert stat[args.metric.format(m)][0] > 0
                    p.append(stat[args.metric.format(m) + "_curve"][0][y_handle])
                    r.append(stat[args.metric.format(m) + "_curve"][0][x_handle])
                    model.extend([model_name for _ in range(len(p[-1]))])
                except Exception as e:
                    ax[idx,subidx].axis('off')
                    continue

            if p and r:
                p = np.concatenate(p)
                r = np.concatenate(r)
                model = np.asarray(model)
                df = pd.DataFrame({y_handle:p, x_handle:r, "model":model})
                print ("row:{}-col:{} \t {}".format(m, ei, df.shape))
                sns.lineplot(x=x_handle, y=y_handle, hue='model', data=df, ax=ax[idx,subidx])
    pad = 3
    for colax, ei in zip(ax[0], exclusion_intervals[0]):
        colax.annotate("Exclusion interval : {} months".format(ei), xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')
    
    for rowax, t in zip(ax[:, 0], args.timepoints):
        rowax.annotate("Prediction interval : {} months".format(t), xy=(0, 0.5), 
                xytext=(-rowax.yaxis.labelpad - pad, 0),
                xycoords=rowax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')

    fig_path =  "figures/{}summary_performance_{}.png".format(args.name_prefix, args.metric.replace('{', '').replace('}', ''))
    print ("saving fig at:\t {}".format(fig_path))
    fig.savefig(fig_path, bbox_inches='tight')
    fig.savefig(fig_path.replace('.png', '.svg'), bbox_inches='tight', format='svg')
    
    sys.exit(0)

## python scripts/summarizer/generate_summary_plot.py --timepoints 3-6-12-36-60-120 --metric dev_{}month_auprc_c