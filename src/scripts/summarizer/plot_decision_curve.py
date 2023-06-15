import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import pickle as pkl
import sys
import os
import argparse
import json
from tqdm import tqdm
from collections import Counter
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
from pancnet.utils.eval import include_exam_and_determine_label
from pancnet.utils.parsing import md5


PREDICTION_INTERVALS = [3, 6, 12, 36, 60, 120]

plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=24)  # fontsize of the figure title

def get_probs_golds(test_preds, month=36):   #TODO this function should be moved inside pancnet in utils or eval
    probs_for_eval, golds_for_eval = [], []
    for prob_arr, censor_time, gold in tqdm(zip(test_preds["probs"], test_preds["censor_times"], test_preds["golds"])):
        index = PREDICTION_INTERVALS.index(month)
        include, label = include_exam_and_determine_label(index, censor_time, gold,  args.independent_eval)
        # include, label = include_exam_and_determine_label(index, censor_time, gold,  False)
        if include:
            probs_for_eval.append(prob_arr[index])
            golds_for_eval.append(label)
    return probs_for_eval, golds_for_eval


def get_confusion_matrix(df_slice):
    ct = Counter(df_slice['golds'])
    tp = ct[1.0]
    fp = ct[0.0]
    return tp, fp


def get_metrics(df_slice, total_n, total_p):
    """
    | TP | FN |
    | FP | TN |
    """
    predicted_postive = df_slice.shape[0]
    incidence = total_p / (total_p+total_n)
    
    tp, fp = get_confusion_matrix(df_slice)
    assert tp+fp == predicted_postive, (tp, fp, predicted_postive)
    fn = total_p - tp
    tn = total_n - fp
    if fn + tn == 0:  
        res = {
            'tpr':1, 'PPV':incidence, 
            'relative_risk':np.nan, 'odds_ratio':np.nan
        }
        return pd.Series(res)

    tpr = tp/total_p
    precision = tp/(tp+fp)
    relative_risk = precision/incidence
    if fp == 0: 
        odds_ratio = np.nan
    else:
        odds_ratio = (tp/fp)/(fn/tn)

    res = {
        'tp': tp, 'total_p': total_p, 'incidence': incidence,
        'tpr':tpr, 'PPV':precision, 
        'relative_risk':relative_risk, 'odds_ratio':odds_ratio
    }
    return pd.Series(res)


def compute_plot_dataframe(test_preds, month=36, plot_mode='log'):
    probs_for_eval, golds_for_eval = get_probs_golds(test_preds, month=month)
    df = pd.DataFrame({"probs": probs_for_eval, "golds": golds_for_eval})
    df = df.sort_values(by='probs', ascending=False)
    if plot_mode == 'percentile': # Percentile version
        plot_xs = range(100)
        probs_bins = np.quantile(df['probs'], np.linspace(0,1,100))
    elif plot_mode == 'log': # Log scale version
        # plot_xs = 10**np.arange(2,8)
        plot_xs = 250 * 2 ** np.arange(10)
        actual_xs = (plot_xs / 1e6 * df.shape[0]).astype(int)
        probs_bins = df['probs'].iloc[actual_xs]
    else:
        raise ValueError
    
    TOTAL_NUM_POS = df['golds'].sum()
    TOTAL_NUM_NEG = df.shape[0] - TOTAL_NUM_POS
    res_against_all_patients = {}
    for j in probs_bins:
        df_slice = df[df['probs']>j]
        mx = get_metrics(df_slice, total_n=TOTAL_NUM_NEG, total_p=TOTAL_NUM_POS)
        res_against_all_patients.update({df_slice.shape[0]: mx})
    # print(res_against_all_patients)
    res_df = pd.DataFrame.from_dict(res_against_all_patients, orient='index')
    res_df['xs'] = plot_xs
    if plot_mode == 'log':
        res_df['xs_actual'] = actual_xs
    return res_df


def generate_single_plot(panel_kwargs, fig_size, filepath_stem, plot_mode='log', ymax=400):

    # Create the plot
    f, ax = plt.subplots(figsize=fig_size)

    pred_paths = panel_kwargs['pred_paths']
    line_idx = 0
    res_dfs = {}
    for pred_path, model in zip(pred_paths, kwargs['models']):
        # Load the pred file
        if pred_path not in g_loaded_preds:
            g_loaded_preds['pred_path'] = pkl.load(open(pred_path, 'rb'))
            print ("Data loaded from {}...".format(pred_path))
        else:
            print ("Using cached data...")
        test_preds = g_loaded_preds['pred_path']

        for month in kwargs['prediction_intervals']:
            color = kwargs['colors'][line_idx]
            if color == 'md5':
                hue_hash = md5(f'{pred_path}:{month}')[-2:]
                hue = int(hue_hash, base=16)/255
                color = matplotlib.colors.hsv_to_rgb([hue, 0.6, 0.7])
            res_df = compute_plot_dataframe(test_preds=test_preds, month=month, plot_mode=plot_mode)
            sns.lineplot(data=res_df, x='xs', y='relative_risk', ax=ax, label=kwargs['legend'][line_idx], color=color)
            dat_path = filepath_stem.format(name.replace(' ', '_').lower())
            res_df.to_csv('{}_{}_{}.csv'.format(dat_path, model, month))
            if len(res_df['xs']) < 30:
                sns.scatterplot(data=res_df, x='xs', y='relative_risk', ax=ax, color=color)
            res_dfs[line_idx] = res_df.to_dict()
            line_idx += 1

    ax.set_xlabel('N at risk (per 1M patients)')
    ax.set_ylabel('Relative Risk')
    ax.set_ylim([0, ymax])
    yticks = np.linspace(0, ymax, 6)[1:]  # drop the first tick: 0
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{int(y)}x'for y in yticks])
    ax.set_title(name)
    ax.legend(fontsize='smaller')
    if plot_mode == 'log':
        ax.set_xscale('log')    

    f.tight_layout()
    filepath = filepath_stem.format(name.replace(' ', '_').lower())
    f.savefig("{}.png".format(filepath))
    f.savefig("{}.svg".format(filepath), format='svg')
    with open(filepath+'.json', 'w') as jf:
        json.dump(res_dfs, jf)
    print(f'The plot for "{name}" is generated at {filepath}.')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PancNet Grid Search Results Collector.')
    parser.add_argument("--experiment_config_path", required=False, type=str, default='configs/decision_curve_va01.json', help="Path of the master experiment config")
    parser.add_argument('--independent_eval', action='store_true', default=False, help='choose between indipendent or cumulative evalution')
    parser.add_argument('--plot_mode', type=str, default='log', help='Choose between ["log", "percentile"].')

    args = parser.parse_args()
    assert args.plot_mode in ["log", "percentile"]
    best_exp_ids_config = json.load(open(args.experiment_config_path, 'r'))

    fig_size = best_exp_ids_config['fig_size']
    filepath_stem = os.path.join(best_exp_ids_config["save_dir"], best_exp_ids_config['filename'])
    if "{}" not in filepath_stem:
        # add formatter to the end if not specified in the input
        filepath_stem += '-{}'
    g_loaded_preds = {}

    for name, kwargs in best_exp_ids_config['axs'].items():
        generate_single_plot(
            panel_kwargs=kwargs, 
            fig_size=fig_size,
            filepath_stem=filepath_stem,
            plot_mode=args.plot_mode,
            ymax=best_exp_ids_config['yaxis_max']
        )

# pred_path = 'logs_test_transformer_va3m/646397ef78b3aa2c49f96a80b2db1cfa.results.test_preds'
# preds = pkl.load(open(pred_path, 'rb'))

# mx = compute_plot_dataframe(preds, month=36)