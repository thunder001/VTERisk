## Takes as input a metric and the output folders of the grid to comprare
# 
import argparse
import os
import pickle as pkl
import csv
import json
import sys
import subprocess
from tqdm import tqdm
from os.path import dirname, realpath
import random
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.pylab as pylab
import seaborn as sns
import sklearn.metrics
from sklearn.metrics._ranking import _binary_clf_curve
from pancnet.utils.visualization import save_figure_and_subplots
from pancnet.utils.eval import include_exam_and_determine_label
from pancnet.utils.parsing import md5

plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=24)  # fontsize of the figure title

def get_probs_golds(test_preds, month='36'):

    probs_for_eval, golds_for_eval = [], []

    for prob_arr, censor_time, gold in tqdm(zip(test_preds["probs"], test_preds["censor_times"], test_preds["golds"])):
        index = args.timepoints.index(month)
        include, label = include_exam_and_determine_label(index, censor_time, gold,  args.indipendent_eval)
        if include:
            probs_for_eval.append(prob_arr[index])
            golds_for_eval.append(label)

    return probs_for_eval, golds_for_eval

def plot_curve(x_handle, y_handle, auc_handle, curve_type='roc', ax=None, c=None, operating_point=None, 
    highlight_coord=False, model_name='', score=None):
    
    dot_size = 22

    if curve_type=='roc':

        roc_df = pd.DataFrame({"True positive rate":y_handle, "False positive rate":x_handle})
        sns.lineplot(x='False positive rate', y='True positive rate', data=roc_df, ax=ax, label="{}: AUROC {:.3f}".format(model_name, auc_handle), color=c, ci=None)
        
        if operating_point is not None:
            xs_roc, ys_roc = operating_point

        if highlight_coord:
            ax.plot([xs_roc, xs_roc+0.2], [ys_roc, ys_roc+0.2], color='black', alpha=0.8)
            ax.text(xs_roc+0.2, ys_roc+0.2, "sensitivity:{:.1%}\nspecificity:{:.2%}".format(ys_roc, (1-xs_roc)), transform=ax.transData, ha='left', va='bottom')
        
        if operating_point is not None:
            ax.scatter(xs_roc, ys_roc, s=dot_size, c='red', zorder=10)
        
        return 0
    else:
        prc_df = pd.DataFrame({"Precision":y_handle, "Recall":x_handle})
        sns.lineplot(x='Recall', y='Precision',data=prc_df, ax=ax, label="{}: AUPRC {:.3f}".format(model_name, auc_handle), color=c, ci=None)
        

        if operating_point:
            xs_prc, ys_prc = operating_point
            ax.scatter(xs_prc, ys_prc, s=dot_size, c='red', zorder=10)
        
        if highlight_coord:
            ax.plot([xs_prc, xs_prc+0.1], [ys_prc, ys_prc+0.1], color='black', alpha=0.8)
            ax.text(xs_prc+0.1, ys_prc+0.1, "precision:{:.1%}\nrecall:{:.1%}".format(ys_prc, xs_prc), transform=ax.transData, ha='left', va='bottom')
        
        ax.set_ylim([0, 1])
        return 0

def load_preds(log_dir, exp_id, n_samples=1):
    try:
        test_preds_path = os.path.join(log_dir, "{}.results.test_preds".format(exp_id))
        test_preds = pkl.load(open(test_preds_path, 'rb'))
        test_preds = {k:v[::n_samples] for k,v in test_preds.items()}
        return test_preds
    except FileNotFoundError:
        print ("Missing {}".format(test_preds_path))
        return None
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PancNet Grid Search Results Collector.')
    parser.add_argument("--experiment_config_path", required=True, type=str, help="Path of the master experiment config")
    parser.add_argument('--highlight_model', type=str, default="Transformer", help='Specify main model to highlight.')
    parser.add_argument('--best_month', type=str, default='36', help='Specify maximum number of years to collect.')
    parser.add_argument('--indipendent_eval', action='store_true', default=False, help='choose between indipendent or cumulative evalution')
    parser.add_argument('--timepoints', type=str, default='3-6-12-36-60-120', help='choose between indipendent or cumulative evalution')
    parser.add_argument('--log_dir', type=str, default='output/28-01-2021-1703_transformer_logs', help='Specify maximum number of years to collect.')
    parser.add_argument('--fig_suffix', type=str, default='', help='Specify maximum number of years to collect.')
    parser.add_argument('--n_samples', type=int, default=1, help='Sample every N samples. [1 takes all samples]')
    parser.add_argument('--score', type=str, help='score to find the best threshold [f1, mcc or None]')

    args = parser.parse_args()

    best_exp_ids_config = json.load(open(args.experiment_config_path, 'r'))
    performance_table = pd.read_csv(best_exp_ids_config['performance_table'])
    if 'plot_barplot' not in best_exp_ids_config: best_exp_ids_config['plot_barplot'] = []
    best_exp_ids_config['filename'] = ""  if 'filename' not in best_exp_ids_config else best_exp_ids_config['filename'] + '_'
    args.timepoints = args.timepoints.split('-')
    recall_thresholds = [0.8, 0.6, 0.4, 0.2, 0.1, 0.05]
    # Axes Definition
    fig = plt.figure(figsize=[18,12])
    fig.suptitle('Model performance')
    gs = GridSpec(4, 6)
    ax_prc=fig.add_subplot(gs[:2,:2])
    ax_roc=fig.add_subplot(gs[:2,-2:])
    ax_prc.set(xlim=(-0.01, 1.01), title='Models - PRC')
    ax_roc.set(xlim=(-0.01, 1.01), title='Models - ROC')
    ax_roc.plot([0, 1], [0, 1], linestyle='--', color='grey', label='random guess (AUROC: 0.500)')
    assert len(best_exp_ids_config['plot_heatmap']) == 4
    heatmap_idx = [(0,2), (0,3), (1,2), (1,3)]
    heatmap_idx = dict(zip(best_exp_ids_config['plot_heatmap'], heatmap_idx))
    models = list(set(best_exp_ids_config['model_name']))
    blues = sns.color_palette("Blues", 12)[4:][::-1]
    colors = {'Bag-of-words': blues[4], 
        'MLP': blues[3], 
        'GRU': blues[2], 
        'Transformer': blues[1], 
        'Transformer - Prior knowledge disease': blues[3]}

    ax_prediction_interval=fig.add_subplot(gs[2:,:2])
    ax_prediction_interval.plot([0, 1], [0, 1], linestyle='--', color='grey', label='random guess (AUROC: 0.500)')
    ax_prediction_interval.set(title='Prediction intervals - ROC')

    ax_disease_code=fig.add_subplot(gs[2:,2:-2]) # PRIOR KNOWLEDGE DISEASE COMPARISON
    ax_disease_code.set(title='Disease codes - ROC') # PRIOR KNOWLEDGE DISEASE COMPARISON
    ax_disease_code.plot([0, 1], [0, 1], linestyle='--', color='grey', label='random guess (AUROC: 0.500)') # PRIOR KNOWLEDGE DISEASE COMPARISON

    ax_exclusion=fig.add_subplot(gs[2:,-2:]) # EXCLUSION INTERVAL COMPARISON
    ax_exclusion.set(title='Exclusion intervals - ROC') # EXCLUSION INTERVAL COMPARISON
    ax_exclusion.plot([0, 1], [0, 1], linestyle='--', color='grey', label='random guess (AUROC: 0.500)') # EXCLUSION INTERVAL COMPARISON

    barplot_records = []
    barplot_records_for_all_pred_intervals = []

    for i, (model_name, model_type) in enumerate(zip(best_exp_ids_config['model_name'], best_exp_ids_config['model_type'])):

        if "plot_main_ax" not in best_exp_ids_config or i in best_exp_ids_config['plot_main_ax']:
            exclusion = model_type.split('-')[0].strip()

            print (f"Starting plotting for {model_name} - {model_type}")
            
            if model_name == args.highlight_model:
                c = 'orange'
            else:
                c = colors[model_name]
            
            experiment_metrics = performance_table[(performance_table.Model == model_name) & \
                (performance_table["Exclusion Interval"] == int(exclusion.split()[1])) & \
                (performance_table["Prediction Interval"] == int(args.best_month))]
            
            exp_id, month = experiment_metrics[['exp_id', 'Prediction Interval']].iloc[0]
            dict_metrics = dict(zip(experiment_metrics.Metric, experiment_metrics.Median))
            dict_metrics = {k:float(v) if k!='curves' else json.loads(v) for k,v in dict_metrics.items()}
            dict_metrics['curves'] = {k:np.array(v) for k,v in dict_metrics['curves'].items()}
            incidence = np.round(dict_metrics['incidence'], 4)
            
            if "Prior knowledge" in model_type and args.highlight_model in model_name:
                plot_curve(dict_metrics["curves"]["fpr"], 
                    dict_metrics["curves"]["tpr"], 
                    auc_handle=dict_metrics["auroc"], 
                    curve_type='roc', 
                    c=c,
                    ax=ax_disease_code, 
                    model_name=model_name)
                continue

            if any([e in model_type for e in ["Exclusion 3", "Exclusion 6", "Exclusion 12"]]):
                print (f"Running {model_name} - {model_type}")

                if model_name == "GRU":
                    plot_curve(dict_metrics["curves"]["fpr"], 
                        dict_metrics["curves"]["tpr"], 
                        auc_handle=dict_metrics["auroc"],
                        curve_type='roc', ax=ax_exclusion,
                        c = blues[["Exclusion 3", "Exclusion 6", "Exclusion 12"].index(exclusion) + 1 ],
                        model_name="{}-{}".format(exclusion, model_name))

                for rec in recall_thresholds:
                    record = (rec, dict_metrics["curves"]["precision"][dict_metrics["curves"]["recall"]>rec][0], \
                        dict_metrics["curves"]["recall"][dict_metrics["curves"]["recall"]>rec][0], \
                        dict_metrics["curves"]["odds_ratio"][dict_metrics["curves"]["recall"]>rec][0], \
                        model_name, exclusion, i in best_exp_ids_config['plot_barplot'])
                    barplot_records.append(record)
                continue

            if i==0:
                ax_prc.axhline(incidence, linestyle='--', color='grey', label='random guess (AUPRC: {})'.format(incidence))

            show_coord = False#model_name in args.highlight_model

            plot_curve(dict_metrics["curves"]["recall"], 
                dict_metrics["curves"]["precision"], 
                auc_handle=dict_metrics["auprc"], 
                curve_type='prc', ax=ax_prc,
                #operating_point=(dict_metrics["recall"], dict_metrics["precision"]), 
                c=c,
                highlight_coord=show_coord, 
                model_name=model_name)

            plot_curve(dict_metrics["curves"]["fpr"], 
                dict_metrics["curves"]["tpr"], 
                auc_handle=dict_metrics["auroc"],
                curve_type='roc', ax=ax_roc,
                #operating_point=(dict_metrics["fpr"], dict_metrics["tpr"]),
                c=c,
                highlight_coord=show_coord,
                model_name="{}-{}".format(exclusion, model_name))

            tn, fp, fn, tp = dict_metrics["curves"]["cm"]#tns[best_threshold], fps[best_threshold], fns[best_threshold], tps[best_threshold]
            tp_ = dict_metrics["precision"]#tp / (tp+fp) #precision
            fp_ = (1-tp_)#fp / (tp+fp) #(1-precision)
            tn_ = (1-dict_metrics["fpr"])*(1-dict_metrics["incidence"]) \
                    /((1-dict_metrics["fpr"])*(1-dict_metrics["incidence"]) \
                    +(1-dict_metrics["recall"])*(dict_metrics["incidence"]))#tn / (tn+fn) #()
            fn_ = 1-tn_#fn / (tn+fn)

            if i in best_exp_ids_config['plot_heatmap']:
                df = pd.DataFrame([[tp_, fn_], [fp_, tn_]], index=["+", "-"],
                               columns=["+","-"])
                xc, yc = heatmap_idx[i]
                heatmap_ax = fig.add_subplot(gs[xc, yc])
                sns.heatmap(df, annot=True, cbar=False, fmt=".1%", linecolor='black', linewidths=1, ax=heatmap_ax, cmap="mako", vmin=-10, vmax=0)
                heatmap_ax.xaxis.tick_top()
                heatmap_ax.set(xlabel='Predicted', ylabel='Observed')
                heatmap_ax.set_title('{}'.format(model_name),fontweight="bold")

            for rec in recall_thresholds:
                record = (rec, dict_metrics["curves"]["precision"][dict_metrics["curves"]["recall"]>rec][0], \
                    dict_metrics["curves"]["recall"][dict_metrics["curves"]["recall"]>rec][0], \
                    dict_metrics["curves"]["odds_ratio"][dict_metrics["curves"]["recall"]>rec][0], \
                    model_name, exclusion, i in best_exp_ids_config['plot_barplot'])
                barplot_records.append(record)

            if model_name == args.highlight_model and "Exclusion 0" in model_type:
                for i, month in enumerate(['3','6','12','36','60']):
                    print (f"Running {month} month {model_name}")
                    experiment_metrics = performance_table[(performance_table.Model == model_name) & \
                        (performance_table["Exclusion Interval"] == int(exclusion.split()[1])) & \
                        (performance_table["Prediction Interval"] == int(month))]
                    exp_id, _ = experiment_metrics[['exp_id', 'Prediction Interval']].iloc[0]
                    dict_metrics = dict(zip(experiment_metrics.Metric, experiment_metrics.Median))
                    dict_metrics = {k:float(v) if k!='curves' else json.loads(v) for k,v in dict_metrics.items()}
                    dict_metrics['curves'] = {k:np.array(v) for k,v in dict_metrics['curves'].items()}
                    
                    if month==args.best_month:
                        c='orange'
                        plot_curve(dict_metrics["curves"]["fpr"], 
                            dict_metrics["curves"]["tpr"], 
                            auc_handle=dict_metrics["auroc"],
                            curve_type='roc', 
                            c=c,
                            ax=ax_disease_code, model_name=model_name)
                            
                        plot_curve(dict_metrics["curves"]["fpr"], 
                            dict_metrics["curves"]["tpr"], 
                            auc_handle=dict_metrics["auroc"],
                            curve_type='roc', ax=ax_exclusion, 
                            c=c,
                            model_name="Exclusion 0-{}".format(model_name))
                    else:
                        c = blues[i+1]

                    plot_curve(dict_metrics["curves"]["fpr"], 
                        dict_metrics["curves"]["tpr"], 
                        auc_handle=dict_metrics["auroc"],
                        curve_type='roc', 
                        c=c,
                        ax=ax_prediction_interval, 
                        model_name=f"Prediction interval {month}")

                    for rec in recall_thresholds:
                        record = (int(month), rec, dict_metrics["curves"]["precision"][dict_metrics["curves"]["recall"]>rec][0], \
                            dict_metrics["curves"]["recall"][dict_metrics["curves"]["recall"]>rec][0], \
                            dict_metrics["curves"]["odds_ratio"][dict_metrics["curves"]["recall"]>rec][0], \
                            model_name, exclusion, i in best_exp_ids_config['plot_barplot'])
                        barplot_records_for_all_pred_intervals.append(record)


    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    figname = "figures/{}figure2{}".format(best_exp_ids_config['filename'], args.fig_suffix)
    save_figure_and_subplots(figname, fig, format='svg')
    save_figure_and_subplots(figname, fig, format='png')


    df = pd.DataFrame.from_records(barplot_records, 
        columns=["Recall (%)", "Precision", "Recall", "Odds Ratio", "Model", "Exclusion", "In comparison plot"])
    df["Precision"] = df.Precision*100
    df["Recall (%)"] = df["Recall (%)"]*100
    #df['Exclusion'] = pd.Categorical(df['Exclusion'], ["Exclusion 0", "Exclusion 3", "Exclusion 6", "Exclusion 12"])
    #df = df.sort_values(by="Exclusion")
    exclusion_order = {'Exclusion 0': 0, 'Exclusion 3': 1, 'Exclusion 6': 2, 'Exclusion 12': 3}
    model_order = {'bag of words': 0, 'mlp': 1, 'gru': 2, 'transformer': 3}
    model_order.update(exclusion_order)
    df = df.loc[df['Recall (%)'] != 5.0]  # manually removing the 5% recall entries
    df = df.sort_values(by="Exclusion", key=lambda x: x.map(exclusion_order))
    df = df.sort_values(by=["Model", "Exclusion"], key=lambda x: x.map(model_order))
    df.to_csv("figures/{}performance_boxplot.tsv".format(best_exp_ids_config['filename']), sep='\t', index=False)
    highlight_idx = [m in args.highlight_model for m in df.Model]
    not_exclusion_0_idx = df['Exclusion'].isin(['Exclusion 3', 'Exclusion 6', 'Exclusion 12'])
    if any(not_exclusion_0_idx):
        fig = plt.figure(figsize=[8,8])
        fig.suptitle('Model precision recall')
        gs = GridSpec(1,1)
        ax_bar=fig.add_subplot(gs[:,:])
        sns.barplot(
            data=df[highlight_idx & not_exclusion_0_idx],
            x="Exclusion",
            y="Precision", ax=ax_bar,
            hue='Recall (%)',
            palette=sns.color_palette("crest"),
            edgecolor=".2")
        ax_bar.set(ylabel="Precision (%)")
        ax_bar.axhline(incidence, linestyle='--', color='grey', label='random guess: {}'.format(incidence))
        plt.savefig("figures/{}figure2{}_barplot_pr.png".format(best_exp_ids_config['filename'], args.fig_suffix), bbox_inches='tight')
        plt.savefig("figures/{}figure2{}_barplot_pr.svg".format(best_exp_ids_config['filename'],  args.fig_suffix), bbox_inches='tight', format='svg')

        fig = plt.figure(figsize=[8,8])
        fig.suptitle('Odds ratio for different exclusions')
        gs = GridSpec(1,1)
        ax_bar=fig.add_subplot(gs[:,:])
        sns.barplot(
            data=df[highlight_idx & not_exclusion_0_idx],
            x="Exclusion",
            y="Odds Ratio", ax=ax_bar,
            hue='Recall (%)',
            palette=sns.color_palette("crest"),
            edgecolor=".2")
        ax_bar.set(ylabel="Odds Ratio", ylim=(0, 60))  # 60 for RPDR and 30 for DNPR
        ax_bar.axhline(1, linestyle='--', color='grey', label='random guess: 1')
        plt.savefig("figures/{}figure2{}_barplot_or.png".format(best_exp_ids_config['filename'], args.fig_suffix), bbox_inches='tight')
        plt.savefig("figures/{}figure2{}_barplot_or.svg".format(best_exp_ids_config['filename'], args.fig_suffix), bbox_inches='tight', format='svg')

    fig = plt.figure(figsize=[8, 8])
    fig.suptitle('Odds ratio for different models')
    gs = GridSpec(1, 1)
    ax_bar = fig.add_subplot(gs[:, :])
    sns.barplot(
        data=df[df["In comparison plot"]],
        x="Model",
        y="Odds Ratio", ax=ax_bar,
        hue='Recall (%)',
        palette=sns.color_palette("crest"),
        edgecolor=".2")
    ax_bar.set(ylabel="Odds Ratio", ylim=(0, 40)) # 200 for independent training and 40 for cross-eval
    ax_bar.axhline(1, linestyle='--', color='grey', label='random guess: 1')
    plt.savefig("figures/{}figure2_barplot_comparison.png".format(best_exp_ids_config['filename']), bbox_inches='tight')
    plt.savefig("figures/{}figure2_barplot_comparison.svg".format(best_exp_ids_config['filename']), bbox_inches='tight', format='svg')


    df = pd.DataFrame.from_records(barplot_records_for_all_pred_intervals, 
        columns=["Month", "Recall (%)", "Precision", "Recall", "Odds Ratio", "Model", "Exclusion", "In comparison plot"])
    df['Exclusion'] = pd.Categorical(df['Exclusion'], ["Exclusion 0", "Exclusion 3", "Exclusion 6", "Exclusion 12"])
    df.to_csv("figures/{}performance_boxplot_2.tsv".format(best_exp_ids_config['filename']), sep='\t', index=False)
    df = df.loc[df['Recall (%)'] != 0.05]  # manually removing the 5% recall entries
    df["Precision"] = df.Precision*100
    df["Recall (%)"] = df["Recall (%)"]*100
    df = df.sort_values(by="Exclusion")
    highlight_idx = [m in args.highlight_model for m in df.Model]
    fig = plt.figure(figsize=[8,8])
    fig.suptitle('Odds ratio for the best model')
    gs = GridSpec(1,1)
    ax_bar=fig.add_subplot(gs[:,:])
    sns.barplot(
        data=df[highlight_idx],
        x="Month",
        y="Odds Ratio", ax=ax_bar,
        hue='Recall (%)',
        palette=sns.color_palette("crest"),
        edgecolor=".2")
    ax_bar.set(ylabel="Odds Ratio")
    ax_bar.axhline(1, linestyle='--', color='grey', label='random guess: 1')
    plt.savefig("figures/{}figure2{}_barplot_predint.png".format(best_exp_ids_config['filename'], args.fig_suffix), bbox_inches='tight')
    plt.savefig("figures/{}figure2{}_barplot_predint.svg".format(best_exp_ids_config['filename'], args.fig_suffix), bbox_inches='tight', format='svg')

# python scripts/summarizer/plot_model_performance.py \
# --experiment_config_path configs/figures/model_performances.json \
# --best_month 60 --n_samples 10 --highlight_model Transformer