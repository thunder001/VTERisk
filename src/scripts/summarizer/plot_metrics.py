import os

import matplotlib
import pandas as pd
import numpy as np
import scipy.stats as sps
import math

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle as pkl
import argparse


def plot_auc(fprs, tprs, aucs):
    classes = list(fprs.keys())
    plt.figure(figsize=(10, 7))
    lines = []
    labels = []
    for cls in classes:
        # print(cls)
        # print(fprs[cls])
        line, = plt.plot(fprs[cls], tprs[cls], lw=1)
        lines.append(line)
        labels.append('{0:s} month: (AUC={1:.2f})'.format(cls, aucs[cls][0]))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False positive rate', size=20)
    plt.ylabel('True positive rate', size=20)
    plt.legend(lines, labels, loc='best', prop=dict(size=12))
    plt.title("ROC curve for transformer model", size=16)

def plot_auc_paper(fprs, tprs, aucs):
    classes = list(fprs.keys())
    plt.figure(figsize=(6, 6))
    lines = []
    labels = []
    for cls in classes:
        # print(cls)
        # print(fprs[cls])
        line, = plt.plot(fprs[cls], tprs[cls], lw=1)
        lines.append(line)
        labels.append('{0:s} month: (AUC={1:.3f})'.format(cls, aucs[cls][0]))
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='random guess (AUROC: 0.500)')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False positive rate', size=20)
    plt.ylabel('True positive rate', size=20)
    plt.legend(lines, labels, loc='best', prop=dict(size=14))
    plt.title("'Models - ROC'", size=24)


def plot_prs(precisions, recalls, auprcs):
    classes = list(precisions.keys())
    plt.figure(figsize=(10, 7))
    lines = []
    labels = []
    for cls in classes:
        # print(cls)
        # print(fprs[cls])
        line, = plt.plot(precisions[cls], recalls[cls], lw=1)
        lines.append(line)
        labels.append('{0:s} month: (AUPRC={1:.2f})'.format(cls, auprcs[cls][0]))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Precision', size=20)
    plt.ylabel('Recall', size=20)
    plt.legend(lines, labels, loc='best', prop=dict(size=12))
    plt.title("Precision-Recall curve for transformer model", size=16)


def make_auc_tb(results_dir):
    files = os.listdir(results_dir)
    stats_files = [file for file in files if any(x in file for x in ['cross_eval_stats', 'test_stats'])]
    # Make primary auc table
    aucs = dict()
    auprcs = dict()
    auc_keys = ['test_3month_auroc_c', 'test_6month_auroc_c', 'test_12month_auroc_c',
                'test_36month_auroc_c', 'test_60month_auroc_c']
    auprc_keys = ['test_3month_auprc_corrected_c', 'test_6month_auprc_corrected_c',
                  'test_12month_auprc_corrected_c', 'test_36month_auprc_corrected_c',
                  'test_60month_auprc_corrected_c']
    new_keys = ['3_month', '6_month', '12_month', '36_month', '60_month']

    for file in stats_files:
        file_path = os.path.join(results_dir, file)
        with open(file_path, 'rb') as f:
            stats = pkl.load(f)
        for new_key, key in zip(new_keys, auc_keys):
            auc = stats[key][0]
            if new_key not in aucs.keys():
                aucs[new_key] = []
                aucs[new_key].append(auc)
            else:
                aucs[new_key].append(auc)
        for new_key, key in zip(new_keys, auprc_keys):
            auprc = stats[key][0]
            if new_key not in auprcs.keys():
                auprcs[new_key] = []
                auprcs[new_key].append(auprc)
            else:
                auprcs[new_key].append(auprc)
    auc_tb = pd.DataFrame(aucs)
    auprc_tb = pd.DataFrame(auprcs)
    # Compute some basic stats
    mean = pd.Series(auc_tb.mean(axis=0), name='mean')
    std = pd.Series(auc_tb.std(axis=0), name='std')
    confs = sps.norm.interval(0.95, loc=mean, scale=std / math.sqrt(auc_tb.shape[0]))
    confs_95 = pd.Series(list(zip(np.round(confs[0], 3), np.round(confs[1], 3))), name='conf_95', index=mean.index)
    sts = pd.concat([mean, std, confs_95], axis=1).transpose()
    mean_pr = pd.Series(auprc_tb.mean(axis=0), name='mean')
    std_pr = pd.Series(auprc_tb.std(axis=0), name='std')
    confs_pr = sps.norm.interval(0.95, loc=mean_pr, scale=std_pr / math.sqrt(auprc_tb.shape[0]))
    confs_95_pr = pd.Series(list(zip(np.round(confs_pr[0], 3), np.round(confs_pr[1], 3))),
                            name='conf_95', index=mean_pr.index)
    sts_pr = pd.concat([mean_pr, std_pr, confs_95_pr], axis=1).transpose()
    # Concatenate primary and stats
    auc_tb = pd.concat([auc_tb, sts])
    auprc_tb = pd.concat([auprc_tb, sts_pr])

    return auc_tb, auprc_tb


def main():
    parser = argparse.ArgumentParser(description='Plot model metrics')
    # parser.add_argument('--result_path', type=str,
    #                     default=r'logs_test_va3m_epoch_10_w8_batch_8008_devlen10random_nocont\part00_epoch_10_w8_batch_8008_devlen10.results.part_07.test_stats',
    #                     help="Path of model evaluation results")
    # parser.add_argument('--roc_path', type=str,
    #                     default=r'G:\FillmoreCancerData\chunlei\pancpred\result\test\figures\transformer_roc_w8_batch_8008_devlen10random_nocont_part_07.png',
    #                     help="Path for saving figures")
    # parser.add_argument('--pr_path', type=str,
    #                     default=r'G:\FillmoreCancerData\chunlei\pancpred\result\test\figures\transformer_pr_w8_batch_8008_devlen10random_nocont_part_07.png',
    #                     help="Path for saving figures")

    # parser.add_argument('--result_path', type=str,
    #                     default=r'G:\FillmoreCancerData\chunlei\pancpred\code\pancnet\logs_cross_eval_va100k_686\7b7c31a4096a74d80322d4631a4bcdb1.results.cross_eval_stats',
    #                     help="Path of model evaluation results")
    # parser.add_argument('--roc_path', type=str,
    #                     default=r'G:\FillmoreCancerData\chunlei\pancpred\result\cross_evaluation\figures\roc_686_wo3_part01.png',
    #                     help="Path for saving figures")
    # parser.add_argument('--pr_path', type=str,
    #                     default=r'G:\FillmoreCancerData\chunlei\pancpred\result\cross_evaluation\figures\pr_686_wo3_part_01.png',
    #                     help="Path for saving figures")

    # parser.add_argument('--result_path', type=str,
    #                     default=r'logs_test_va1m_part0_whole_epoch_10_w1_batch_8008_devlen10\part0_epoch_10_w2_batch_8008_devlen10.results.test_stats',
    #                     help="Path of model evaluation results")
    # parser.add_argument('--roc_path', type=str,
    #                     default=r'G:\FillmoreCancerData\chunlei\pancpred\result\test\figures\transformer_auroc_part0_whole_epoch_10_w2_batch_8008_devlen10.png',
    #                     help="Path for saving figures")
    # parser.add_argument('--pr_path', type=str,
    #                     default=r'G:\FillmoreCancerData\chunlei\pancpred\result\test\figures\transformer_auprc_part0_whole_epoch_10_w2_batch_8008_devlen10.png',
    #                     help="Path for saving figures")

    # parser.add_argument('--result_path', type=str,
    #                     default=r'logs_test_va1m_part1_whole_epoch_10_w1_batch_8008_devlen10\part1_epoch_10_w1_batch_8008_devlen10.results.test_stats',
    #                     help="Path of model evaluation results")
    # parser.add_argument('--roc_path', type=str,
    #                     default=r'G:\FillmoreCancerData\chunlei\pancpred\result\test\figures\transformer_auroc_part1_whole_epoch_10_w1_batch_8008_devlen10.png',
    #                     help="Path for saving figures")
    # parser.add_argument('--pr_path', type=str,
    #                     default=r'G:\FillmoreCancerData\chunlei\pancpred\result\test\figures\transformer_auprc_part1_whole_epoch_10_w1_batch_8008_devlen10.png',
    #                     help="Path for saving figures")
    # parser.add_argument('--result_path', type=str,
    #                     default=r'logs_cross_eval_va100k_dnpr_test\a31f332f314f45511db503996b98752f.results.test_stats',
    #                     help="Path of model evaluation results")
    # parser.add_argument('--roc_path', type=str,
    #                     default=r'G:\FillmoreCancerData\chunlei\pancpred\result\cross_evaluation\dnpr_gru\figures\auroc_part1.png',
    #                     help="Path for saving figures")
    # parser.add_argument('--pr_path', type=str,
    #                     default=r'G:\FillmoreCancerData\chunlei\pancpred\result\cross_evaluation\dnpr_gru\figures\auprc_part1.png',
    #                     help="Path for saving figures")
    

    # parser.add_argument('--result_path', type=str,
    #                     default=r'logs_cross_eval_dnpr_transformer_va1m_part0\5c47b9cdabb3c4b86c39eceac8ee92e3.results.test_stats',
    #                     help="Path of model evaluation results")
    # parser.add_argument('--roc_path', type=str,
    #                     default=r'G:\FillmoreCancerData\chunlei\pancpred\result\cross_evaluation\dnpr_gru_3\figures\auroc_part0.png',
    #                     help="Path for saving figures")
    # parser.add_argument('--pr_path', type=str,
    #                     default=r'G:\FillmoreCancerData\chunlei\pancpred\result\cross_evaluation\dnpr_gru_3\figures\auprc_part0.png',
    #                     help="Path for saving figures")

    # parser.add_argument('--result_path', type=str,
    #                     default=r'logs_cross_eval_dnpr_transformer_va1m_part0\1b8d665eb7958715fbd9ea80719b83e8.results.test_stats',
    #                     help="Path of model evaluation results")
    # parser.add_argument('--roc_path', type=str,
    #                     default=r'G:\FillmoreCancerData\chunlei\pancpred\result\cross_evaluation\dnpr_transformer_3\figures\auroc_part0.png',
    #                     help="Path for saving figures")
    # parser.add_argument('--pr_path', type=str,
    #                     default=r'G:\FillmoreCancerData\chunlei\pancpred\result\cross_evaluation\dnpr_transformer_3\figures\auprc_part0.png',
    #                     help="Path for saving figures")

    parser.add_argument('--result_path', type=str,
                        default=r'logs_test_transformer_va3m\646397ef78b3aa2c49f96a80b2db1cfa.results.test_stats',
                        help="Path of model evaluation results")
    parser.add_argument('--roc_path', type=str,
                        default=r'G:\FillmoreCancerData\chunlei\pancpred\code\pancnet\results\test\transformer\figures\auroc_3m.png',
                        help="Path for saving figures")
    parser.add_argument('--pr_path', type=str,
                        default=r'G:\FillmoreCancerData\chunlei\pancpred\code\pancnet\results\test\transformer\figures\auprc_3m.png',
                        help="Path for saving figures")
    
    
    
    args = parser.parse_args()
    test_stats_fname = args.result_path
    roc_path = args.roc_path
    pr_path = args.pr_path

    with open(test_stats_fname, 'rb') as f:
        stats = pkl.load(f)

    # Plot ROC curve
    rocc_keys = ['test_3month_auroc_c_curve', 'test_6month_auroc_c_curve', 'test_12month_auroc_c_curve',
                  'test_36month_auroc_c_curve', 'test_60month_auroc_c_curve']
    auc_keys = ['test_3month_auroc_c', 'test_6month_auroc_c', 'test_12month_auroc_c',
                'test_36month_auroc_c', 'test_60month_auroc_c']
    new_keys = ['3', '6', '12', '36', '60']

    fprs = {}
    tprs = {}
    aucs = {}
    for new_key, key in zip(new_keys, rocc_keys):
        fprs[new_key] = stats[key][0]['fpr']
        tprs[new_key] = stats[key][0]['tpr']
    for new_key, key in zip(new_keys, auc_keys):
        aucs[new_key] = stats[key]

    plot_auc(fprs, tprs, aucs)
    plt.savefig(roc_path)

    # Plot Precision-Recall curve
    # prcs_keys = ['test_3month_auprc_corrected_c_curve', 'test_6month_auprc_corrected_c_curve',
    #              'test_12month_auprc_corrected_c_curve', 'test_36month_auprc_corrected_c_curve',
    #              'test_60month_auprc_corrected_c_curve']
    # auprc_keys = ['test_3month_auprc_corrected_c', 'test_6month_auprc_corrected_c',
    #               'test_12month_auprc_corrected_c', 'test_36month_auprc_corrected_c',
    #               'test_60month_auprc_corrected_c']
    prcs_keys = ['test_3month_auprc_c_curve', 'test_6month_auprc_c_curve',
                 'test_12month_auprc_c_curve', 'test_36month_auprc_c_curve',
                 'test_60month_auprc_c_curve']
    auprc_keys = ['test_3month_auprc_c', 'test_6month_auprc_c',
                  'test_12month_auprc_c', 'test_36month_auprc_c',
                  'test_60month_auprc_c']
    precisions = {}
    recalls = {}
    auprcs = {}
    for new_key, key in zip(new_keys, prcs_keys):
        precisions[new_key] = stats[key][0]['precisions']
        recalls[new_key] = stats[key][0]['recalls']
    for new_key, key in zip(new_keys, auprc_keys):
        auprcs[new_key] = stats[key]

    plot_prs(precisions, recalls, auprcs)
    plt.savefig(pr_path)

    # results_dir = r'G:\FillmoreCancerData\chunlei\pancpred\code\pancnet\logs_cross_eval_va100k_686'
    # auc_path = r'G:\FillmoreCancerData\chunlei\pancpred\result\cross_evaluation\cross_eval_summary_auc.csv'
    # auprc_path = r'G:\FillmoreCancerData\chunlei\pancpred\result\cross_evaluation\cross_eval_summary_auprc.csv'
    # results_dir = r'G:\FillmoreCancerData\chunlei\pancpred\code\pancnet\logs_test_va1m_part00_epoch_20_w6_batch_4004_devlen200_at_2'
    # save_path = r'G:\FillmoreCancerData\chunlei\pancpred\result\test\test_part0_transformer_summary.csv'
    # results_dir = r'G:\FillmoreCancerData\chunlei\pancpred\code\pancnet\logs_test_va1m_part0_epoch_10_w6_batch_8008_devlen10_at'
    # save_path = r'G:\FillmoreCancerData\chunlei\pancpred\result\test\test_part0_transformer_summary_dev10.csv'
    # results_dir = r'G:\FillmoreCancerData\chunlei\pancpred\code\pancnet\logs_test_va3m_epoch_10_w8_batch_8008_devlen10random_cont'
    # save_path = r'G:\FillmoreCancerData\chunlei\pancpred\result\test\test_va3m_transformer_cont_summary_dev10random.csv'
    # results_dir = r'G:\FillmoreCancerData\chunlei\pancpred\code\pancnet\logs_test_va1m_part0_whole_epoch_10_w1_batch_8008_devlen10'
    # auc_path = r'G:\FillmoreCancerData\chunlei\pancpred\result\test\part_0_whole_transformer_w1_summary_aupuc.csv'
    # auprc_path = r'G:\FillmoreCancerData\chunlei\pancpred\result\test\part_0_whole_transformer_w1_summary_auprc.csv'
    #
    # auc_tb, auprc_tb = make_auc_tb(results_dir)
    # auc_tb.to_csv(auc_path)
    # auprc_tb.to_csv(auprc_path)



if __name__ == '__main__':
    main()

# result_path = r'G:\FillmoreCancerData\chunlei\pancpred\code\pancnet\logs_test_va3m_epoch_10_w8_batch_8008_devlen10random_cont\part0_epoch_10_w8_batch_8008_devlen10.results'
# with open(result_path, 'rb') as f:
#     results = pkl.load(f)
