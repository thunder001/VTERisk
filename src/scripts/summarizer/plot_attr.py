import json
from collections import Counter
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd

pd.options.display.max_colwidth = 100
import matplotlib

matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import os
import argparse
from os.path import dirname, realpath

sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
from disrisknet .utils.parsing import CODE2DESCRIPTION, get_code, parse_args
from disrisknet.datasets.bth_disease_progression import PANC_CANCER_CODE, ICD10_PANC_CANCER, \
    BASELINE_DISEASES, END_OF_TIME_DATE


def normalize_attr(row):
    # return np.log1p(row['attribution'])*code_freq_cancer[row['code']]
    norm_by_all_codes = row['attribution'] * code_freq_all[row['code']]  # /sum(code_freq_cancer.values())
    norm_by_pc_codes = row['attribution'] * code_freq_cancer[row['code']]  # /sum(code_freq_cancer.values())
    return norm_by_all_codes  # , norm_by_pc_codes


def add_code_frew(row):
    freq_all = code_freq_all[row['code']]  # /sum(code_freq_cancer.values())
    freq_pc = code_freq_cancer[row['code']]  # /sum(code_freq_cancer.values())
    return freq_all, freq_pc


def process_df(df):
    # remove non diagnoses
    df = df[~(df.code.str.contains("DVA") | df.code.str.contains("DUH") | df.code.str.contains(
        "DUA") | df.code.str.contains("Age"))]
    df['weighted contribution'] = df.apply(lambda row: normalize_attr(row), axis=1)
    df['code_freq_1'] = df.apply(lambda row: add_code_frew(row)[0], axis=1)
    df['code_freq_2'] = df.apply(lambda row: add_code_frew(row)[1], axis=1)
    df['description'] = df.code.apply(code_fn)
    ages = np.array(["Age-{}".format(a) for a in range(120)])
    ages = np.append(ages, np.array(["Add-Age-{}".format(a) for a in range(120)]))
    ages = np.append(ages, np.array(["Scale-Age-{}".format(a) for a in range(120)]))
    ages = np.append(ages, np.array(["Combined-Age-{}".format(a) for a in range(120)]))
    age_df = df[df.code.isin(ages)]
    codes_df = df[~df.code.isin(ages)]
    return codes_df, age_df


def plot_attribution(df, lower_inc=1e-05, best=True):
    if best:
        print(df[df.code_freq_1 > lower_inc].sort_values(by='attribution')[-30:])
    else:
        print(df[df.code_freq_1 > lower_inc].sort_values(by='attribution')[:30])


def plot_attribution_vs_freq(df, suffix='tmp'):
    df['significant'] = False
    intercept = df.sort_values(by='weighted contribution', ascending=False)['weighted contribution'].iloc[20]
    df.loc[df['weighted contribution'] > intercept, 'significant'] = True
    df['baseline_disease'] = df.code.isin(baseline_disease)

    fig = plt.figure(figsize=[6, 10])
    for i in range(1, 3):
        ax = fig.add_subplot(2, 1, i)
        subset = df[(df.significant == False) & (df.baseline_disease == False)]
        ax.scatter(x=subset['code_freq_' + str(i)].values, y=subset.attribution.values, c='dimgrey', alpha=0.3)
        subset = df[(df.significant == False) & (df.baseline_disease == True)]
        ax.scatter(x=subset['code_freq_' + str(i)].values, y=subset.attribution.values, c='salmon', alpha=0.6)
        subset = df[(df.significant == True) & (df.baseline_disease == False)]
        ax.scatter(x=subset['code_freq_' + str(i)].values, y=subset.attribution.values, label='All diseases', c='black',
                   alpha=0.7)
        subset = df[(df.significant == True) & (df.baseline_disease == True)]
        ax.scatter(x=subset['code_freq_' + str(i)].values, y=subset.attribution.values,
                   label='Prior knowledge diseases', c='red', alpha=0.7)
        ax.set_title('Attribution - code frequency (for {} patients)'.format(["all", "cancer"][i - 1]))

    plt.ylim(-0.02, 0.05)
    plt.xlim(-0.01, 7e04)
    xs = np.linspace(1e-10, 7e04)
    ys = intercept / xs
    ax.plot(xs, ys, linestyle='--', color='grey', label='Threshold for weighted contribution')
    if '0' in suffix:
        title = 'Disease contribution without data exclusion'
    elif '3' in suffix:
        title = 'Disease contribution with 3 month data exclusion'
    elif '6' in suffix:
        title = 'Disease contribution with 6 month data exclusion'
    else:
        title = ''

    ax.set(xlabel='Code count', ylabel='Contribution', title=title)
    ax.legend()
    plt.savefig(f"figures/attribution_frequency{suffix}.png")
    plt.savefig(f"figures/attribution_frequency{suffix}.svg", bbox_inches='tight', format='svg')
    plt.tight_layout()
    plt.close()


def plot_age_attribution(df, suffix='tmp'):
    plt.close()

    fig = plt.figure(figsize=[6, 4])
    ax = fig.add_subplot(1, 1, 1)
    ages = np.array(["Combined-Age-{}".format(a) for a in range(120)])
    df = df[df.code.isin(ages)]
    strip_age = lambda x: x.split('-')[-1]
    df["age"] = df.code.apply(strip_age).astype('int32')
    df = df.sort_values(by='age')
    df = df.set_index('age')
    df = df.groupby(df.index // 5).mean()
    sns.barplot(data=df, x=df.index * 5, y='attribution', ax=ax, color='mediumseagreen')
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % 2 != 0:
            label.set_visible(False)
    ax.set(xlabel='Age', ylabel='Contribution', title="Age contribution")
    plt.tight_layout()
    plt.savefig(f"figures/age_attribution{suffix}.png")
    plt.savefig(f"figures/age_attribution{suffix}.svg", bbox_inches='tight', format='svg')
    plt.close()


def code_fn(x):
    if x in CODE2DESCRIPTION:
        return CODE2DESCRIPTION[x]
    if x[0] == 'D':
        if x[1:] in CODE2DESCRIPTION:
            return CODE2DESCRIPTION[x[1:]]
    else:
        return x


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PancNet Grid Search Results Collector.')
    parser.add_argument("--num_features", type=int, default=30, help="Path of the master experiment config")
    args = parser.parse_args()

    code_freq_cancer = pickle.load(open("data/pc_icd_freq.pkl", 'rb'))
    code_freq_all = pickle.load(open("data/all_icd_freq.pkl", 'rb'))
    # cwd = os.getcwd()
    month_idxes = [1]
    for month_idx in month_idxes:
        log_dir= 'logs_00_39'
        # log_dir_stem = '_'.join(log_dir.split('_')[:2])
        with open(os.path.join(log_dir, "part0_epoch_10_w4_batch_4004_devlen10_at.results.test_attribution_{}".format(month_idx)), 'rb') as f:
            word2attr = pickle.load(f)

        attribution_records = [(k, code_fn(k), np.mean(v)) for k, v in word2attr.items() if len(v) > 5]
        df = pd.DataFrame.from_records(attribution_records, columns=["code", "description", "attribution"])
        experiment_args = AttrDict()
        experiment_args.update({'icd10_level': 3, 'icd8_level': 3})
        baseline_disease = np.array([get_code(experiment_args, c) for c in BASELINE_DISEASES])
        assert not df.empty
        os.chdir('data')
        os.makedirs("figures", exist_ok=True)
        df, age_df = process_df(df)
        html_df = df.sort_values(by='attribution', ascending=False)[:args.num_features].round(
            {'attribution': 4, 'weighted contribution': 1})
        html_df.to_html('figures/attribution_{}.html'.format(month_idx),
                        index=False, columns=['code', 'description', 'attribution', 'weighted contribution'],
                        justify='center')
        html_df = df.sort_values(by='weighted contribution', ascending=False)[:args.num_features].round(
            {'attribution': 4, 'weighted contribution': 1})
        html_df.to_html('figures/weighted_attribution_{}.html'.format(month_idx),
                        index=False, columns=['code', 'description', 'attribution', 'weighted contribution'],
                        justify='center')
        plot_attribution_vs_freq(df, suffix=str(month_idx))
        if not age_df.empty:
            plot_age_attribution(age_df, suffix=str(month_idx))


# POSS_VAL_NOT_LIST = 'Flag {} has an invalid list of values: {}. Length of list must be >=1'
# ICD10_MAPPER_NAME = "code/pancnet/data/icd10_eng_diag_chapters.tsv"   # TODO: These files should been renamed
# ICD9_MAPPER_NAME = "code/pancnet/data/icd9_disease_descriptions.tsv"
#
# CODEDF = pd.read_csv(ICD10_MAPPER_NAME, sep='\t', header=None,
#     names=['code', 'description', 'chapter', 'chapter_name', 'block_name', 'block']
#     )
# #CODEDF = CODEDF.append(pd.read_csv(ICD8_MAPPER_NAME, sep='\t', header=None,
# #    names=['code', 'description', 'chapter', 'chapter_name', 'block_name', 'block']
# #    ))
# CODEDF_9 = pd.read_csv(ICD9_MAPPER_NAME, sep='\t', header=0, names=['code_long', 'description', 'shorter description', "NA"])
# CODEDF_9['code'] = [c[:3] for c in CODEDF_9['code_long']]
# CODEDF_9.groupby('code').first()
# CODEDF_w_9 = pd.concat([CODEDF, CODEDF_9])
# CODE2DESCRIPTION = (dict(zip(CODEDF_w_9.code, CODEDF_w_9.description)))
#
# test_attr_fname = 'code/pancnet/logs_00_39/part0_epoch_10_w4_batch_4004_devlen10_at.results.test_attribution'
# with open(test_attr_fname, 'rb') as f:
#     word2attr = pkl.load(f)
#
# attribution_records = [(k, code_fn(k), np.mean(v)) for k, v in word2attr.items() if len(v) > 5]
# df = pd.DataFrame.from_records(attribution_records, columns=["code", "description", "attribution"])