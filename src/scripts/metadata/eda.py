"""
    This script generates some statistic on the data which are listed in the paper starting from the json
    It also generate the figure used as a dataset description in the paper
"""
import pickle as pkl
import json
from collections import Counter
from tqdm import tqdm
import pickle
import numpy as np
import pickle
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
sns.set_style('white')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.pylab as pylab
import matplotlib.patches as mpatches
import os
from os.path import dirname, realpath
import sys
assert sys.version_info > (3,7,0)
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
import pancnet.datasets.factory as dataset_factory
import pancnet.models.factory as model_factory
import pancnet.learn.train as train
import pancnet.learn.state_keeper as state
from pancnet.utils.parsing import CODEDF, parse_args
from pancnet.utils.time_logger import time_logger
import torch
import datetime
from dateutil.relativedelta import relativedelta
from pancnet.utils.parsing import md5, get_code
from pancnet.utils.date import parse_date
from pancnet.utils.visualization import DK_ICD10_ICD8_DISEASE_HISTOGRAM, US_ICD10_ICD9_DISEASE_HISTOGRAM,\
    chapterColors, save_figure_and_subplots
from pancnet.datasets.bth_disease_progression import PANC_CANCER_CODE, ICD10_PANC_CANCER, \
    ICD9_PANC_CANCER, BASELINE_DISEASES,END_OF_TIME_DATE

def print_stat(array, statistic='', denominator=None):
    array_count = Counter(array)
    if not denominator:
        normalized_array_count = {k:v/sum(array_count.values())*100 for k,v in array_count.items()}
    else:
        normalized_array_count = {k:v/denominator*100 for k,v in array_count.items()}
    print (f"###\t{statistic} stats")
    for k in sorted(array_count):
        print (f"{k}: {array_count[k]} ({normalized_array_count[k]:1.2f}%)")
    print ("##\n\n")

# results_path = "N:/v4/chunlei/PancNet/logs/result.p"
args = parse_args()
# with open(results_path, "rb") as f:
#     resumed_args = pickle.load(f)
# args.__dict__ = resumed_args
baseline_disease = np.array([get_code(args, c) for c in BASELINE_DISEASES])
baseline_disease_set = set(baseline_disease)

# -------------------- Load data ------------------
# j = json.load(open('F:\\tmp_pancreatic\\temp_json\\global\\meta_split\\part_0.json', 'r'))
# j = json.load(open('F:\\tmp_pancreatic\\temp_json\\test\\va_10000_metadata.json', 'r'))
# j = json.load(open('F:\\tmp_pancreatic\\temp_json\\global\\meta_split_level_2\\part_00.json', 'r'))
# j = json.load(open('F:\\tmp_pancreatic\\temp_json\\global\\test\\part_0.json', 'r'))
j = json.load(open('F:\\tmp_pancreatic\\temp_json\\global\\meta_3m\\combined.json', 'r'))

# for p in j:
#     j[p]['split_group']='train'
dataset_class = dataset_factory.get_dataset_class(args)
dataclass = dataset_class(j, args, 'all')

# for p in j:
#     j[p]['split_group']='test'
# dataset_class = dataset_factory.get_dataset_class(args)
# dataclass = dataset_class(j, args, 'test')
# print(dataclass.patients[1])

# ----------- ICD codes statistics ----------------
icd9_codes, icd10_codes = [], []
for p in tqdm(j): 
    for e in j[p]['events']: 
        code = get_code(args, e['codes'])
        if code in ICD10_PANC_CANCER:
            icd10_codes.append(p) 
        elif code in ICD9_PANC_CANCER:
            icd9_codes.append(p)

num_pc_icd10 = len(set(icd10_codes))
num_pc_icd9 = len(set(icd9_codes))
print (f"Original number of pancreatic cancer:\nICD10 codes {num_pc_icd10}\nICD9 codes {num_pc_icd9}")
# time elapsed 1:16:07
# Number of patients with missing end of data is: 0
# Label weights are {False: 8.101692443551458e-07, True: 0.00014792899408284024}
# Number of positive patients  in train is: 3380
# Label weights are {False: 8.101692443551458e-07, True: 0.00014792899408284024}
# Original number of pancreatic cancer:
# ICD10 codes 1383
# ICD9 codes 2518


icd9_codes, icd10_codes = [], []
for p in tqdm(dataclass.patients):
    if p['future_panc_cancer']:
        for e in p['events']:
            code = get_code(args, e['codes'])
            if code in ICD10_PANC_CANCER:
                icd10_codes.append(p['patient_id']) 
            elif code in ICD9_PANC_CANCER:
                icd9_codes.append(p['patient_id'])

num_pc_icd10 = len(set(icd10_codes))
num_pc_icd9 = len(set(icd9_codes))
print (f"After filtering number of pancreatic cancer:\nICD10 codes {num_pc_icd10}\nICD9 codes {num_pc_icd9}")
# After filtering number of pancreatic cancer:
# ICD10 codes 1302
# ICD9 codes 2195

# ---------------- Death status statistics -----------------------
## total number of patients
print ("Total number of patients:{}".format(len(j)))
# Total number of patients:956705

## male female #TODO ADAPT FOR BOSTON DATA
# cpr_df = pd.read_csv('t_person.tsv', sep='\t', dtype='str')
# pid2gender = dict(zip(cpr_df.v_pnr_enc, cpr_df.C_KON))
# pid2end = dict(zip(cpr_df.v_pnr_enc, map(parse_date, cpr_df.D_STATUS_HEN_START)))
# enc_pid2end = {md5(k):v for k,v in pid2end.items()}
# enc_pid2gender = {md5(k):v for k,v in pid2gender.items()}
# pid2bday = dict(zip(cpr_df.v_pnr_enc, cpr_df.D_FODDATO))
# valid_pids = [p for p in j if 'end_of_data' in j[p]]
# gender = [j[p]['gender'] for p in valid_pids ]
#
# ##death alive
# alive_preprocess = {"K":[], "M":[]}
# for p in valid_pids:
#     status = "Alive" if j[p]['end_of_data']==END_OF_TIME_DATE.strftime("%Y-%m-%d") else "Dead"
#     alive_preprocess[j[p]["gender"]].append(status)

alive_preprocess = {"M": [], "F": [], "Both": []}
valid_pids = [p for p in j if 'end_of_data' in j[p] and "U" not in j[p]['gender'] and "NA" not in j[p]['gender']]
enc_pid2gender = {md5(p): j[p]['gender'] for p in j}
gender = [j[p]['gender'] for p in j]
for p in valid_pids:
    status = "Alive" if j[p]['end_of_data'] == END_OF_TIME_DATE.strftime("%Y-%m-%d") else "Dead"
    alive_preprocess[j[p]["gender"]].append(status)
    alive_preprocess["Both"].append(status)

print_stat(gender, statistic='Original data Gender', denominator=len(valid_pids))
print_stat(alive_preprocess["M"], statistic='Original data Status Males', denominator=len(valid_pids))
print_stat(alive_preprocess["F"], statistic='Original data Status Females', denominator=len(valid_pids))
print_stat(alive_preprocess['Both'], statistic='Original data Status All', denominator=len(valid_pids))
### Total number of valid positive patients for the model:3904
# ###	Original data Gender stats
# F: 131869 (13.78%)
# M: 824834 (86.22%)
# NA: 2 (0.00%)
# ###	Original data Status Males stats
# Alive: 83 (0.01%)
# Dead: 824751 (86.21%)
# ###	Original data Status Females stats
# Alive: 21 (0.00%)
# Dead: 131848 (13.78%)
###     Original data Status All stats
# Alive: 104 (0.01%)
# Dead: 956599 (99.99%)


# ----------------------------------------------------
# ----------------age at first panc cancer------------
# ----------------------------------------------------

# age_at_cancer = {'K':[], 'M':[]}
# age_at_end = {'K':[], 'M':[]}
age_at_cancer = {'M':[], 'F':[], 'Both': []}
age_at_end = {'M':[], 'F':[], 'Both': []}
pc_ages = {'M':[], "F":[]}
num_codes = {"all":[], 'cancer':[]}
timespan_trajectory = {"all":[], 'cancer':[]}
timepoint_codes = {'cancer':[]}
# pc_ages = {'K':[], 'M':[]}
gender_at_cancer = []
gender_processed = []
bins = np.array([-12*30, -6*30, -3*30, 0])
intervals = [
    "<12",
    "12-6",
    "6-3",
    "3-0",
]

for p in tqdm(dataclass.patients):
    sex = p['gender'] # no gender key in patients
    if 'U' in sex or 'NA' in sex:
        continue
    age = relativedelta(p['outcome_date'], p['dob']).years
    age = (age//10)*10
    age_interval = '{}-{}'.format(age, age+10)
    trajectory_length = len([True for e in p['events'] if e['admit_date']<p['outcome_date']])
    num_codes['all'].append(trajectory_length)
    time_trajectory = relativedelta(p['outcome_date'], p['events'][0]['admit_date']).years
    timespan_trajectory['all'].append(time_trajectory)
    if p['future_panc_cancer']:
        for e in p['events']:
            timediff = (e['admit_date'] - p['outcome_date']).days
            if timediff>=0:
                continue
            timepoint_codes['cancer'].append(intervals[np.digitize(timediff, bins)])
        num_codes['cancer'].append(trajectory_length)
        timespan_trajectory['cancer'].append(time_trajectory)
        pc_ages[sex].append(age)
        gender_at_cancer.append(sex)
        age_at_cancer[sex].append(age_interval)
        age_at_cancer['Both'].append(age_interval)
    age_at_end[sex].append(age_interval)
    age_at_end['Both'].append(age_interval)
    gender_processed.append(sex)

print (f"### Total number of valid positive patients for the model:{sum([p['future_panc_cancer'] for p in dataclass.patients])}")
# Total number of valid patients for the model:3380
#


all_pc_ages = pc_ages['M'] + pc_ages['F']
print (f"Age pancreatic Cancer Processed Popuation: Mean {np.mean(all_pc_ages)} Median {np.median(all_pc_ages)} STD {np.std(all_pc_ages)}")
print (f"Male Age pancreatic Cancer Processed Popuation: Mean {np.mean(pc_ages['M'])} Median {np.median(pc_ages['M'])} STD {np.std(pc_ages['M'])}")
print (f"Female Age pancreatic Cancer Processed Popuation: Mean {np.mean(pc_ages['F'])} Median {np.median(pc_ages['F'])} STD {np.std(pc_ages['F'])}")
print()
print (f"Number of codes before outcome date for Processed Population: Mean {np.mean(num_codes['all'])} Median {np.median(num_codes['all'])} STD {np.std(num_codes['all'])}")
print (f"Number of codes before outcome date for Processed Cancer Population: Mean {np.mean(num_codes['cancer'])} Median {np.median(num_codes['cancer'])} STD {np.std(num_codes['cancer'])}")
print()
print (f"Length of trajectory for Processed Population: Mean {np.mean(timespan_trajectory['all'])} Median {np.median(timespan_trajectory['all'])} STD {np.std(timespan_trajectory['all'])}")
print (f"Length of trajectory for Processed Cancer Population: Mean {np.mean(timespan_trajectory['cancer'])} Median {np.median(timespan_trajectory['cancer'])} STD {np.std(timespan_trajectory['cancer'])}")
print()
# Age pancreatic Cancer Processed Popuation: Mean 65.50591715976331 Median 70.0 STD 10.908913917390807
# Male Age pancreatic Cancer Processed Popuation: Mean 65.68043742405833 Median 70.0 STD 10.805580836235375
# Female Age pancreatic Cancer Processed Popuation: Mean 58.97727272727273 Median 60.0 STD 12.616707242893947
#
# Number of codes before outcome date for Processed Population: Mean 297.94785789681487 Median 163.0 STD 410.96581485032766
# Number of codes before outcome date for Processed Cancer Population: Mean 289.79023668639053 Median 159.0 STD 414.63414838296836
#
# Length of trajectory for Processed Population: Mean 11.84881916410839 Median 11.0 STD 5.626687041988851
# Length of trajectory for Processed Cancer Population: Mean 7.970710059171598 Median 7.0 STD 5.766001668007124
#
print_stat(timepoint_codes["cancer"], statistic='Number of codes at different bins Cancer Population')
print()
print_stat(gender_processed, statistic='Gender for Processed Cancer population', denominator=len(valid_pids))

print_stat(age_at_end['M'], statistic='Male Age For Processed Cancer Population', denominator=len(valid_pids))
print_stat(age_at_end['F'], statistic='Female Age For Processed Cancer Population', denominator=len(valid_pids))
print_stat(age_at_end['Both'], statistic='All Age For General Population', denominator=len(valid_pids))
print()
print_stat(gender_at_cancer, statistic='Gender at cancer Processed Cancer Popuation')
print_stat(age_at_cancer['M'], statistic='Male Age at cancer Processed Cancer Popuation', denominator=len(gender_at_cancer))
print_stat(age_at_cancer['F'], statistic='Female at cancer Processed Cancer Popuation', denominator=len(gender_at_cancer))
print_stat(age_at_cancer['Both'], statistic='All at cancer', denominator=len(gender_at_cancer))
###	Number of codes at different bins Cancer Population stats
# 12-6: 71474 (7.30%)
# 3-0: 66530 (6.79%)
# 6-3: 39903 (4.07%)
# <12: 801584 (81.84%)
#
# # ###	Gender for Processed Cancer population stats
# F: 56003 (5.85%)
# M: 564532 (59.01%)

#
# # ###	Male Age For Processed Cancer Population stats
# 0-10: 7 (0.00%)
# 10-20: 31 (0.00%)
# 100-110: 822 (0.09%)
# 110-120: 4 (0.00%)
# 20-30: 5625 (0.59%)
# 30-40: 34910 (3.65%)
# 40-50: 34271 (3.58%)
# 50-60: 60149 (6.29%)
# 60-70: 104174 (10.89%)
# 70-80: 163188 (17.06%)
# 80-90: 119232 (12.46%)
# 90-100: 42119 (4.40%)

#
# # ###	Female Age For Processed Cancer Population stats
# 0-10: 10 (0.00%)
# 10-20: 37 (0.00%)
# 100-110: 68 (0.01%)
# 110-120: 11 (0.00%)
# 120-130: 1 (0.00%)
# 20-30: 1727 (0.18%)
# 30-40: 9406 (0.98%)
# 40-50: 9895 (1.03%)
# 50-60: 13231 (1.38%)
# 60-70: 13716 (1.43%)
# 70-80: 4142 (0.43%)
# 80-90: 2354 (0.25%)
# 90-100: 1405 (0.15%)
#
###     All Age For General Population stats
# 0-10: 17 (0.00%)
# 10-20: 68 (0.01%)
# 100-110: 890 (0.09%)
# 110-120: 15 (0.00%)
# 120-130: 1 (0.00%)
# 20-30: 7352 (0.77%)
# 30-40: 44316 (4.63%)
# 40-50: 44166 (4.62%)
# 50-60: 73380 (7.67%)
# 60-70: 117890 (12.32%)
# 70-80: 167330 (17.49%)
# 80-90: 121586 (12.71%)
# 90-100: 43524 (4.55%)

# # ###	Gender at cancer Processed Cancer Popuation stats
# F: 88 (2.60%)
# M: 3292 (97.40%)
#
# # ###	Male Age at cancer Processed Cancer Popuation stats
# 30-40: 10 (0.30%)
# 40-50: 67 (1.98%)
# 50-60: 435 (12.87%)
# 60-70: 1053 (31.15%)
# 70-80: 1065 (31.51%)
# 80-90: 582 (17.22%)
# 90-100: 80 (2.37%)
#
# # ###	Female at cancer Processed Cancer Popuation stats
# 30-40: 3 (0.09%)
# 40-50: 6 (0.18%)
# 50-60: 24 (0.71%)
# 60-70: 33 (0.98%)
# 70-80: 9 (0.27%)
# 80-90: 12 (0.36%)
# 90-100: 1 (0.03%)
#
###     All at cancer stats
# 30-40: 13 (0.38%)
# 40-50: 73 (2.16%)
# 50-60: 459 (13.58%)
# 60-70: 1086 (32.13%)
# 70-80: 1074 (31.78%)
# 80-90: 594 (17.57%)
# 90-100: 81 (2.40%)
#
# -----------------------------------------------------------------------
# -------------------Disease distribution visualization------------------
# -----------------------------------------------------------------------

# ---------------------- Code distribution - histogram plot -------------
icd2chapter = dict(zip(CODEDF.code, CODEDF.chapter_name))
chapt2chaptname = dict(zip(CODEDF.chapter, CODEDF.chapter_name))
# icd2chapter = {d: d[0] for d in CODEDF.code}
# chapt2chaptname = {c: "Chapter-"+c for c in icd2chapter.values()}
boxplot_codes_prior_pc = []
for p in dataclass.patients:
    if p['future_panc_cancer']:
        for e in p['events']:
            #if e['codes'][0]!='D': #TODO this filter works only for Danish data
            #    continue
            try:
                code = get_code(args, e['codes'])
                chapter = icd2chapter[code]
            except KeyError:
                continue
            time_to_cancer = (e['admit_date'] - p['outcome_date']).days
            boxplot_codes_prior_pc.append([p['patient_id'], time_to_cancer, chapter, code])

df = pd.DataFrame.from_records(boxplot_codes_prior_pc, columns=['pid', 'time', 'chapter', 'code'])
print(df.head())
df['time'] = df.time//30
binranges = [(i,i+1) for i in range(-12,13)]
bins = pd.IntervalIndex.from_tuples(binranges)
binedges = bins.left.tolist() + bins.right[-1:].tolist()

binned_df = df.groupby(['pid', pd.cut(df.time, bins=bins)]).size()
binned_df = binned_df.reset_index(name='counts')
fig,axes = plt.subplots(2,1,figsize=[12,18])
sns.boxplot(data=binned_df, x='time',
    y='counts', whis=10,
    fliersize=1, ax=axes[0],
    palette=sns.diverging_palette(145,300, n=len(bins))
    )
axes[0].set_xlabel('Time interval (months)', fontsize=18)
axes[0].set_ylabel('Number of codes', fontsize=18)
axes[0].set_title('Code density', fontsize=20)
axes[0].tick_params(axis='both', rotation=45, labelsize=16)
sns.histplot(data=df, x='time', bins=binedges,
    hue='chapter', multiple='stack',
    stat='percent', ax=axes[1], legend=False,
    hue_order=[v for k,v, in chapt2chaptname.items() if v>0],
    palette=chapterColors
)
axes[1].set_xlabel('Time interval (months)', fontsize=18)
axes[1].set_ylabel('Code proportion', fontsize=18)
axes[1].set_title('Code density by chapter', fontsize=20)
axes[1].tick_params(axis='both', labelsize=16)

legend_patches = [mpatches.Patch(color=c, label=l) for c,l in zip(chapterColors, chapt2chaptname.keys())]
axes[1].legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(0,-0.1))
fig.tight_layout()
os.makedirs('data/figures/', exist_ok=True)
plt.savefig('data/figures/disease_distribution_pc.svg', format='svg', dpi=300)

# ---------------------- Code distribution - ridge plot -------------
gs = GridSpec(len(chapt2chaptname),1)
fig = plt.figure(figsize=(8,8))
i = 0

ax_objs = []
chapt2chaptname = dict(sorted(chapt2chaptname.items(), key=lambda item: item[1]))
for chapter, chaptercolor in zip(chapt2chaptname, chapterColors):

    chapter_name = chapt2chaptname[chapter]
    print(chapter_name)
    if df[df.chapter == chapter_name].time.empty:
        print(f"WARNING MISSING CHAPT {chapter_name}")
        continue

    ax_objs.append(fig.add_subplot(gs[i:i+1, 0:]))
    # plotting the distribution
    plot = (df[df.chapter == chapter_name]
            .time.plot.kde(ax=ax_objs[-1], color="#0f0f0f", lw=0.5)
           )
    # make background transparent
    rect = ax_objs[-1].patch
    rect.set_alpha(0)
    x = plot.get_children()[0]._x
    y = plot.get_children()[0]._y
    ax_objs[-1].fill_between(x,y,color=chaptercolor, alpha=0.8)
    ax_objs[-1].text(0.05,0.1,"Chapter {}".format(chapter_name),
                    fontsize=16, ha="left", transform=ax_objs[-1].transAxes)
    i += 1

for i, a in enumerate(ax_objs):
    if i == len(ax_objs)-1:
        a.set_xlabel("Time (Months)", fontsize=22)
        a.tick_params(axis='x', labelsize=18)
    elif i == 0:
        a.set_xlabel('')
        a.tick_params(axis='x', labelsize=10)
    else:
        a.set_xticklabels([])
        a.set_xlabel('')
    a.set_yticklabels([])
    a.set_ylabel('')
    spines = ["top", "right", "left", "bottom"]
    for s in spines:
        a.spines[s].set_visible(False)

gs.update(hspace=-0.6)
fig.suptitle("Distribution of Disease by Chapter before and after PC", fontsize=16)
plt.tight_layout()
plt.savefig('data/figures/ridgeplot_disease_distribution_pc.png', format='png', dpi=300)

# ------------------------ Disease distribution ----------------------------------
# histograms_code_description = DK_ICD10_ICD8_DISEASE_HISTOGRAM #TODO change this for USA codes
histograms_code_description = US_ICD10_ICD9_DISEASE_HISTOGRAM
records_time_visit_patient_count = []
records_age_at_cancer = []
records_age_at_disease = []
records_disease_incidence = {bd:[0,0] for bd in histograms_code_description.values()}
is_cancer = [p['future_panc_cancer'] for p in dataclass.patients]
total_cancer_pt = sum(is_cancer)
total_non_cancer_pt = len(is_cancer) - total_cancer_pt

for p in tqdm(dataclass.patients):
    is_pc = p['future_panc_cancer']
    sex = enc_pid2gender[p['patient_id']]
    run_bd_code_patients = {bd:True for bd in histograms_code_description.values()}
    if is_pc:
        date_limit = p['outcome_date']
        age_at_cancer = relativedelta(p['outcome_date'], p['dob']).years
        records_age_at_cancer.append((sex, age_at_cancer))
    else:
        date_limit = p['outcome_date'] - relativedelta(years=2)
    for e in p['events']:
        year = e['admit_date'].year##extract_year
        records_time_visit_patient_count.append((year, is_pc))
        age = relativedelta(e['admit_date'], p['dob']).years
        records_age_at_disease.append((age, is_pc))
        code = get_code(args, e['codes'])
        if code in histograms_code_description and \
            run_bd_code_patients[histograms_code_description[code]] and \
            e['admit_date'] < date_limit:
            run_bd_code_patients[histograms_code_description[code]] = False
            if is_pc:
                records_disease_incidence[histograms_code_description[code]][1]+=1
            else:
                records_disease_incidence[histograms_code_description[code]][0]+=1
# time elapsed: 50mins

export_pkl = [records_age_at_cancer, records_time_visit_patient_count, records_age_at_disease, records_disease_incidence]
pkl.dump(export_pkl, open("data/figures/metadata_stats_data.pkl", 'wb'))
# records_age_at_cancer, records_time_visit_patient_count, records_age_at_disease, records_disease_incidence = \
#     pkl.load(open("data/figures/metadata_stats_data.pkl", 'rb'))

plt.style.use('seaborn-deep')
plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=24)  # fontsize of the figure title
plt.close()

fig = plt.figure(figsize=[24,7])

fig.suptitle('Dataset composition', fontsize=16)
gs = GridSpec(1,4)
print ("age_at_cancer_by_sex")
ax1=fig.add_subplot(gs[0])
df = pd.DataFrame.from_records(records_age_at_cancer, columns=['Sex', 'age'])
df = df.replace({'K':"F"})
sns.set_context(rc = {'patch.linewidth': 0.0})
sns.histplot(df, x='age', bins=15,ax=ax1, multiple='dodge', shrink=0.8)
ax1.set(xlabel='Age', ylabel='Number of patients', title='A - Age at pancreatic cancer')
p = ax1.patches
age_histvalues = pd.DataFrame( list( zip([patch.get_x() for patch in p], [patch.get_height() for patch in p]) ), columns=['x','y'] )
age_histvalues.to_pickle('data/figures/age_histvalues.p')

ax1=fig.add_subplot(gs[1])
print ("records_per_year")
df = pd.DataFrame.from_records(records_time_visit_patient_count, columns=['Year', 'Cancer'])
ca = df[df.Cancer==True].Year.values
nonca = df[df.Cancer==False].Year.values
ax1.hist([ca, nonca], bins=40, label=["Yes", "No"], density=True, color=sns.color_palette(['#D4624E','#bebebe']))
ax1.set(xlim=(1998, 2022), xlabel='Year', ylabel='Disease code frequency', title="B - Disease distribution")
# ax1.axvline(1994, -1, 1, c='grey')
# ax1.text(1987, 0.06, "ICD-8")
# ax1.text(2000, 0.06, "ICD-10")
ax1.legend(title="PC")

ax1=fig.add_subplot(gs[2])
print ("age_at_disease_by_cancer")
df = pd.DataFrame.from_records(records_age_at_disease, columns=['Age', 'Cancer'])
ca = df[df.Cancer==True].Age.values
nonca = df[df.Cancer==False].Age.values
ax1.hist([ca, nonca], bins=52, label=["Yes", "No"], density=True, color=sns.color_palette(['#D4624E','#bebebe']))
ax1.legend(title="PC")
ax1.set(xlim=(0,100), xlabel='Age', ylabel='Disease code frequency', title='C - Disease distribution at age of PC')

ax1=fig.add_subplot(gs[3])
print ("disease_freq_by_cancer")
df = pd.DataFrame.from_dict(records_disease_incidence, columns=['No', 'Yes'], orient='index')
df = df.sort_values('Yes', ascending=False)
df = df[['Yes', 'No']]
df['disease_char'] = [l for i,l in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ') if i< len(df)]

df = df.reset_index().rename(columns={'index':'disease'})
char_to_disease = dict(zip(df.disease_char, df.disease))
df['Yes'] = df['Yes']/total_cancer_pt
df['No'] = df['No']/total_non_cancer_pt
df = df.melt(id_vars=['disease', 'disease_char'])
sns.set_palette(sns.color_palette(['#D4624E','#bebebe']))
sns.barplot(data=df, x='disease_char', y='value', hue='variable', ax=ax1, alpha=0.8)
ax1.tick_params(axis='x', labelrotation=0)
ax1.legend(title="PC")
ax1.set(xlabel='Diseases', ylabel='Incidence', title='D - Prior knowledge disease incidence')
[i.set_ha('right') for i in ax1.get_yticklabels()]
cancer_legend = ax1.get_legend()
legend_patches = [mpatches.Patch(color='w', label=f"{k} : {v}") for k,v in char_to_disease.items()]
disease_legend = plt.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(0,-0.2))
ax1.add_artist(disease_legend)
ax1.add_artist(cancer_legend)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
save_figure_and_subplots('data/figures/metadata_stats', fig, format='svg', dpi=300)


# Generate code freq files
df = pd.DataFrame.from_records(boxplot_codes_prior_pc, columns=['pid', 'time', 'chapter', 'code'])
freq_pkl_cancer = Counter(df.code)
# pkl.dump(freq_pkl_cancer, open('data/pc_icd_freq.pkl', 'wb'))
pkl.dump(freq_pkl_cancer, open('data/figures/pc_icd_freq.pkl', 'wb'))

freq_pkl_all = {c:0 for c in freq_pkl_cancer.keys()}
# freq_pkl_all = codes_all.copy()
for p in tqdm(dataclass.patients):
    for e in p['events']:
        try:
            code = get_code(args, e['codes'])
        except KeyError:
            continue
        try:
            freq_pkl_all[code] += 1
        except KeyError:
            freq_pkl_all.update({code:1})

# pkl.dump(freq_pkl_all, open('data/all_icd_freq.pkl', 'wb'))
pkl.dump(freq_pkl_all, open('data/figures/all_icd_freq.pkl', 'wb'))


df_c = pd.DataFrame(list(zip(num_codes['cancer'], timespan_trajectory['cancer'])), columns=['ncodes', 'trajlength'])
df_c['cancer'] = "yes"
df_a = pd.DataFrame(list(zip(num_codes['all'], timespan_trajectory['all'])), columns=['ncodes', 'trajlength'])
df_a['cancer'] = "no"
df = pd.concat([df_c, df_a])
df.to_csv("data/figures/metadata_stats_trajectory_length_all.csv")
#         ncodes  trajlength cancer
# 0           61           0    yes
# 1           49           6    yes
# 2            7           0    yes
# 3          100           2    yes
# 4            8           0    yes
# ...        ...         ...    ...
# 759855     593          18     no
# 759856     239          23     no
# 759857     367           7     no
# 759858    1078          18     no
# 759859     831          19     no
#
# [763764 rows x 3 columns]
# df = pd.read_csv("figures/metadata_stats_trajectory_length_all.csv", index_col=0)
x, y, hue = df['trajlength'], df['ncodes'], df['cancer']
jointdf = pd.DataFrame.from_records(zip(x,y,hue), columns=['Trajectory time range','# of codes', 'Cancer'])
xrange = range(0,21,4)
yrange = range(0,2001,100)

plt.subplots(figsize=[6,4])
ax = plt.subplot(121)
sns.histplot(data=jointdf.loc[jointdf['Cancer'] == 'no'], x='Trajectory time range', y='# of codes', bins=[xrange, yrange], pmax=0.5,
             ax=ax, element='step', cbar=True, cmap='mako_r', edgecolor="0.6", linewidth=0.4)
plt.yticks(yrange[::5])
plt.ylim([0, 2000]); plt.xlim([0, 21])

ax = plt.subplot(122)
sns.histplot(data=jointdf.loc[jointdf['Cancer'] == 'yes'], x='Trajectory time range', y='# of codes', bins=[xrange, yrange], pmax=0.5,
             ax=ax, element='step', cbar=True, cmap='rocket_r', edgecolor="0.6", linewidth=0.4)
plt.yticks(yrange[::5])
plt.ylim([0, 2000]); plt.xlim([0, 21])

plt.tight_layout()
plt.savefig('data/figures/metadata_stats_trajectory_length_heatmap.svg', format='svg', dpi=300)



df_a_sub = df_a.iloc[np.random.randint(0, df_a.shape[0], df_c.shape[0])]
df = pd.concat([df_c, df_a_sub])
df.to_csv("data/figures/metadata_stats_trajectory_length.csv")
#         ncodes  trajlength cancer
# 0           61           0    yes
# 1           49           6    yes
# 2            7           0    yes
# 3          100           2    yes
# 4            8           0    yes
# ...        ...         ...    ...
# 680410      53           7     no
# 123491     245           4     no
# 721084     237          16     no
# 612080    1028          14     no
# 690498     117          26     no
# [7808 rows x 3 columns]
# df = pd.read_csv("figures/metadata_stats_trajectory_length.csv", index_col=0)

plt.title('Trajectory codes vs. length', fontsize=18)
sns.set_palette(sns.color_palette(['#D4624E','#bebebe']))

# def customJoint(x,y,hue,*args,**kwargs):
#     jointdf = pd.DataFrame.from_records(zip(x,y,hue), columns=['x','y','Cancer'])
#     sns.kdeplot(data=jointdf, x='x', y='y', hue='Cancer', shade=False, bw_adjust=0.5, thresh=0.1, alpha=.6)
#
# def customMarginal(x,hue,*args,**kwargs):
#     margdf = pd.DataFrame.from_records(zip(x,hue), columns=['value','hue'])
#     if kwargs['vertical']:
#         sns.histplot(data=margdf, y='value', hue='hue', bins=40, binwidth=2, stat='density', legend=False, alpha=.6)
#     else:
#         sns.histplot(data=margdf, x='value', hue='hue', bins=32, binrange=[0,32], stat='density', legend=False, alpha=.6)
#
# g = sns.JointGrid(x="trajlength", y="ncodes", hue='cancer', data=df)
# g = g.plot(customJoint, customMarginal)

x, y, hue = df['trajlength'], df['ncodes'], df['cancer']
jointdf = pd.DataFrame.from_records(zip(x,y,hue), columns=['x','y','Cancer'])
sns.kdeplot(data=jointdf, x='x', y='y', hue='Cancer')

# margdf = pd.DataFrame.from_records(zip(x, hue), columns=['value', 'hue'])
# sns.histplot(data=margdf, y='value', hue='hue', multiple='dodge', binwidth=2, stat='density', legend=False)
#
# margdf = pd.DataFrame.from_records(zip(y, hue), columns=['value', 'hue'])
# sns.histplot(data=margdf, x='value', hue='hue', multiple='dodge', bins=30, stat='density', legend=False)

plt.ylim([0, 2000])
plt.xlim([0, 21])
plt.savefig('data/figures/metadata_stats_trajectory_length.svg', format='svg', dpi=300)
