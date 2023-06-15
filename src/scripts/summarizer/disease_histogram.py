import json
import pickle
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
from pancnet.utils.visualization import US_ICD10_ICD9_DISEASE_HISTOGRAM
from pancnet.utils.date import parse_date
import pancnet.datasets.factory as dataset_factory
from pancnet.datasets.bth_disease_progression import BTH_Disease_Progression_Dataset,\
    PANC_CANCER_CODE, BASELINE_DISEASES
from pancnet.utils.parsing import parse_args, md5, get_code
import pandas as pd
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from matplotlib.gridspec import GridSpec
import math

#generates two histograms side by side, one with birth as reference and the other as end of date/ pc diagnose
args = parse_args()
# args.results_path = 'output/20210311-1928_transformer_revision_logs/ac3f725f4cdefda0a4a4f0db6a3bf838.results' #TODO this should not be hard-coded
# resumed_args = pickle.load(open(args.results_path, "rb"))
# args.__dict__ = resumed_args
# metadata = json.load(open(args.metadata_path))
metadata_path = 'F:\\tmp_pancreatic\\temp_json\\global\\train_2\\combined.json'
metadata = json.load(open(metadata_path))

baseline_disease = np.array([get_code(args, c) for c in BASELINE_DISEASES])
mean_cancer_age = 64
std_cancer_age = 10.44
norm_dist_cancer_occurrence = np.linspace(-3 * std_cancer_age + mean_cancer_age, 3 * std_cancer_age + mean_cancer_age, 100)

histograms_code_description = US_ICD10_ICD9_DISEASE_HISTOGRAM #TODO change this for USA visualization
class BTH_Disease_Progression_Histogram(BTH_Disease_Progression_Dataset):
    def __init__(self, metadata, args):
        self.args = args
        self.metadata = metadata
        self.patients = []
        shard_path = None
        count_missing_date = 0
        self.shard = True if type(metadata) is list else False
        self.histograms_code = histograms_code_description
        self.patient_summary = {
            "code":[],
            "future_panc_cancer":[], 
            "event_to_eod":[],
            "dob_to_event":[]
        }
        
        for patient in tqdm.tqdm(metadata):
            if self.shard:
                (patient_not_encoded, current_shard_path), = patient.items()
                
                patient_id = md5(patient_not_encoded)
                if shard_path !=  current_shard_path:
                    shard_path = current_shard_path
                    patient_metadata = json.load(open(shard_path, 'r'))
            else:
                patient_id = md5(patient)
                patient_metadata = {patient_id:metadata[patient]}

            if 'end_of_data' not in patient_metadata[patient_id]:
                count_missing_date += 1
                continue
                
            obs_time_end = parse_date(patient_metadata[patient_id]['end_of_data'])
            dob = parse_date(patient_metadata[patient_id]['birthdate'])
            events = self.process_events(patient_metadata[patient_id]['events'], obs_time_end)
            future_panc_cancer, outcome_date = self.get_outcome_date(events, end_of_date=obs_time_end)
            for event in events:
                c = get_code(args, event["codes"])
                if c in self.histograms_code:
                    self.patient_summary['code'].append(self.histograms_code[c])
                    event_to_eod = (event['admit_date'] - outcome_date).days/365
                    dob_to_event = (event['admit_date'] - dob).days/365
                    self.patient_summary["future_panc_cancer"].append(future_panc_cancer)
                    self.patient_summary["event_to_eod"].append(event_to_eod) 
                    self.patient_summary["dob_to_event"].append(dob_to_event)

summary_data = BTH_Disease_Progression_Histogram(metadata, args)
histograms = pd.DataFrame.from_dict(summary_data.patient_summary)
print(histograms.head())
pickle.dump(histograms, open("data/figures/histograms/disease_histogram.pkl", 'wb'))
fig, ax = plt.subplots(5,2, figsize=(12,18))
fig.suptitle('Disease distribution', fontsize=14)
print(set(summary_data.histograms_code.values()))
for r,code in enumerate(set(summary_data.histograms_code.values())):
    
    axes = ax[r%5, r//5]
    df = histograms[histograms.code==code]
    try:
        dob_to_event = pd.DataFrame({'Non cancer patients': df.groupby('future_panc_cancer').get_group(False).dob_to_event,
                    'Cancer patients':   df.groupby('future_panc_cancer').get_group(True).dob_to_event})
        dob_to_event.plot(kind='hist', bins=80, ax=axes, density=1, stacked=False, alpha=.5)
        axes.set(xlim=(-3,110), title=f"{code}")
        axes.plot(norm_dist_cancer_occurrence, stats.norm.pdf(norm_dist_cancer_occurrence, mean_cancer_age, std_cancer_age), alpha=0.5, color='red', label='Cancer diagnosis')
        axes.set_xlabel("Age at Event")
        axes.legend()
    except KeyError:
        pass

plt.legend()
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig(f"data/figures/histograms/disease_histogram.png", bbox_inches='tight')
fig.savefig(f"data/figures/histograms/disease_histogram.svg", bbox_inches='tight', format='svg')
plt.close()