import json
import pandas as pd
import numpy as np
import argparse
import pdb

def add_vte_event():
    pass

def filter_against_cohort(raw_dat_dict, cohort_dict, indexdate='2006-01-01'):
    """Remove patients not in cohort and add index_date, vte event"""
    # pdb.set_trace()
    dat_dict = dict()
    for p in cohort_dict.keys():
        if p in raw_dat_dict.keys() and cohort_dict[p]['IndexDate'] >= indexdate:
            dat_dict[p] = raw_dat_dict[p]
            dat_dict[p]['indexdate'] = cohort_dict[p]['IndexDate']
            dat_dict[p]['ks_mod_score'] = cohort_dict[p]['ks_mod_score']
            if cohort_dict[p]['vte_date'] != 'None': #! add quote is important
                event_dict = dict()
                event_dict['admdate'] = cohort_dict[p]['vte_date']
                event_dict['codes'] = 'VTE'
                event_dict['admid'] = '0000000000'
                dat_dict[p]['events'].append(event_dict)
    return dat_dict

def prep_cohort(cohort_path):
    cohort = pd.read_feather(cohort_path)
    # ! Covert columns to proper data types for downstream use and storage as json format
    # pdb.set_trace()
    cohort_dict = dict()
    cohort['PatientICN'] = cohort['PatientICN'].astype(np.int64)
    cohort['IndexDate'] = cohort['IndexDate'].astype(str)
    cohort['vte_date'] = cohort['vte_date'].astype(str)
    indexdate_dict = dict(zip(cohort.PatientICN, cohort.IndexDate))
    vte_dict = dict(zip(cohort.PatientICN, cohort.vte_date))
    ks_mod_dict = dict(zip(cohort.PatientICN, cohort.ks_mod_score))
    for p in indexdate_dict.keys():
        cohort_dict[p] = dict()
        cohort_dict[p]['IndexDate'] = indexdate_dict[p]
        cohort_dict[p]['vte_date'] = vte_dict[p]
        cohort_dict[p]['ks_mod_score'] = ks_mod_dict[p]
    return cohort_dict

# def find_patients_with_vte_phecode(dat_dict):
#     patients = []
#     val_patients = []
#     for pat in dat_dict:
#         pat_dict = {}
#         pat_dict['patientid'] = pat
#         pat_dict['indexdate'] = dat_dict[pat]['indexdate']
#         pat_dict['vte_events'] = []
#         for event in dat_dict[pat]['events']:
#             # print(event)
#             if '452' in event['codes']:
#                 pat_dict['vte_events'].append(event)

#         if len(pat_dict['vte_events']) >= 1:
#             patients.append(pat_dict)
#             for vte_event in pat_dict['vte_events']:
#                 if parse_date(vte_event['admdate']) > parse_date(pat_dict['indexdate']):
#                     val_patients.append(pat_dict)
#                     break   
#     print(len(patients))
#     print(len(val_patients))
#     return patients, val_patients


def add_ks_score(phe_json_path, ks_feather_path, new_phe_json_path):
    print('Loading metadata json file ...')
    phe_dt = json.load(open(phe_json_path, 'r'))
    print('Loading data with ks scores ...')
    ks_df = pd.read_feather(ks_feather_path)
    ks_score_dt = dict(zip(ks_df.PatientICN, ks_df.ks_score))
    ks_mod_score_dt = dict(zip(ks_df.PatientICN, ks_df.ks_mod_score))
    ks_cat_dt = dict(zip(ks_df.PatientICN, ks_df.ks_cat))
    ks_mod_cat_dt = dict(zip(ks_df.PatientICN, ks_df.ks_mod_cat))

    print('Adding ks score to metadata ... ')
    dat_dt = {}
    common_ids = set(phe_dt.keys()).intersection(set(ks_df.PatientICN))
    print('Cohort size: {}'.format(len(common_ids)))
    for id in common_ids:
        dat_dt[id] = phe_dt[id]
        dat_dt[id].update({'ks_score': ks_score_dt[id],
                           'ks_mod_score': ks_mod_score_dt[id],
                           'ks_cat': ks_cat_dt[id],
                           'ks_mod_cat': ks_mod_cat_dt[id]})
    print("Saving final data into {}".format(new_phe_json_path))
    json.dump(dat_dt, open(new_phe_json_path, 'w'))


def parse_args():
    parser = argparse.ArgumentParser(description='Pre-process json data')
    # parser.add_argument('--raw_json_path', type=str, default='F:\\tmp_pancreatic\\temp_json\\test\\vte\\data_100000.json',
    #                     help="Where all raw tsv files are stored")
    # parser.add_argument('--cohort_path', type=str, default='F:\\tmp_pancreatic\\temp_fst\\global\\raw\\vte\\indexdate_score_and_vtedate.feather',
    #                     help="Where all demographic data stored")
    # parser.add_argument('--out_path', type=str, default='F:\\tmp_pancreatic\\temp_json\\test\\vte\\data_100000_final.json',
    #                     help="Where all demographic data stored")
    
    parser.add_argument('--raw_json_path', type=str, default='F:\\tmp_pancreatic\\temp_json\\test\\vte\\data_100223.json',
                        help="Where all raw tsv files are stored")
    parser.add_argument('--cohort_path', type=str, default='F:\\tmp_pancreatic\\temp_fst\\global\\raw\\vte\\indexdate_score_and_vtedate.feather',
                        help="Where all demographic data stored")
    parser.add_argument('--out_path', type=str, default='F:\\tmp_pancreatic\\temp_json\\test\\vte\\data_100223_final.json',
                        help="Where all demographic data stored")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print("Preprocessing cohot info ...")
    cohort_dict = prep_cohort(args.cohort_path)
    print("Cohort info processed!\n")
    # pdb.set_trace()
    print("Loading raw data ...")
    raw_dat_path = args.raw_json_path
    raw_dat_dict_0 = json.load(open(raw_dat_path, 'r'))
    raw_dat_dict = dict()
    for p in raw_dat_dict_0.keys():
        pd = int(p)
        raw_dat_dict[pd] = raw_dat_dict_0[p]
    del raw_dat_dict_0
    
    print("Preprocessing raw data ...")
    dat_dict = filter_against_cohort(raw_dat_dict, cohort_dict)
    # pdb.set_trace()
    out_path = args.out_path
    print('Saving to {}'.format(out_path))
    with open(out_path, 'w') as fw:
        print("Saving data to {}".format(out_path))
        json.dump(dat_dict, fw, indent=None, sort_keys=True)

    print('All done!')