import json
import os
import argparse
import json
import numpy as np
import pandas as pd
import pickle as pkl
import gc
from tqdm import tqdm
import pdb
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))


def parse_args():
    parser = argparse.ArgumentParser(description='Build metadata json from tsv')
    parser.add_argument('--tsv_dir', type=str, default='F:\\tmp_pancreatic\\temp_tsv\\global\\drug_split',
                        help="Where all raw tsv files are stored")
    parser.add_argument('--json_cohort_dir', type=str, default='F:\\tmp_pancreatic\\temp_json\\global\\icd_split',
                        help="Where all json formatted cohorts based on icd codes stored")
    parser.add_argument('--demo_path', type=str, default='F:\\tmp_pancreatic\\temp_tsv\\global\\raw\\demo.tsv',
                        help="Where all demographic data stored")
    parser.add_argument('--out_dir', type=str, default='F:\\tmp_pancreatic\\temp_json\\global\\drug_split',
                        help="Where all demographic data stored")
    # parser.add_argument('--library_path', type=str, default=None,
    #                     help="Where to general the all_icd.pkl file. If empty, do not generate.")
    parser.add_argument('--delimiter', type=str, default='\t')
    parser.add_argument('--head', type=int, default=None)
    return parser.parse_args()


def tsv_to_json(tsv_filepath, dob_dict, dod_dict, gender_dict):
    print("Loading tsv data...")
    with open(tsv_filepath, "r") as tsv:
        if args.head is None:
            lines = tsv.readlines()
        else:
            lines = []
            for _ in range(args.head):
                lines.append(tsv.readline())
        legend = lines[0].strip().split(args.delimiter)
        rows = [row.strip().split(args.delimiter) for row in lines[1:]]
        print(f'legend: {legend}')
        assert all([j in legend for j in ["PatientICN", "Date", "ICDCodeType", "ICDCode"]])
        metadata = {}
        key_translater = {
            'PatientICN': 'PID',
            'Date': 'admdate',
            'ICDCode': 'codes',
            'ICDCodeType': 'code_type'
        }
        all_codes = set()

        print("Parsing tsv data...")
        for rowid, row in tqdm(enumerate(rows)):
            # ! Some rows missed patient id, resulting in less elements for these rows, but still can be zipped
            if len(legend) != len(row):
                continue

            row_dict = {
                key_translater[k]: v for k, v in zip(legend, row)
            }
            row_dict.update({'admid': str(rowid).zfill(10)})

            # missing or being the header line
            if row_dict["PID"] == 'NaN' or row_dict["codes"] == '' or row_dict["codes"] == 'ICDCode' or \
                    'Missing' in row_dict["codes"]:
                continue

            # ! Pay attention to the following date type change for correct mapping
            patient_id = int(float(row_dict['PID']))

            if patient_id in dob_dict:
                row_dict['DOB'] = dob_dict[patient_id]
                row_dict['DOD'] = dod_dict[patient_id]
                row_dict['gender'] = gender_dict[patient_id]
            else:
                row_dict['DOB'] = 'NA'
                row_dict['DOD'] = 'NA'
                row_dict['gender'] = 'NA'

            if patient_id not in metadata:
                metadata[patient_id] = {'events': []}
                metadata[patient_id]['events'].append(row_dict)
            else:
                metadata[patient_id]['events'].append(row_dict)

            if args.library_path:
                all_codes.add(row_dict['codes'])

        print("Generating json data...")
        warning_flag = False
        for patient_id in tqdm(metadata.keys()):
            metadata[patient_id]['events'] = sorted(metadata[patient_id]['events'], key=lambda i: i['admdate'])
            metadata[patient_id]['split_group'] = np.random.choice(['train', 'dev', 'test'], p=[.80, .10, .10])
            end_of_data = metadata[patient_id]['events'][0]['DOD'] \
                if metadata[patient_id]['events'][0]['DOD'] != 'NA' and \
                   metadata[patient_id]['events'][0]['DOD'] != 'None' \
                else ''
            metadata[patient_id]['end_of_data'] = end_of_data \
                if end_of_data != '' else max([e['admdate'] for e in metadata[patient_id]['events']])
            start_of_data = metadata[patient_id]['events'][0]['DOB'] \
                if metadata[patient_id]['events'][0]['DOB'] != 'NA' and \
                   metadata[patient_id]['events'][0]['DOB'] != 'None' \
                else ''
            metadata[patient_id]['birthdate'] = start_of_data \
                if start_of_data != '' else min([e['admdate'] for e in metadata[patient_id]['events']])
            metadata[patient_id]['gender'] = metadata[patient_id]['events'][0]['gender'] \
                if 'gender' in metadata[patient_id]['events'][0] and \
                   metadata[patient_id]['events'][0]['gender'] in ['M', 'F'] \
                else 'U'
            for e in metadata[patient_id]['events']:
                try:
                    del e['PID']
                    del e['DOD']
                    del e['DOB']
                    del e['gender']
                except KeyError:
                    if not warning_flag:
                        print("Warning: Date of birth, death, or other entries are missing!")
                        warning_flag = True
        assert not [p for p in metadata if any(['_' in e['admdate'] for e in metadata[p]['events']])]
        return all_codes, metadata

def drug_tsv_to_json(tsv_filepath, dob_dict, dod_dict, gender_dict, tdt_dict):
    print("Loading tsv data...")
    with open(tsv_filepath, "r") as tsv:
        if args.head is None:
            lines = tsv.readlines()
        else:
            lines = []
            for _ in range(args.head):
                lines.append(tsv.readline())
        legend = lines[0].strip().split(args.delimiter)[:-1]
        rows = [row.strip().split(args.delimiter) for row in lines[1:]]
        print(f'legend: {legend}')
        assert all([j in legend for j in ["person_source_value", "drug_era_start_date", "drug_name"]])
        metadata = {}
        key_translater = {
            'person_source_value': 'PID',
            'drug_era_start_date': 'admdate',
            'drug_name': 'codes'
        }
        # all_codes = set()

        print("Parsing tsv data...")
        for rowid, row in tqdm(enumerate(rows)):
            # ! Some rows missed patient id, resulting in less elements for these rows, but still can be zipped
            if len(legend) != len(row):
                continue

            row_dict = {
                key_translater[k]: v for k, v in zip(legend, row) if k in key_translater.keys()
            }
            row_dict.update({'admid': str(rowid).zfill(10)})

            row_dict['code_type'] = 'ICD10'

            # missing or being the header line
            if row_dict["PID"] == 'NaN' or row_dict["codes"] == '' or row_dict["codes"] == 'ICDCode' or \
                    'Missing' in row_dict["codes"]:
                continue

            # ! Pay attention to the following date type change for correct mapping
            patient_id = int(float(row_dict['PID']))

            if patient_id in dob_dict:
                row_dict['DOB'] = dob_dict[patient_id]
                row_dict['DOD'] = dod_dict[patient_id]
                row_dict['gender'] = gender_dict[patient_id]
            else:
                row_dict['DOB'] = 'NA'
                row_dict['DOD'] = 'NA'
                row_dict['gender'] = 'NA'
            
            if patient_id not in metadata:
                metadata[patient_id] = {'events': []}
                metadata[patient_id]['events'].append(row_dict)
            else:
                metadata[patient_id]['events'].append(row_dict)

            # if args.library_path:
            #     all_codes.add(row_dict['codes'])

        print("Generating json data...")
        warning_flag = False
        for patient_id in tqdm(metadata.keys()):
            metadata[patient_id]['events'] = sorted(metadata[patient_id]['events'], key=lambda i: i['admdate'])
            if patient_id in tdt_dict:
                metadata[patient_id]['split_group'] = tdt_dict[patient_id] #! match icd data split
            else:
                metadata[patient_id]['split_group'] = 'NA'
            end_of_data = metadata[patient_id]['events'][0]['DOD'] \
                if metadata[patient_id]['events'][0]['DOD'] != 'NA' and \
                   metadata[patient_id]['events'][0]['DOD'] != 'None' \
                else ''
            metadata[patient_id]['end_of_data'] = end_of_data \
                if end_of_data != '' else max([e['admdate'] for e in metadata[patient_id]['events']])
            start_of_data = metadata[patient_id]['events'][0]['DOB'] \
                if metadata[patient_id]['events'][0]['DOB'] != 'NA' and \
                   metadata[patient_id]['events'][0]['DOB'] != 'None' \
                else ''
            metadata[patient_id]['birthdate'] = start_of_data \
                if start_of_data != '' else min([e['admdate'] for e in metadata[patient_id]['events']])
            metadata[patient_id]['gender'] = metadata[patient_id]['events'][0]['gender'] \
                if 'gender' in metadata[patient_id]['events'][0] and \
                   metadata[patient_id]['events'][0]['gender'] in ['M', 'F'] \
                else 'U'
            for e in metadata[patient_id]['events']:
                try:
                    del e['PID']
                    del e['DOD']
                    del e['DOB']
                    del e['gender']
                except KeyError:
                    if not warning_flag:
                        print("Warning: Date of birth, death, or other entries are missing!")
                        warning_flag = True
        assert not [p for p in metadata if any(['_' in e['admdate'] for e in metadata[p]['events']])]
        # return all_codes, metadata
        return metadata

def prep_demo(demo_filepath):
    demo = pd.read_csv(demo_filepath, engine='pyarrow', sep="\t")
    demo = demo[~demo['PatientICN'].isna()]  # Remove patient with na
    # ! Covert columns to proper data types for downstream use and storage as json format
    demo['PatientICN'] = demo['PatientICN'].astype(int)
    demo['BirthDate'] = demo['BirthDate'].astype(str)
    demo['DeathDate'] = demo['DeathDate'].astype(str)
    dob_dict = dict(zip(demo.PatientICN, demo.BirthDate))
    dod_dict = dict(zip(demo.PatientICN, demo.DeathDate))
    gender_dict = dict(zip(demo.PatientICN, demo.Gender))
    del demo
    gc.collect()
    return dob_dict, dod_dict, gender_dict

def get_tdt_group(metadata_json_filepath):
    """
    Match patient split group between drug data with ICD data
    """
    metadata_json = json.load(open(metadata_json_filepath,  'r'))
    tdt_dict = {int(k): metadata_json[k]['split_group'] for k in metadata_json}
    return tdt_dict

if __name__ == '__main__':
    args = parse_args()
    print("Preprocessing demo info ...")
    dob_dict, dod_dict, gender_dict = prep_demo(args.demo_path)
    print("Demo info processed!\n")

    files = os.listdir(args.tsv_dir)
    print(files)
    for fname in files:
        file_path = os.path.join(args.tsv_dir, fname)
        file_path_cohort = os.path.join(args.json_cohort_dir, fname.replace('tsv', 'json'))
        print("Processing {} with cohort {}".format(file_path, file_path_cohort))
        fname_out = fname.replace('tsv', 'json')
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
        out_path = os.path.join(args.out_dir, fname_out)
        print('Will save to {}'.format(out_path))
        # # with open(out_path, 'w') as fw:
        # #     assert os.path.isfile(out_path)
        # #     print('Will save to {}'.format(out_path))
        # #     pass

        tdt_dict = get_tdt_group(file_path_cohort)
        metadata = drug_tsv_to_json(file_path, dob_dict, dod_dict, gender_dict, tdt_dict)  # time-consuming!

        with open(out_path, 'w') as fw:
            print("Saving data to {}".format(out_path))
            json.dump(metadata, fw, indent=None, sort_keys=True)

        # if args.library_path:
        #     with open(args.library_path, 'w') as fw:
        #         pkl.dump(list(all_codes), fw)
    print('All done!')

# if __name__ == '__main__':
#     args = parse_args()
#     for root, subdirs, files in os.walk(args.tsv_path):
#         if len(files) > 0 and 'demo.tsv' not in files:
#             file_path = os.path.join(root, files[0])
#             print("Processing {}".format(file_path))
#             all_codes, metadata = tsv_to_json(file_path)
#             out_dir = root.replace("tsv", "json")
#             # print(out_dir)
#             if not os.path.exists(out_dir):
#                 os.makedirs(out_dir)
#             out_path = os.path.join(out_dir, files[0])
#             # print(out_path)
#             out_path = out_path.replace("tsv", "json")
#             # print(out_path)
#             with open(out_path, 'w') as fw:
#                 print("Save to {}".format(out_path))
#                 json.dump(metadata, fw, indent=None, sort_keys=True)
#             if args.library_path:
#                 with open(args.library_path, 'w') as fw:
#                     pkl.dump(list(all_codes), fw)

# # Generate small dataset for testing
# import pandas as pd
# tsv_fname = 'F:\\tmp_pancreatic\\temp_tsv\\global\\drug_split\\part_0.tsv'
# part_0 = pd.read_csv(tsv_fname, sep='\t', engine='pyarrow')
# part_t = part_0.sample(10000)
# tsv_out = 'F:\\tmp_pancreatic\\temp_tsv\\global\\tsv_test_drug.tsv'
# part_t.to_csv(tsv_out, sep='\t', index=False)
# # demo_path = 'F:\\tmp_pancreatic\\temp_tsv\\global\\raw\\demo.tsv'
# # demo = pd.read_csv(demo_path, sep='\t', engine='pyarrow')
# # demo.loc[demo['PatientICN'].isin([1000648986, 1000649372])]

# # # Testing using small dataset
# if __name__ == '__main__':
#     args = parse_args()
#     print("Preprocessing demo info ...")
#     dob_dict, dod_dict, gender_dict = prep_demo(args.demo_path)
#     print("Demo info processed!\n")

#     file_path = 'F:\\tmp_pancreatic\\temp_tsv\\global\\tsv_test_drug.tsv'
#     file_path_cohort = r'F:\tmp_pancreatic\temp_json\test\test_2\combined.json'
#     out_path = 'F:\\tmp_pancreatic\\temp_tsv\\global\\tsv_test_drug.json'

#     tdt_dict = get_tdt_group(file_path_cohort)
#     metadata = drug_tsv_to_json(file_path, dob_dict, dod_dict, gender_dict, tdt_dict)  # time-consuming!

#     with open(out_path, 'w') as fw:
#         print("Save to {}".format(out_path))
#         json.dump(metadata, fw, indent=None, sort_keys=True)

#     # if args.library_path:
#     #     with open(args.library_path, 'w') as fw:
#     #         pkl.dump(list(all_codes), fw)
#     print('All done!')
