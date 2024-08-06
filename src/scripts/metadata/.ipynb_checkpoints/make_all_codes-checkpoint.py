import pandas as pd
import pickle as pkl
import os
import multiprocessing as mp
import itertools
import pdb

def get_codes(tsvfile, code_col, code_type, sep=','):
    dat = pd.read_csv(tsvfile, sep=sep, engine='pyarrow')
    dat = dat.astype({code_col: str})
    # codes = set(dat['ICDCode'].tolist())
    # codes = set(dat['drug_name'].tolist())
    codes = list(set(dat[code_col].tolist()))
    if '' in codes:
        codes.remove('')
    return {code_type: codes}

if __name__ == '__main__':
    # tsv_dir = r'F:\tmp_pancreatic\temp_tsv\global\icd_split'
    # save_path = 'F:\\tmp_pancreatic\\temp_tsv\\global\\all_observed_icd.pkl'
    # tsv_dir = r'F:\tmp_pancreatic\temp_tsv\global\drug_split'
    # save_path = 'F:\\tmp_pancreatic\\temp_tsv\\global\\all_observed_drug.pkl'
    # files = os.listdir(tsv_dir)
    # file_paths = [os.path.join(tsv_dir, file) for file in files]
    # print(file_paths)
    # with mp.Pool(8) as p:
    #     all_codes_list = p.map(get_codes, file_paths)
    #     all_codes = set(list(itertools.chain.from_iterable(all_codes_list)))
    #     print(len(all_codes))

    # with open(save_path, 'wb') as fw:
    #     pkl.dump(list(all_codes), fw)

    ### --------- phe --------------------
    icd9_phe_fpath = r'G:\FillmoreCancerData\chunlei\VTERISK\src\data\phecode_icd9_map_unrolled.csv'
    icd10_phe_fpath = r'G:\FillmoreCancerData\chunlei\VTERISK\src\data\Phecode_map_v1_2_icd10_beta.csv'
    save_path = r'G:\FillmoreCancerData\chunlei\VTERISK\src\data\all_observed_phe_2.pkl'
    # pdb.set_trace()
    codes1 = get_codes(icd9_phe_fpath, 'phecode', 'phe')
    # codes1 = set(map(float, codes1))
    print(len(codes1['phe']))
    codes2 = get_codes(icd10_phe_fpath, 'PHECODE', 'phe')
    # codes2 = set(map(float, codes2))
    print(len(codes2['phe']))

    all_codes = list(set(codes1['phe']).union(set(codes2['phe'])))
    all_codes.append('VTE')

    all_codes_dict = {'phe': all_codes}
    print(len(all_codes_dict['phe']))
    
    with open(save_path, 'wb') as fw:
        pkl.dump(all_codes_dict, fw)

    ### ---------- drug -------------------
    drug_path = r'f:\tmp_pancreatic\temp_tsv\global\vte\systemic_treatment.tsv'
    save_path = r'G:\FillmoreCancerData\chunlei\VTERISK\src\data\all_observed_drug_2.pkl'
    all_codes_dict = get_codes(drug_path, 'Generic', 'drug', sep='\t')
    print(len(all_codes_dict['drug']))
    
    with open(save_path, 'wb') as fw:
        pkl.dump(all_codes_dict, fw)