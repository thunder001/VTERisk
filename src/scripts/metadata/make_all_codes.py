import pandas as pd
import pickle as pkl
import os
import multiprocessing as mp
import itertools


def get_codes(tsvfile, code_col, sep=','):
    dat = pd.read_csv(tsvfile, sep=sep, engine='pyarrow')
    # codes = set(dat['ICDCode'].tolist())
    # codes = set(dat['drug_name'].tolist())
    codes = set(dat[code_col].tolist())
    return codes

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

    icd9_phe_fpath = r'G:\FillmoreCancerData\chunlei\pancrisk\src\pancnet\data\phecode_icd9_map_unrolled.csv'
    icd10_phe_fpath = r'G:\FillmoreCancerData\chunlei\pancrisk\src\pancnet\data\Phecode_map_v1_2_icd10_beta.csv'
    save_path = 'F:\\tmp_pancreatic\\temp_tsv\\global\\all_observed_phe.pkl'
    codes1 = get_codes(icd9_phe_fpath, 'phecode')
    codes1 = set(map(float, codes1))
    print(len(codes1))
    codes2 = get_codes(icd10_phe_fpath, 'PHECODE')
    codes2.remove('')
    codes2 = set(map(float, codes2))
    print(len(codes2))
    all_codes = list(codes1.union(codes2))
    all_codes.append('VTE')
    print(len(all_codes))

    with open(save_path, 'wb') as fw:
        pkl.dump(all_codes, fw)