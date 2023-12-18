import pandas as pd
import random
import os
import json
import gc
import datetime
# from disrisknet.utils.date import parse_date
import copy

def split_file(big_file_path, split_dir, size_of_split=1000000):
    '''
    This function is used for split a big data file into smaller ones
    and save them into a give directory
    '''
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
    print("Loading big data.......\n")
    if 'tsv' in big_file_path:
        dat = pd.read_csv(big_file_path, sep='\t', engine='pyarrow')
        pats = dat['PatientICN'].unique().tolist()
    if 'json' in big_file_path:
        dat = json.load(open(big_file_path, 'r'))
        pats = list(dat.keys())
    random.shuffle(pats)
    num_of_pats = len(pats)
    num_of_split = num_of_pats // size_of_split if num_of_pats % size_of_split == 0 else len(pats) // size_of_split + 1
    print("Start splitting.......\n")
    for i in range(num_of_split):
        idx_start = i * size_of_split
        idx_end = min((i + 1) * size_of_split, num_of_pats)
        subpats = pats[idx_start:idx_end]
        if 'tsv' in big_file_path:
            subdat = dat.loc[dat.PatientICN.isin(subpats)]
            fname = ''.join(['part_', str(i), '.tsv'])
            split_path = os.path.join(split_dir, fname)
            subdat.to_csv(split_path, sep='\t', index=False)
        if 'json' in big_file_path:
            subdat = {k: dat[k] for k in subpats}
            filebase = os.path.split(big_file_path)[1].split('.')[0]
            fname = ''.join([filebase, str(i), '.json'])
            split_path = os.path.join(split_dir, fname)
            with open(split_path, 'w') as fw:
                print("Saving to {}\n".format(split_path))
                json.dump(subdat, fw, indent=None)

def file_split_drug(big_file_path, json_cohort_dir, split_dir):
    """
    Split big drug files into smaller files according to given patients cohorts
    """
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
    print("Loading big data.......\n")
    if 'tsv' in big_file_path:
        dat = pd.read_csv(big_file_path, sep='\t', engine='pyarrow')
        dat['person_source_value'] = dat['person_source_value'].astype(str)
    fnames = os.listdir(json_cohort_dir)
    for fname in fnames:
        print('Processing {}\n'.format(fname))
        file_path = os.path.join(json_cohort_dir, fname)
        cohort_dat = json.load(open(file_path, 'r'))
        cohort = list(cohort_dat.keys())

        subdat = dat.loc[dat.person_source_value.isin(cohort)]
        fname_base = fname.split('.')[0]
        fname = ''.join([fname_base, '.tsv'])
        split_path = os.path.join(split_dir, fname)
        subdat.to_csv(split_path, sep='\t', index=False)
    print("All done!")


def train_dev_test_split(metadir):
    '''
    This function automatically performs train/dev/test splits for multiple files in a directory.
    Note: each file in the directory contains 'split_group' key.
    '''
    par_dir = os.path.dirname(metadir)
    train_dir = os.path.join(par_dir, 'train')
    dev_dir = os.path.join(par_dir, 'dev')
    test_dir = os.path.join(par_dir, 'test')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(dev_dir):
        os.mkdir(dev_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    files = os.listdir(metadir)
    file_paths = [os.path.join(metadir, f) for f in files]
    file_paths = file_paths[1:]
    for i in range(len(file_paths)):
        print('Loading data: {}'.format(file_paths[i]))
        dat = json.load(open(file_paths[i], 'r'))

        train = {p: dat[p] for p in dat.keys() if dat[p]['split_group'] == 'train'}
        train_path = os.path.join(train_dir, files[i])
        with open(train_path, 'w') as fwt:
            print("Saving to {}\n".format(train_path))
            json.dump(train, fwt, indent=None)
        del train
        gc.collect()

        dev = {p: dat[p] for p in dat.keys() if dat[p]['split_group'] == 'dev'}
        dev_path = os.path.join(dev_dir, files[i])
        with open(dev_path, 'w') as fwd:
            print("Saving to {}\n".format(dev_path))
            json.dump(dev, fwd, indent=None)
        del dev
        gc.collect()

        test = {p: dat[p] for p in dat.keys() if dat[p]['split_group'] == 'test'}
        test_path = os.path.join(test_dir, files[i])
        with open(test_path, 'w') as fwtt:
            print("Saving to {}\n".format(test_path))
            json.dump(test, fwtt, indent=None)
        del test
        gc.collect()

def train_dev_test_file_split(metafile,  sampling=True, train_size=8000, dev_size=1000, test_size=1000):
    '''
    This function has two purposes: 1)perform train/dev/test split from a given json file with 'split_group' key.
    2) perform sampling from each group, which is useful to prepare small subset for testing new functionalities.
    '''
    par_dir = os.path.dirname(metafile)
    # train_dir = os.path.join(par_dir, 'train')
    # dev_dir = os.path.join(par_dir, 'dev')
    # test_dir = os.path.join(par_dir, 'test')
    train_dir = os.path.join(par_dir, 'train-10000')
    dev_dir = os.path.join(par_dir, 'dev-10000')
    test_dir = os.path.join(par_dir, 'test-10000')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(dev_dir):
        os.mkdir(dev_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    
    print('Loading data: {}'.format(metafile))
    dat = json.load(open(metafile, 'r'))

    # Train sampling
    train_all = [p for p in dat.keys() if dat[p]['split_group'] == 'train']
    if sampling:
        if len(train_all) <= train_size:
            train_sample = train_all
        else:
            train_sample = random.sample(train_all, train_size)
    else:
        train_sample = train_all
    train = {p: dat[p] for p in train_sample}
    train_path = os.path.join(train_dir, 'train.json')
    with open(train_path, 'w') as fwt:
        print("Saving to {}\n".format(train_path))
        json.dump(train, fwt, indent=None)
    del train
    gc.collect()

    # Dev sampling
    dev_all = [p for p in dat.keys() if dat[p]['split_group'] == 'dev']
    if sampling:
        if len(dev_all) <= dev_size:
            dev_sample = dev_all
        else:
            dev_sample = random.sample(dev_all, dev_size)
    else:
        dev_sample = dev_all
    dev = {p: dat[p] for p in dev_sample}
    dev_path = os.path.join(dev_dir, 'dev.json')
    with open(dev_path, 'w') as fwd:
        print("Saving to {}\n".format(dev_path))
        json.dump(dev, fwd, indent=None)
    del dev
    gc.collect()

    # Test sampling
    test_all = [p for p in dat.keys() if dat[p]['split_group'] == 'test']
    if sampling:
        if len(test_all) <= test_size:
            test_sample = test_all
        else:
            test_sample = random.sample(test_all, test_size)
    else:
        test_sample = test_all
    test = {p: dat[p] for p in test_sample}
    test_path = os.path.join(test_dir, 'test.json')
    with open(test_path, 'w') as fwtt:
        print("Saving to {}\n".format(test_path))
        json.dump(test, fwtt, indent=None)
    del test
    gc.collect()

def train_dev_test_file_random_split(metafile, train_size=0.8, dev_size=0.1, test_size=0.1):
    '''
    This function randomly splits a json file into train, dev and test datasets, and 
    automatically saved splitted files under train, dev and test folders.
    Note: there is special handling for non-string type diagnosis codes when necessary
    '''
    par_dir = os.path.dirname(metafile)
    train_dir = os.path.join(par_dir, 'train')
    dev_dir = os.path.join(par_dir, 'dev')
    test_dir = os.path.join(par_dir, 'test')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(dev_dir):
        os.mkdir(dev_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    
    print('Loading data: {}'.format(metafile))
    dat = json.load(open(metafile, 'r'))

    # change phecodes datatype to string
    for p in dat.keys():
        for i in range(len(dat[p]['events'])):
            dat[p]['events'][i]['codes'] = str(dat[p]['events'][i]['codes'])

    all = [p for p in dat.keys()]
    random.shuffle(all)
    dev_ind_start = int(len(all) * train_size)
    test_ind_start = int(len(all) * (train_size + dev_size))

    # Train sampling
    train_sample = all[:dev_ind_start]
    train = {p: dat[p] for p in train_sample}
    for p in train:
        train[p]['split_group'] = 'train'
    train_path = os.path.join(train_dir, 'train.json')
    with open(train_path, 'w') as fwt:
        print("Saving to {}\n".format(train_path))
        json.dump(train, fwt, indent=None)
    del train
    gc.collect()

    # Dev sampling
    dev_sample = all[dev_ind_start:test_ind_start]
    dev = {p: dat[p] for p in dev_sample}
    for p in dev:
        dev[p]['split_group'] = 'dev'
    dev_path = os.path.join(dev_dir, 'dev.json')
    with open(dev_path, 'w') as fwd:
        print("Saving to {}\n".format(dev_path))
        json.dump(dev, fwd, indent=None)
    del dev
    gc.collect()

    # Test sampling
    test_sample = all[test_ind_start:]
    test = {p: dat[p] for p in test_sample}
    for p in test:
        test[p]['split_group'] = 'test'

    test_path = os.path.join(test_dir, 'test.json')
    with open(test_path, 'w') as fwtt:
        print("Saving to {}\n".format(test_path))
        json.dump(test, fwtt, indent=None)
    del test
    gc.collect()

def combine_json_files(json_dir, files_to_combine, combined_dir):
    fnames = os.listdir(json_dir)[:files_to_combine]
    print(fnames)
    json_paths = [os.path.join(json_dir, file) for file in fnames]
    combined_dict = {}
    for json_path in json_paths:
        print('Combining {} ...'.format(json_path))
        dat = json.load(open(json_path, 'r'))
        combined_dict.update(dat)
    combined_path = os.path.join(combined_dir, 'combined.json')
    print('Dumping combined data ....')
    json.dump(combined_dict, open(combined_path, 'w'))
    print('All done!')

def combine_json_files_2(json_paths, combined_dir):
    combined_dict = {}
    for json_path in json_paths:
        print('Combining {} ...'.format(json_path))
        dat = json.load(open(json_path, 'r'))
        combined_dict.update(dat)
    combined_path = os.path.join(combined_dir, 'combined.json')
    print('Dumping combined data ....')
    json.dump(combined_dict, open(combined_path, 'w'))
    print('All done!')
    
def combine_icd_drug(phe_json, drug_json):
    '''
    phe_json: 
    drug_json:
    '''
    combined_dict = {}
    phe = json.load(open(phe_json, 'r'))
    drug = json.load(open(drug_json, 'r'))
    phe_cohort = set(phe.keys())
    drug_cohort = set(drug.keys())
    # pdb.set_trace()
    union_cohort = phe_cohort | drug_cohort
    intersect_cohort = phe_cohort & drug_cohort
    phe_cohort_only = phe_cohort - drug_cohort
    drug_cohort_only = drug_cohort - phe_cohort
    print(f'There are {len(phe_cohort)} and {len(drug_cohort)} in icd cohort and drug cohort.')
    print(f'Union: {len(union_cohort)}; Intersect: {len(intersect_cohort)}\nICD only: {len(phe_cohort_only)}; drug only {len(drug_cohort_only)}')
    
    for pat in union_cohort:
        if pat in phe_cohort_only:
            combined_dict[pat] = phe[pat]
        # if pat in drug_cohort_only:
        #     combined_dict[pat] = drug[pat]
        if pat in intersect_cohort:
            pat_dict = {}

            phe_events = phe[pat]['events']
            phe_end_of_data = phe[pat]['end_of_data']
            drug_events = drug[pat]['events']
            drug_end_of_data = drug[pat]['end_of_data']
            # In case duplicated events, especially for pancreatic cancer patients
            # pdb.set_trace()
            if len(phe_events) > 0 and len(drug_events) > 0:
                phe_events.extend(drug_events)
                # events_set = set(frozenset(e.items()) for e in icd_events if e is not None)
                # events = [dict(s) for s in events_set]
                end_of_data = max(phe_end_of_data, drug_end_of_data)
                pat_dict['events'] = phe_events
                pat_dict['end_of_data'] = end_of_data
                for key in ['birthdate', 'gender', 'split_group', 'indexdate', 'ks_mod_score']:
                    pat_dict[key] = phe[pat][key]
                combined_dict[pat] = pat_dict

    return combined_dict

if __name__ == '__main__':
    
    # ------ VTE file splits -----
    # metafile = 'F:\\tmp_pancreatic\\temp_json\\test\\vte\\data-500.json'
    # train_dev_test_random_split(metafile)

    # metafile = 'F:\\tmp_pancreatic\\temp_json\\test\\vte\\data-10000.json'
    # train_dev_test_file_split(metafile)
    # metafile = 'F:\\tmp_pancreatic\\temp_json\\test\\vte\\data_100000_final.json'
    # train_dev_test_file_split(metafile)

    # metafile = 'F:\\tmp_pancreatic\\temp_json\\test\\vte\\data-complete.json'
    # metafile = 'F:\\tmp_pancreatic\\temp_json\\test\\vte\\data-081823.json'
    # metafile = 'F:\\tmp_pancreatic\\temp_json\\test\\vte\\data-090123.json'
    # metafile = 'F:\\tmp_pancreatic\\temp_json\\test\\vte\\data-090123-ks-2.json'
    # metafile = 'F:\\tmp_pancreatic\\temp_json\\test\\vte\\data-10000-ks-2.json'
    # metafile = 'F:\\tmp_pancreatic\\temp_json\\test\\vte\\data-100223-final.json'
    # train_dev_test_file_split(metafile, sampling=False)
    # train_dev_test_file_random_split(metafile)
    # metafile = 'F:\\tmp_pancreatic\\temp_json\\test\\vte\\data_100223_final.json'
    # train_dev_test_file_random_split(metafile)

    # ---- Add ks score to VTE dataset ------
    # metafile = 'F:\\tmp_pancreatic\\temp_json\\test\\vte\\data-10000.json'
    # ks_feather = 'F:\\tmp_pancreatic\\temp_fst\\global\\raw\\analytic_final_2000-2021.feather'
    # new_metafile = 'F:\\tmp_pancreatic\\temp_json\\test\\vte\\data-10000-ks-2.json'
    # add_ks_score(metafile, ks_feather, new_metafile)
    
    
    # ---- Add ks score to VTE dataset ------
    # metafile = 'F:\\tmp_pancreatic\\temp_json\\test\\vte\\data-090123.json'
    # ks_feather = 'F:\\tmp_pancreatic\\temp_fst\\global\\raw\\analytic_final_2000-2021.feather'
    # new_metafile = 'F:\\tmp_pancreatic\\temp_json\\test\\vte\\data-090123-ks-2.json'
    # add_ks_score(metafile, ks_feather, new_metafile)

    # ----- Combine phe and drug -----
    # phe_path = 'F:\\tmp_pancreatic\\temp_json\\global\\vte\\phe\\data_100223_final.json'
    # drug_path = 'F:\\tmp_pancreatic\\temp_json\\global\\vte\\drug\\medication.json'
    # combined_dict = combine_icd_drug(phe_path, drug_path)
    # combined_path = 'F:\\tmp_pancreatic\\temp_json\\global\\vte\\phe_drug\\combined.json'

    # print('Dumping combined data ....')
    # json.dump(combined_dict, open(combined_path, 'w'))
    # print('All done!')

    metafile = 'F:\\tmp_pancreatic\\temp_json\\global\\vte\\phe_drug\\combined_final.json'
    # train_dev_test_file_random_split(metafile)

    train_dev_test_file_split(metafile)