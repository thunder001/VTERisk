import pandas as pd
import random
import os
import json
import gc
import datetime
from pancnet.utils.date import parse_date

def find_patients_with_vte_phecode(dat_dict):
    patients = []
    val_patients = []
    for pat in dat_dict:
        pat_dict = {}
        pat_dict['patientid'] = pat
        pat_dict['indexdate'] = dat_dict[pat]['indexdate']
        pat_dict['vte_events'] = []
        for event in dat_dict[pat]['events']:
            # print(event)
            if '452' in event['codes']:
                pat_dict['vte_events'].append(event)

        if len(pat_dict['vte_events']) >= 1:
            patients.append(pat_dict)
            for vte_event in pat_dict['vte_events']:
                if parse_date(vte_event['admdate']) > parse_date(pat_dict['indexdate']):
                    val_patients.append(pat_dict)
                    break   
    print(len(patients))
    print(len(val_patients))
    return patients, val_patients


def split_file(big_file_path, split_dir, size_of_split=1000000):
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
    
    


# demo_path = 'F:\\tmp_pancreatic\\temp_tsv\\global\\raw\\demo.tsv'
# demo_split_dir = 'F:\\tmp_pancreatic\\temp_tsv\\global\\split'
# split_file(demo_path, demo_split_dir)

# icd_path = 'F:\\tmp_pancreatic\\temp_tsv\\global\\icd_code_counts_final.tsv'
# icd_split_dir = 'F:\\tmp_pancreatic\\temp_tsv\\global\\icd_split'
# split_file(icd_path, icd_split_dir)


def train_dev_test_split(metadir):
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

def train_dev_test_random_split(metafile, train_size=0.8, dev_size=0.1, test_size=0.1):
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

if __name__ == '__main__':
    # json_dir = 'F:\\tmp_pancreatic\\temp_json\\global\\meta_split'
    # files = os.listdir(json_dir)
    # files = files[2:8]
    # for file in files:
    #     json_path = os.path.join(json_dir, file)
    #     json_split_dir = 'F:\\tmp_pancreatic\\temp_json\\global\\meta_split_level_2'
    #     print('{} will be split and save to the directory: \n{}'.format(json_path, json_split_dir))
    #     split_file(json_path, json_split_dir, size_of_split=100000)
    
    # metadir = 'F:\\tmp_pancreatic\\temp_json\\global\\meta_split'
    # train_dev_test_split(metadir)

    # json_dir = 'F:\\tmp_pancreatic\\temp_json\\test\\test'
    # files_to_combine = 2
    # combined_dir = 'F:\\tmp_pancreatic\\temp_json\\test\\test_2'
    # combine_json_files(json_dir, files_to_combine, combined_dir)

    # json_dir = 'F:\\tmp_pancreatic\\temp_json\\global\\train'
    # files_to_combine = 3
    # combined_dir = 'F:\\tmp_pancreatic\\temp_json\\global\\train_2'
    # combine_json_files(json_dir, files_to_combine, combined_dir)

    # json_dir = 'F:\\tmp_pancreatic\\temp_json\\global\\dev'
    # files_to_combine = 3
    # combined_dir = 'F:\\tmp_pancreatic\\temp_json\\global\\dev_2'
    # combine_json_files(json_dir, files_to_combine, combined_dir)

    # json_dir = 'F:\\tmp_pancreatic\\temp_json\\global\\test'
    # files_to_combine = 3
    # combined_dir = 'F:\\tmp_pancreatic\\temp_json\\global\\test_2'
    # combine_json_files(json_dir, files_to_combine, combined_dir)

    # json_path_1 = 'F:\\tmp_pancreatic\\temp_json\\global\\train_2\\combined.json'
    # json_path_2 = 'F:\\tmp_pancreatic\\temp_json\\global\\dev_2\\combined.json'
    # json_path_3 = 'F:\\tmp_pancreatic\\temp_json\\global\\test_2\\combined.json'
    # json_paths = [json_path_1, json_path_2, json_path_3]
    # combined_dir = 'F:\\tmp_pancreatic\\temp_json\\global\\meta_3m'
    # combine_json_files_2(json_paths, combined_dir)

    # big_file_path = r'F:\tmp_pancreatic\temp_tsv\global_drug\drugs.tsv' 
    # json_cohort_dir = r'F:\tmp_pancreatic\temp_json\global\icd_split'
    # split_dir = r'F:\tmp_pancreatic\temp_tsv\global\drug_split'
    # file_split_drug(big_file_path, json_cohort_dir, split_dir)

    # metadir = 'F:\\tmp_pancreatic\\temp_json\\global\\drug_split_panc'
    # train_dev_test_split(metadir)

    # metafile = 'F:\\tmp_pancreatic\\temp_json\\test\\drug\\part_0.json'
    # train_dev_test_split(metafile)

    # metafile = 'F:\\tmp_pancreatic\\temp_json\\test\\vte\\data-500.json'
    # train_dev_test_random_split(metafile)

    # metafile = 'F:\\tmp_pancreatic\\temp_json\\test\\vte\\data-10000.json'
    # train_dev_test_file_split(metafile)

    metafile = 'F:\\tmp_pancreatic\\temp_json\\test\\vte\\data-complete.json'
    train_dev_test_file_split(metafile, sampling=False)
