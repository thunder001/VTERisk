import json
from disrisknet.utils.parsing import md5
from disrisknet.utils.parsing import get_code
import os
import pickle
import tqdm
from collections import Counter, defaultdict
import pandas as pd

NO_DATASET_ERR = "Dataset {} not in DATASET_REGISTRY! Available datasets are {}"
DATASET_REGISTRY = {}
NUM_PICKLES = 50
UNK_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'
NO_OP_TOKEN = '_'

def RegisterDataset(dataset_name):
    """Registers a dataset."""

    def decorator(f):
        DATASET_REGISTRY[dataset_name] = f
        return f

    return decorator


def get_dataset_class(args):
    if args.dataset not in DATASET_REGISTRY:
        raise Exception(
            NO_DATASET_ERR.format(args.dataset, DATASET_REGISTRY.keys()))

    return DATASET_REGISTRY[args.dataset]


def build_code_to_index_map(args):
    
    print("Building code to index map")
    pkl_name = 'data/all_observed_{}.pkl'.format(args.disease_code_system)
    # pkl_name = 'data/all_observed_phe_wo_vte.pkl'

    with open(pkl_name, 'rb') as f:
        pkl_codes = pickle.load(f)
        if '' in pkl_codes:
            pkl_codes.remove('')
    if 'icd' in args.disease_code_system:
        all_observed_codes = [get_code(args, code) for code in pkl_codes]
    else:
        all_observed_codes = pkl_codes
    print("Length of all_observed", len(all_observed_codes))
    all_codes_counts = dict(Counter(all_observed_codes))
    print(len(all_codes_counts))
    all_codes = list(all_codes_counts.keys())
    all_codes_p = list(all_codes_counts.values())
    all_codes_p = [i/sum(all_codes_p) for i in all_codes_p]
    code_to_index_map = {code:indx+1 for indx, code in enumerate(all_codes)}
    code_to_index_map.update({PAD_TOKEN: 0, UNK_TOKEN: len(code_to_index_map)+1, NO_OP_TOKEN: len(code_to_index_map)+2})
    args.code_to_index_map = code_to_index_map
    args.all_codes = all_codes
    args.all_codes_p = all_codes_p


def get_level3icd10_from_icd9(icd9, df_map, args):
    icd10 = df_map[df_map.icd9cm == icd9].icd10cm.tolist()
    #get level 3
    icd10_level3 = [get_code(args, 'D' + icd) for icd in icd10]
    if len(set(icd10_level3))==1:
        return icd10_level3[0]
    elif icd10_level3:
        return '-'.join(icd10_level3)
    else:
        return ''


def get_dk_compatible_icd10code(rpdr_code, df_map, args):
    if rpdr_code[0].isdigit():
        return get_level3icd10_from_icd9(rpdr_code, df_map, args)
    else:
        return get_code(args, 'D' + rpdr_code)


def map_code_to_index_to_usa(args):
    USA_ICD_CODES_PATH = "data/all_observed_icd.pkl"
    MAPPER_ICD10_ICD9 = "data/icd10cmtoicd9gem.txt"
    MAPPER_ICD9_ICD8 = "data/pcbi.1007927.s015.csv"  # TODO: These files should been renamed
    icd_codes = pickle.load(open(USA_ICD_CODES_PATH, 'rb'))
    icd10toicd9 = pd.read_csv(MAPPER_ICD10_ICD9,names=["icd9cm", "icd10cm", "choice"],)
    icd_codes_clean = [icd.replace('.', '') for icd in icd_codes if len(icd) > 0]
    print(icd_codes_clean[:10])
    print("Generating code map projection from [DK] to [USA] code system...")
    icd10_for_dk = [get_dk_compatible_icd10code(code, icd10toicd9, args) for code in tqdm.tqdm(icd_codes_clean)
                    if len(code) > 0]
    print (f"Number of codes processed {len(icd10_for_dk)}")
    print (f"Number of ICD9 codes mapping to more than one ICD10: {len([c for c in icd10_for_dk if '-' in c])}")
    print (f"Number of ICD9 codes not mapping to any ICD10: {len([c for c in icd10_for_dk if c==''])}")

    usa2dk = dict(zip(icd_codes_clean, icd10_for_dk))
    usa_mapped_code_to_index_map = defaultdict(lambda : 0)  # TODO: change this to nonzero for UNK codes?
    missing_icd10_matches = 0
    for el in icd_codes_clean:
        if usa2dk[el] in args.code_to_index_map:
            usa_mapped_code_to_index_map[get_code(args, el)] = args.code_to_index_map[usa2dk[el]]
        else:
            missing_icd10_matches += 1
            usa_mapped_code_to_index_map[get_code(args, el)]
    print (f"Number of codes icd10 cm not matching DK code to index map {missing_icd10_matches}")
    usa_mapped_code_to_index_map[UNK_TOKEN]
    usa_mapped_code_to_index_map[PAD_TOKEN]
    return dict(usa_mapped_code_to_index_map)


# Depending on arg, build dataset
def get_dataset(args):

    # metadata = json.load(open(args.metadata_path, 'r'))
    dataset_class = get_dataset_class(args)

    if args.train:
        fname = os.listdir(args.train_data_dir)[args.data_file_idx]
        data_path = os.path.join(args.train_data_dir, fname)
        metadata = json.load(open(data_path, 'r'))
        train = dataset_class(metadata, args, 'train')
        del metadata
        
    else:
        train = []

    if args.train or args.dev:
        fname = os.listdir(args.dev_data_dir)[args.data_file_idx]
        data_path = os.path.join(args.dev_data_dir, fname)
        metadata = json.load(open(data_path, 'r'))
        dev = dataset_class(metadata, args, 'dev')
        del metadata
    else:
        dev = []
    
    if args.test:
        fname = os.listdir(args.test_data_dir)[args.data_file_idx]
        data_path = os.path.join(args.test_data_dir, fname)
        metadata = json.load(open(data_path, 'r'))
        test = dataset_class(metadata, args, 'test')
        del metadata
        if args.attribute:
            attribution_set = test
            attribution_set.split_group = "attribute"
        else: 
            attribution_set = []
    else:
        test = []
        attribution_set = []
    
    if args.attribute:
        fname = os.listdir(args.test_data_dir)[args.data_file_idx]
        data_path = os.path.join(args.test_data_dir, fname)
        metadata = json.load(open(data_path, 'r'))
        attribution_set = dataset_class(metadata, args, 'test')
        attribution_set.split_group = "attribute"
    else:
        attribution_set=[]
    
    if args.cross_eval:
        cross_set = dataset_class(metadata, args, 'all')
    else:
        cross_set = []

    code_to_index_path = os.path.join(args.model_dir, '.code_map')
    if args.train:
        # Build a new code to index map only during training.
        build_code_to_index_map(args)
        json.dump(args.code_to_index_map, open(code_to_index_path, 'w'))
    else:
        if args.map_to_icd_system == "usa":
            args.code_to_index_map = map_code_to_index_to_usa(args)
        elif args.map_to_icd_system == "dk":
            raise NotImplementedError
        elif 'code_to_index_map' in args:
            print("Warning! No map_to_icd_system is used, using the previous map from result files directly.")
            pass
        elif 'code_to_index_file' in args:
            print("Warning! No map_to_icd_system is used, using specified code_to_index json file..")
            args.code_to_index_map = json.load(open(args.code_to_index_file, 'r'))
        else:
            print("Warning! No map_to_icd_system is used and no previous map is found from result files. Loading from separate files.")
            args.code_to_index_map = json.load(open(code_to_index_path + '.code_map', 'r'))

    # args.index_map_length = len(args.code_to_index_map)

    if args.max_events_length == None:
        args.max_events_length = max([len(record['codes']) for record in train.dataset])

    if args.pad_size is None:
        args.pad_size = args.max_events_length

    args.PAD_TOKEN = PAD_TOKEN
    return train, dev, test, attribution_set, cross_set, args
