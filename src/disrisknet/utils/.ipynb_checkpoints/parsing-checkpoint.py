import hashlib
import torch
import argparse
import pandas as pd
import sys
import pickle
import disrisknet.learn.state_keeper as state

POSS_VAL_NOT_LIST = 'Flag {} has an invalid list of values: {}. Length of list must be >=1'
ICD10_MAPPER_NAME = "src/data/icd10_eng_diag_chapters.tsv"   # TODO: These files should been renamed
ICD8_MAPPER_NAME = "src/data/icd8_eng.tsv"
ICD9_MAPPER_NAME = "src/data/icd9_disease_descriptions.tsv"
ICD10TO9_MAPPER_NAME = "src/data/icd10cmtoicd9gem.txt"

CODEDF10 = pd.read_csv(ICD10_MAPPER_NAME, sep='\t', header=None,
    names=['code', 'description', 'chapter', 'chapter_name', 'block_name', 'block']
    )

CODE_DF_10TO9 = pd.read_csv(ICD10TO9_MAPPER_NAME, sep=',', header=None,
    names=['ICD9', 'ICD10', 'unk']
)
map_9_to_10 = {c9:c10 for c9, c10 in zip(CODE_DF_10TO9['ICD9'], CODE_DF_10TO9['ICD10'])}
map_10_to_description = {c:name for c, name in zip(CODEDF10['code'], CODEDF10['description'])}
def map_10_to_lv3_description(icd10_code):
    if icd10_code[:3] in map_10_to_description:
        return map_10_to_description[icd10_code[:3]]
    else:
        return icd10_code

CODEDF_9 = pd.read_csv(ICD9_MAPPER_NAME, sep='\t', header=0, names=['code_long', 'description', 'shorter description', "NA"])
CODEDF_9['code_icd10'] = CODEDF_9['code_long'].map(map_9_to_10)
CODEDF_9['code'] = CODEDF_9['code_long'].map(lambda x: x[:3])
CODEDF_9['description'] = CODEDF_9['code_icd10'].map(map_10_to_lv3_description)
CODEDF_9_lv3 = CODEDF_9[['code', 'description']].drop_duplicates()
CODEDF_9_lv3 = CODEDF_9_lv3.groupby('code').agg({
    'description': (lambda x: "  \n".join(x.values))
}).reset_index()
CODEDF_9 = CODEDF_9[['code_long', 'description']]
CODEDF_9 = CODEDF_9.rename(columns={'code_long':'code'})

CODEDF_w_9 = pd.concat([CODEDF10, CODEDF_9, CODEDF_9_lv3])
CODEDF = CODEDF_w_9
CODE2DESCRIPTION = (dict(zip(CODEDF_w_9.code, CODEDF_w_9.description)))


def parse_args(args_str=None):
    parser = argparse.ArgumentParser(description='DiskRisk Classifier')
    # What to execute
    parser.add_argument('--train', action='store_true', default=False, help='Whether or not to train model')
    parser.add_argument('--test', action='store_true', default=False, help='Whether or not to run model on test set')
    parser.add_argument('--dev', action='store_true', default=False, help='Whether or not to run model on dev set')
    parser.add_argument('--attribute', action='store_true', default=False,
                        help='Whether or not to run interpretation on test set')
    parser.add_argument('--cross_eval', action='store_true', default=False,
                        help='Whether or not to run cross evaluation')
    # Device level information
    parser.add_argument('--cuda', action='store_true', default=True, help='enable the gpu. Only relevant for NNs')
    parser.add_argument('--device', type=str, default='cuda:0', help='specify which gpu will be used')
    parser.add_argument('--num_workers', type=int, default=3, help='num workers for each data loader [default: 4]')
    # Dataset setup
    # TODO: remember to put --dataset in use
    parser.add_argument('--dataset', type=str, default='bth_disease_progression', help="Name of dataset to use.")
    parser.add_argument('--train_data_dir', type=str,
                        default='F:\\tmp_pancreatic\\temp_json\\global\\train',
                        help="Folder of source training datafiles")
    parser.add_argument('--dev_data_dir', type=str,
                        default='F:\\tmp_pancreatic\\temp_json\\global\\dev',
                        help="Folder of source dev datafiles")
    parser.add_argument('--test_data_dir', type=str,
                        default='F:\\tmp_pancreatic\\temp_json\\global\\test',
                        help="Folder of source test datafiles")
    parser.add_argument('--data_file_idx', type=int, default=0, help="Specify which data file will be used")
    parser.add_argument('--pred_day', action='store_true', default=True, help="Specify which data file will be used")
    parser.add_argument('--day_endpoints', nargs='+', type=int, default=[7, 14], help="List of month endpoints at which to predict risk")
    parser.add_argument('--month_endpoints', nargs='+', type=int, default=[3, 6], help="List of month endpoints at which to predict risk")
    parser.add_argument('--code_to_index_file', type=str, default='', help="File with code to index information")
    parser.add_argument('--pad_size', type=int, default=100, help="Padding the trajectories to how long for training. Default: use pad_size defined in dataset.")
    parser.add_argument('--use_only_diseases', action='store_true', default=False, help="use only disease code as event")
    parser.add_argument('--use_no_op_token', action='store_true', default=False, help="if true use no op token")
    parser.add_argument('--disease_code_system', type=str, default='phe', help="Which coding system is used. ['icd', 'chapter', 'block', 'drug', 'phe']")
    parser.add_argument('--icd10_level', type=int, default=3, help="Which level of ICD10 code is used, default: 3, i.e. DXXX.")
    parser.add_argument('--icd8_level', type=int, default=3, help="Which level of ICD8 code is used, default: 3, i.e. XXX.")
    parser.add_argument('--use_char_embedding', action='store_true', default=False, help='If to enable char embeddings. In this case, use GRU to charecters in code sequentially. Char embedding is then concat to code-word embedding')
    parser.add_argument('--char_dim', type=int, default=25, help="Dimension to use for char embedding.")
    parser.add_argument('--max_events_length', type=int, default=200, help="Max num events to use (pkl generation)")
    parser.add_argument('--min_events_length', type=int, default=5, help="Min num events to include a patient")
    parser.add_argument('--max_year_before_index', type=int, default=5, help="Min num events to include a patient")
    parser.add_argument('--exclusion_interval', type=int, default=0, help="Exclude events before end of trajectory, default: 0 (month).")
    parser.add_argument('--code_to_index_map', type=str, default=None, help='filename of model code_to_index_map to load[default: None]')

    parser.add_argument('--baseline', action='store_true', default=False, help="Path of baseline diseases json file")
    parser.add_argument('--no_random_sample_eval_trajectories', action='store_true', default=False, help="During dev and test sample the trajectories randomly")
    parser.add_argument('--max_eval_indices', type=int, default=10, help="Max number of indices to include for each pt. during eval")
    parser.add_argument('--confusion', type=str, default=None, help="Adding noise in data. [None(default), 'shuffle', 'replace', 'outcome']")
    parser.add_argument('--confusion_strength', type=float, default=0, help="Noise level, 0: default, no noise; 1: completely random")
    parser.add_argument('--subgroup_validation', type=str, default='random', help="selct which subgroup to use for testing")
    # Model Hyper-params
    # TODO: use_time_embed and use_age_embed should be set to true by default
    parser.add_argument('--model_name', type=str, default='transformer', help="Model to be used.")
    parser.add_argument('--num_layers', type=int, default=1, help="Number of layers to use for seq encoder.")
    parser.add_argument('--num_heads', type=int, default=16, help="Number of heads to use for multihead attention. Only relevant for transformer.")
    parser.add_argument('--hidden_dim', type=int, default=128, help="Representation size at end of network.")
    parser.add_argument('--pool_name', type=str, default='GlobalAvgPool', help='Pooling mechanism')
    parser.add_argument('--dropout', type=float, default=0, help="Dropout value for the neural network model.")
    parser.add_argument('--neuron_norm', action='store_true', default=False, help='Wether or not to condition embeddings by their relative time.')
    parser.add_argument('--use_time_embed', action='store_true', default=False, help='Wether or not to condition embeddings by their relative time.')
    parser.add_argument('--use_age_in_cox', action='store_true', default=False, help='Wether or not to implement age directly into baseline models.')
    parser.add_argument('--add_age_neuron', action='store_true', default=False, help='Wether or not to add age neuron in abstract risk model')
    parser.add_argument('--add_ks_neuron', action='store_true', default=False, help='Wether or not to add ks neuron in abstract risk model')
    parser.add_argument('--add_sex_neuron', action='store_true', default=False, help='Wether or not to add sex neuron in abstract risk model')
    parser.add_argument('--add_bmi_neuron', action='store_true', default=False, help='Wether or not to add bmi neuron in abstract risk model')
    parser.add_argument('--add_race_neuron', action='store_true', default=False, help='Wether or not to add bmi neuron in abstract risk model')
        
    parser.add_argument('--use_age_embed', action='store_true', default=False, help='Wether or not to condition embeddings by their relative time.')
    parser.add_argument('--use_dxtime_embed', action='store_true', default=False, help='Wether or not to condition embeddings by time from diagnosis date')
    parser.add_argument('--dxseq_event', action='store_true', default=False, help='Wether or not to condition time seq before and after by time from diagnosis date')
    parser.add_argument('--days', type=int, default=30, help="lookfwd window days")


    parser.add_argument('--time_embed_dim', type=int, default=128, help="Representation size at for time embeddings.")
    parser.add_argument('--pred_mask', action='store_true', default=False, help='Pred masked out tokens.')
    parser.add_argument('--mask_prob', type=float, default=0, help="Dropout value for the neural network model.")
    parser.add_argument('--pred_mask_lambda', type=float, default=0, help="Weight to use for pred_mask loss.")
    # Learning Hyper-params
    parser.add_argument('--loss_f', type=str, default="binary_cross_entropy_with_logits", help='loss function to use, available: [Xent (default), MSE]')
    parser.add_argument('--optimizer', type=str, default="adam", help='optimizer to use [default: adam]')
    parser.add_argument('--batch_splits', type=int, default=1, help='Splits batch size into smaller batches in order to fit gpu memmory limits. Optimizer step is run only after one batch size is over. Note: batch_size/batch_splits should be int [default: 1]')
    parser.add_argument('--train_batch_size', type=int, default=100, help="Batch size to train neural network model.")
    parser.add_argument('--eval_batch_size', type=int, default=16, help="Batch size to train neural network model.")
    parser.add_argument('--max_batches_per_train_epoch', type=int, default=1000, help='max batches to per train epoch. [default: 10000]')
    parser.add_argument('--max_batches_per_dev_epoch', type=int, default=1000, help='max batches to per dev epoch. [default: 10000]')
    parser.add_argument('--exhaust_dataloader',  action='store_true', default=False,  help='whether to truncate epoch to max batches per dataset epoch or to exhaust the full dataloader')
    parser.add_argument('--init_lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('--lr_decay', type=float, default=1., help='Decay of learning rate [default: no decay (1.)]')
    parser.add_argument('--momentum', type=float, default=0, help='Momentum to use with SGD')
    parser.add_argument('--weight_decay', type=float, default=0, help='L2 Regularization penaty [default: 0]')
    parser.add_argument('--patience', type=int, default=3, help='number of epochs without improvement on dev before halving learning rate and reloading best model [default: 5]')
    parser.add_argument('--tuning_metric', type=str, default='36month_auroc_c', help='Metric to judge dev set results. Possible options include auc, loss, accuracy [default: loss]')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs for train [default: 256]')
    parser.add_argument('--linear_interpolate_risk', action='store_true', default=False, help='linearily interpolate risk from init year to actual year at cancer.') #
    parser.add_argument('--class_bal', action='store_true', default=True, help='Whether to apply a weighted sampler to balance between the classes on each batch.')
    # TODO: class_bal should be removed or not 'store_true

    # evaluation
    parser.add_argument('--eval_auroc', action='store_true', default=False, help='Whether to calculate AUROC')
    parser.add_argument('--eval_auprc', action='store_true', default=False, help='Whether to calculate AUPRC')
    parser.add_argument('--eval_mcc', action='store_true', default=False, help='Whether to calculate MCC')
    parser.add_argument('--eval_c_index', action='store_true', default=False, help='Whether to calculate c-Index')
    parser.add_argument('--annual_eval_type', default='both', help='Whether to calculate AUC for accumulative years or individual years, or both.')
    
    # Where to store stuff
    parser.add_argument('--log_dir', type=str,
                    default="../../logs",
                    help="path to store logs and detailed job level result files")
    parser.add_argument('--log_name', type=str,
                    default='',
                    help="identifier of a model")
    parser.add_argument('--time_logger_verbose', type=int, default=2, help='Verbose of logging (1: each main, 2: each epoch, 3: each step). Default: 2.')
    parser.add_argument('--time_logger_step', type=int, default=1, help='Log the time elapse every how many iterations - 0 for no logging.')
    parser.add_argument('--model_dir', type=str, default='../../snapshot', help='where to dump the model')
    parser.add_argument('--snapshot', type=str, default=None, help='filename of model snapshot to load[default: None]')
    parser.add_argument('--results_path', type=str,
                        default='../../results',
                        help="Path of where to store results dict")
    parser.add_argument('-f', '--file', type=str, default='filepath', help='flag for jupyter')

    # Other arguments
    parser.add_argument('--map_to_icd_system', default=None, help='whether to map current code to index to the other code systems Choose from [None, "dk", "usa"]')
    parser.add_argument('--resume_experiment', type=str, default=None, help='path to results to reload [default: None]')
    parser.add_argument('--overwrite_resumed_experiments', action='store_true', default=False, help='overwrite resumed experiments [default: False]')
    parser.add_argument('--continue_training', action='store_true', default=False, help="Continue training")
    
    
    if args_str is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args_str.split())

    args.cuda = args.cuda and torch.cuda.is_available()
    if not args.cuda:
        args.device = 'cpu'
    args.num_years = max(args.day_endpoints) / 365

    if any([True for argument, values in args.__dict__.items() for metric in argument.split('_')[-1:] if metric in args.tuning_metric and values]):
        print ("Tuning metric {} is not computed in Eval metric! Switch it to true in your config file!".format(args.tuning_metric))

    # learning initial state
    args.optimizer_state = None
    args.current_epoch = None
    args.lr = None
    args.epoch_stats = None
    args.step_indx = 1

    # if args.resume_experiment:
    #     resumed_args = pickle.load(open(args.resume_experiment, "rb"))
    #     # print(resumed_args.keys())
    #     keep_args_from_config = ['train', 'dev', 'test', 'attribute', 'save_dir', 'metadata_path', 'dataset',
    #                              'results_path', 'num_workers', 'eval_batch_size',
    #                              'max_batches_per_dev_epoch', 'resume_experiment']
    #     if 'exclusion_interval' in sys.argv: #TODO find a cleaner solution
    #         keep_args_from_config.append('exclusion_interval')
    #     if args.overwrite_resumed_experiments:
    #         keep_args_from_config.remove('results_path')
    #     for k,v in resumed_args.items():
    #         if k not in keep_args_from_config:
    #             args.__dict__[k] = v
    #     print(resumed_args['save_dir'])
    #     print(args.save_dir)
    #     resumed_args['save_dir'] = args.save_dir  # for flexibility of putting trained models in different folder
    #     print(resumed_args['save_dir'])
    #     args.snapshot = state.get_model_path(resumed_args, using_old_dk_naming=args.map_to_icd_system == 'usa')
    #     args.code_to_index_map = resumed_args['code_to_index_map']
    #     # args.device = args.device if torch.cuda.is_available() else 'cpu'
    #     # args.device = "cuda:1"

    # print("Args".format(args))

    return args


def md5(key):
    '''
    returns a hashed with md5 string of the key
    '''
    return hashlib.md5(key.encode()).hexdigest()

def parse_dispatcher_config(config):
    '''
    Parses an experiment config, and creates jobs. For flags that are expected to be a single item,
    but the config contains a list, this will return one job for each item in the list.
    :config - experiment_config
    returns: jobs - a list of flag strings, each of which encapsulates one job.
        *Example: --train --cuda --dropout=0.1 ...
    returns: experiment_axies - axies that the grid hyperparameter is hyperparametering over
    '''
    jobs = [""]
    experiment_axies = []
    hyperparameter_space = config['search_space']

    hyperparameter_space_flags = hyperparameter_space.keys()
    hyperparameter_space_flags = sorted(hyperparameter_space_flags)
    for ind, flag in enumerate(hyperparameter_space_flags):
        possible_values = hyperparameter_space[flag]
        if len(possible_values) > 1:
            experiment_axies.append(flag)

        children = []
        if len(possible_values) == 0 or type(possible_values) is not list:
            raise Exception(POSS_VAL_NOT_LIST.format(flag, possible_values))
        for value in possible_values:
            for parent_job in jobs:
                if type(value) is bool:
                    if value:
                        new_job_str = "{} --{}".format(parent_job, flag)
                    else:
                        new_job_str = parent_job
                elif type(value) is list:
                    val_list_str = " ".join([str(v) for v in value])
                    new_job_str = "{} --{} {}".format(parent_job, flag,
                                                      val_list_str)
                else:
                    new_job_str = "{} --{} {}".format(parent_job, flag, value)
                children.append(new_job_str)
        jobs = children

    return jobs, experiment_axies

class dict2args(object):
    def __init__(self, d):
        self.__dict__ = d
