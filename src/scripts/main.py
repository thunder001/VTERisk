import json
import os
import time
from os.path import dirname, realpath
import sys
sys.path.append(dirname(dirname(realpath(__file__))))
import pickle
import disrisknet.datasets.factory as dataset_factory
import disrisknet.models.factory as model_factory
from disrisknet.models.utils import AttributionModel
import disrisknet.learn.train as train
import disrisknet.learn.attribute as attribute
from disrisknet.utils.parsing import parse_args
from disrisknet.utils.time_logger import time_logger
import torch
import multiprocessing as mp
from disrisknet.utils.learn import get_dataset_loader
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import Counter
from copy import deepcopy
import pandas as pd

if __name__ == '__main__':
    DATE_FORMAT_STR = "%Y-%m-%d:%H-%M-%S"
    args = parse_args()
    print(args)
    args.world_size = 1
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    log_stem = os.path.join(args.log_dir, args.log_name)
    args.save_path = '{}.results'.format(log_stem)
    
    print('Final args:'.format(args))
    logger_main = time_logger(1, hierachy=5, model_name=args.model_name, logger_dir=args.log_dir, log_name=args.log_name) \
        if args.time_logger_verbose >= 1 else time_logger(0, model_name=args.model_name, logger_dir=args.log_dir)
    start_time = time.time()
    logger_main.log("The main.py started at {}".format(time.asctime(time.localtime(start_time))))
    print("CUDA:", torch.cuda.is_available())

    print("Loading Dataset...\n")
    train_data, dev_data, test_data, attribution_set, cross_eval_data, args = dataset_factory.get_dataset(args)
    print("Number of patient for -Train:{},-Dev:{}, -Test:{}, --Attr:{}, --Cross: {}\n".format(
        train_data.__len__(), dev_data.__len__(), test_data.__len__(),
        attribution_set.__len__(), cross_eval_data.__len__()))
    logger_main.log("Load datasets")

    if args.snapshot is None:
        print("Building model...\n")
        model = model_factory.get_model(args)
    else:
        print("Loading model...\n")
        model = model_factory.load_model(args.snapshot, args)

    print(model)
    print("Working threads: ", torch.get_num_threads())
    if torch.get_num_threads() < args.num_workers:
        torch.set_num_threads(args.num_workers)
        print("Adding threads count to {}.".format(torch.get_num_threads()))
    logger_main.log("Build model")

    print()
    if args.train:
        print("Build dataloader")
        logger_epoch = time_logger(1, hierachy=2, model_name=args.model_name, logger_dir=args.log_dir, log_name=args.log_name) if args.time_logger_verbose >= 2 else time_logger(0)
        train_data_loader = get_dataset_loader(args, train_data)
        dev_data_loader = get_dataset_loader(args, dev_data)
        logger_epoch.log("Get train and dev dataset loaders")
        epoch_stats, model = train.train_model(train_data_loader, dev_data_loader, model, args)
        if args.num_workers > 0:
            train_data_loader._iterator._shutdown_workers()
            dev_data_loader._iterator._shutdown_workers()
        print("Save train/dev results to {}".format(args.save_path))
        logger_main.log("TRAINING")
        args_dict = vars(args).copy()
        del args_dict['code_to_index_map']
        pickle.dump(args_dict, open(args.save_path, 'wb'))
        pickle.dump(epoch_stats, open('{}.{}'.format(args.save_path, "epoch_stats"), 'wb'))
        del epoch_stats
        logger_main.log("Dump results")

    print()
    if args.train_log:
        print("-------------\nTrain")
        #args.class_bal = False
        train_stats, train_preds = train.eval_model(train_data, 'train', model, args)
        print("Save dev results to {}".format(args.save_path))
        logger_main.log("VALIDATION")
        args_dict = vars(args).copy(); del args_dict['code_to_index_map']; pickle.dump(args_dict, open(args.save_path, 'wb'))
        pickle.dump(train_stats, open("{}.{}".format(args.save_path, "train_stats"), 'wb'))
        pickle.dump(train_preds, open("{}.{}".format(args.save_path, "train_preds"), 'wb'))
        del train_stats, train_preds
        logger_main.log("Dump results")

        
    print()
    if args.dev:
        print("-------------\nDev")
        #args.class_bal = False
        dev_stats, dev_preds = train.eval_model(dev_data, 'dev', model, args)
        print("Save dev results to {}".format(args.save_path))
        logger_main.log("VALIDATION")
        args_dict = vars(args).copy(); del args_dict['code_to_index_map']; pickle.dump(args_dict, open(args.save_path, 'wb'))
        pickle.dump(dev_stats, open("{}.{}".format(args.save_path, "dev_stats"), 'wb'))
        pickle.dump(dev_preds, open("{}.{}".format(args.save_path, "dev_preds"), 'wb'))
        del dev_stats, dev_preds
        logger_main.log("Dump results")

    print()
    if args.test:
        print("-------------\nTest")
        test_stats, test_preds = train.eval_model(test_data, 'test', model, args)
        print("Save test results to {}".format(args.save_path))
        args_dict = vars(args).copy();
        del args_dict['code_to_index_map']; pickle.dump(args_dict, open(args.save_path, 'wb'))
        logger_main.log("TESTING")
        print("{}.{}".format(args.save_path, "test_stats"))
        pickle.dump(test_stats, open("{}.{}".format(args.save_path, "test_stats"), 'wb'))
        pickle.dump(test_preds, open("{}.{}".format(args.save_path, "test_preds"), 'wb'))
        del test_stats, test_preds
        logger_main.log("Dump results")

    print()
    if args.attribute:

        print("-------------\nAttribution")
        model_for_attribution = AttributionModel(model)
        print(model_for_attribution.model)
        for month_idx in [0,1 ]:
            
            if args.neg_attribute:
                test_attribution, test_censored_attribution = attribute.compute_attribution(
                    attribution_set, model_for_attribution, args, month_idx, neg=True)                
                cens0_attr_csvname = "I{}_{}0_Neg_attribution.csv".format(args.Interval, args.days_bin)
                # default is 90 days bin, only need 1

                C_I = [v for v in test_censored_attribution.items() ]
                
                C_i0 = C_I[0][1]
                L0 = []
                for a in  C_i0.keys():
                    L0.append( ( [x.item() for x in  C_i0[a] ] ))
                D0 = pd.DataFrame(L0, C_i0.keys())
                D0.to_csv(cens0_attr_csvname)
                logger_main.log("Dump results")
                #print("Save test results to {}".format(args.save_path))
                logger_main.log("ATTRIBUTION")

            elif args.both_attribute:
                test_attribution, test_censored_attribution = attribute.compute_attribution(
                    attribution_set, model_for_attribution, args, month_idx, both=True)                
                cens0_attr_csvname = "I{}_{}0_Both_attribution.csv".format(args.Interval, args.days_bin)
                # default is 90 days bin, only need 1
                C_I = [v for v in test_censored_attribution.items() ]
                C_i0 = C_I[0][1]
                L0 = []
                for a in  C_i0.keys():
                    L0.append( ( [x.item() for x in  C_i0[a] ] ))
                D0 = pd.DataFrame(L0, C_i0.keys())
                D0.to_csv(cens0_attr_csvname)
                logger_main.log("Dump results")
                #print("Save test results to {}".format(args.save_path))
                logger_main.log("ATTRIBUTION")
            
            else:
                test_attribution, test_censored_attribution = attribute.compute_attribution(
                    attribution_set, model_for_attribution, args, month_idx)
                cens0_attr_csvname = "I{}_{}0_attribution.csv".format(args.Interval, args.days_bin)
                C_I = [v for v in test_censored_attribution.items() ]

                C_i0 = C_I[0][1]
                L0 = []
                for a in  C_i0.keys():
                    L0.append( ( [x.item() for x in  C_i0[a] ] ))
                D0 = pd.DataFrame(L0, C_i0.keys())
                D0.to_csv(cens0_attr_csvname)
                logger_main.log("Dump results")
                logger_main.log("ATTRIBUTION")
   
             
            
          

     
            
    if args.cross_eval:
        print("-------------\nTest")
        cross_eval_stats, cross_eval_preds = train.eval_model(cross_eval_data, 'test', model, args)
        print("Save cross evaluation results to {}".format(args.save_path))
        args_dict = vars(args).copy();
        del args_dict['code_to_index_map']; pickle.dump(args_dict, open(args.save_path, 'wb'))
        logger_main.log("CROSS EVALUATION")
        print("{}.{}".format(args.save_path, "cross_eval_stats"))
        pickle.dump(cross_eval_stats, open("{}.{}".format(args.save_path, "cross_eval_stats"), 'wb'))
        pickle.dump(cross_eval_preds, open("{}.{}".format(args.save_path, "cross_eval_preds"), 'wb'))
        del cross_eval_stats, cross_eval_preds
        logger_main.log("Dump results")
    end_time = time.time()
    logger_main.log("The main.py ended at {}".format(time.asctime(time.localtime(end_time))))
    logger_main.log('The total of runtime were {} hours'.format((end_time - start_time) / 3600.0))
    logger_main.newline()