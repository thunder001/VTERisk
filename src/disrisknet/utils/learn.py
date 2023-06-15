import pdb
import pickle
import json
import warnings
import torch
import numpy as np
from torch.utils import data
import sklearn.metrics
from collections import defaultdict
import disrisknet.learn.state_keeper as state
import disrisknet.models.factory as model_factory


def get_train_variables(args, model):
    '''
        Given args, and whether or not resuming training, return
        relevant train variales.

        returns:
        - start_epoch:  Index of initial epoch
        - epoch_stats: Dict summarizing epoch by epoch results
        - state_keeper: Object responsibile for saving and restoring training state
        - batch_size: sampling batch_size
        - models: Dict of models
        - optimizers: Dict of optimizers, one for each model
        - tuning_key: Name of epoch_stats key to control learning rate by
        - num_epoch_sans_improvement: Number of epochs since last dev improvment, as measured by tuning_key
    '''
    state_keeper = state.StateKeeper(args)
    tuning_key = "dev_{}".format(args.tuning_metric)
    start_epoch = 1
    args.lr = args.init_lr
    epoch_stats = init_metrics_dictionary(modes=['train', 'dev'])
    # Set up models and optimizers
    if isinstance(model, dict):
        models = model
    else:
        models = {'model': model}
    optimizers = {}
    for name in models:
        model = models[name].to(args.device)
        optimizers[name] = model_factory.get_optimizer(model, args)
    if args.continue_training:
        models, optimizer_states, _, _, epoch_stats_pre = state_keeper.load()
        # _, _, _, _, epoch_stats_pre = state_keeper.load()
        epoch_stats['pre_best'] = epoch_stats_pre[tuning_key][-1] if len(epoch_stats_pre[tuning_key]) > 0 \
            else epoch_stats_pre['pre_best']
        print('Best pre-metrics is: {}'.format(epoch_stats['pre_best']))
        # Set optimizers
        for name in optimizers:
            optimizer = optimizers[name]
            state_dict = optimizer_states[name]
            optimizers[name] = state_keeper.load_optimizer(optimizer, state_dict)
        # Reset LR
        for name in optimizers:
            optimizer = optimizers[name]
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.init_lr
    num_epoch_sans_improvement = 0

    return start_epoch, epoch_stats, state_keeper, models, optimizers, tuning_key, num_epoch_sans_improvement


def concat_collate(batch):

    concat_batch = []
    for sample in batch:
        concat_batch.extend(sample)
    return data.dataloader.default_collate(concat_batch)


def init_metrics_dictionary(modes):
    '''
    Return empty metrics dict
    '''
    stats_dict = defaultdict(list)
    stats_dict['best_epoch'] = 0
    return stats_dict


def get_dataset_loader(args, data, rank=None):
    persistent_workers = True if args.num_workers > 0 else False
    if args.class_bal and data.split_group in ['train', 'dev', 'attribute']:
        if args.world_size > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                data,
                num_replicas=args.world_size,
                rank=rank
            )
            data_loader = torch.utils.data.DataLoader(
                data,
                num_workers=0,
                sampler=sampler,
                pin_memory=True,
                batch_size=args.train_batch_size if data.split_group == 'train' else args.eval_batch_size,
                collate_fn=concat_collate
            )
        else:
            # sampler = torch.utils.data.sampler.WeightedRandomSampler(
            #         weights=data.weights,
            #         num_samples=len(data),
            #         replacement=data.split_group=='train')
            sampler = torch.utils.data.sampler.WeightedRandomSampler(
                    weights=data.weights,
                    num_samples=len(data),
                    replacement=True)

            data_loader = torch.utils.data.DataLoader(
                    data,
                    num_workers=args.num_workers,
                    persistent_workers=persistent_workers,
                    sampler=sampler,
                    pin_memory=True,
                    batch_size=args.train_batch_size if data.split_group == 'train' else args.eval_batch_size,
                    collate_fn=concat_collate
                    )
    else:
        
        data_loader = torch.utils.data.DataLoader(
            data,
            batch_size=args.train_batch_size if data.split_group == 'train' else args.eval_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            persistent_workers=persistent_workers,
            pin_memory=True,
            collate_fn=concat_collate,
            drop_last=False)

    return data_loader
