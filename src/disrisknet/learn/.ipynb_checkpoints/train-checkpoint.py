import os
import math
import numpy as np
import sklearn.metrics
import torch
from tqdm import tqdm
import disrisknet.models.factory as model_factory
import disrisknet.learn.state_keeper as state
from disrisknet.learn.step import model_step
import disrisknet.utils.stats as stats
from disrisknet.utils.eval import compute_eval_metrics
from disrisknet.utils.learn import init_metrics_dictionary, \
    get_dataset_loader, get_train_variables
from disrisknet.utils.time_logger import time_logger
import warnings
import torch.distributed as dist
import pdb

tqdm.monitor_interval = 0


def train_model(train_data_loader, dev_data_loader, model, args):
    '''
        Train model and tune on dev set. If model doesn't improve dev performance within args.patience
        epochs, then halve the learning rate, restore the model to best and continue training.
        At the end of training, the function will restore the model to best dev version.
        returns epoch_stats: a dictionary of epoch level metrics for train and test
        returns models : dict of models, containing best performing model setting from this call to train
    '''

    logger_train = time_logger(1, hierachy=2, model_name=args.model_name, logger_dir = args.log_dir, log_name=args.log_name) if args.time_logger_verbose >= 2 else time_logger(0)
    logger_epoch = time_logger(1, hierachy=1, model_name=args.model_name, logger_dir = args.log_dir, log_name=args.log_name) if args.time_logger_verbose >= 2 else time_logger(0)

    start_epoch, epoch_stats, state_keeper, models, optimizers, tuning_key, num_epoch_sans_improvement \
        = get_train_variables(args, model)
    # train_data_loader = get_dataset_loader(args, train_data)
    # dev_data_loader = get_dataset_loader(args, dev_data)
    # logger_epoch.log("Get train and dev dataset loaders")

    for epoch in range(start_epoch, args.epochs + 1):
        print("-------------\nEpoch {}:\n".format(epoch))
        logger_train.log("Epoch {}".format(epoch))
        for mode, data_loader in [('Train', train_data_loader), ('Dev', dev_data_loader)]:
            train_model = mode == 'Train'
            key_prefix = mode.lower()
            loss, golds, patient_golds, preds, probs, exams, pids, censor_times, days_to_final_censors, dates \
                = run_epoch(
                data_loader,
                train_model=train_model,
                truncate_epoch=True,
                models=models,
                optimizers=optimizers,
                args=args)
            logger_epoch.log("Run epoch ({})".format(key_prefix))

            log_statement, epoch_stats, _ = compute_eval_metrics(
                args, loss, golds, patient_golds, preds, probs, exams, pids, dates,
                censor_times, days_to_final_censors, epoch_stats, key_prefix)
            logger_epoch.log("Compute eval metrics ({})".format(key_prefix))
            print(log_statement)

        # Save model if beats best dev (min loss or max c-index_{i,a})
        best_func, arg_best = (min, np.argmin) if 'loss' in tuning_key else (max, np.argmax)
        better_than_pre = False
        if args.continue_training:
            if 'loss' in tuning_key:
                if best_func(epoch_stats[tuning_key]) < epoch_stats['pre_best']:
                    better_than_pre = True
            else:
                if best_func(epoch_stats[tuning_key]) > epoch_stats['pre_best']:
                    better_than_pre = True
            improved = best_func(epoch_stats[tuning_key]) == epoch_stats[tuning_key][-1] and better_than_pre
        else:
            improved = best_func(epoch_stats[tuning_key]) == epoch_stats[tuning_key][-1]

        if improved:
            num_epoch_sans_improvement = 0
            if not os.path.isdir(args.log_dir):
                os.makedirs(args.log_dir)
            epoch_stats['best_epoch'] = arg_best(epoch_stats[tuning_key])
            state_keeper.save(models, optimizers, epoch, args.lr, epoch_stats)
            logger_epoch.log("Save improved model")
        else:
            num_epoch_sans_improvement += 1

        print('---- Best Dev {} is {} at epoch {}'.format(
            args.tuning_metric,
            epoch_stats[tuning_key][epoch_stats['best_epoch']],
            epoch_stats['best_epoch'] + 1))
        if args.continue_training:
            print('---- Best Dev {} in previous training is {}'.format(args.tuning_metric, epoch_stats['pre_best']))

        if num_epoch_sans_improvement >= args.patience:
            print("Reducing learning rate")
            num_epoch_sans_improvement = 0

            models, optimizer_states, _, _, _ = state_keeper.load()
            # Reset optimizers
            for name in optimizers:
                optimizer = optimizers[name]
                state_dict = optimizer_states[name]
                optimizers[name] = state_keeper.load_optimizer(optimizer, state_dict)
            # Reduce LR
            for name in optimizers:
                optimizer = optimizers[name]
                # print(optimizer.state_dict())
                for param_group in optimizer.state_dict()['param_groups']:
                    param_group['lr'] *= args.lr_decay
                print(f"Learning rate of optimier: {optimizer.state_dict()['param_groups'][0]['lr']}")
            # Update lr also in args for resumable usage
            args.lr *= args.lr_decay
            logger_epoch.log("Prepare for next epoch")
            logger_epoch.update()

    # Restore model to best dev performance, or last epoch when not tuning on dev
    models, _, _, _, _ = state_keeper.load()

    return epoch_stats, models


def run_epoch(data_loader, train_model, truncate_epoch, models, optimizers, args):
    '''
        Run model for one pass of data_loader, and return epoch statistics.
        args:
        - data_loader: Pytorch dataloader over some dataset.
        - train_model: True to train the model and run the optimizers
        - models: dict of models, where 'model' is the main model, and others can be critics, or meta-models
        - optimizer: dict of optimizers, one for each model
        - args: general runtime args defined in by argparse
        returns:
        - avg_loss: epoch loss
        - golds: labels for all samples in data_loader
        - preds: model predictions for all samples in data_loader
        - probs: model softmaxes for all samples in data_loader
        - exams: exam ids for samples if available, used to cluster samples for evaluation.
    '''
    data_iter = data_loader.__iter__()
    preds = []
    probs = []
    censor_times = []
    days_to_final_censors = []
    dates = []
    golds = []
    patient_golds = []
    losses = []
    exams = []
    pids = []
    logger = time_logger(args.time_logger_step) if args.time_logger_verbose>=3 else time_logger(0)

    torch.set_grad_enabled(train_model)
    for name in models:
        if train_model:
            models[name].train()
            if optimizers is not None:
                optimizers[name].zero_grad()
        else:
            models[name].eval()

    batch_loss = 0
    num_batches_per_epoch = len(data_loader)

    if truncate_epoch:
        max_batches = args.max_batches_per_train_epoch if train_model else args.max_batches_per_dev_epoch
        num_batches_per_epoch = min(num_batches_per_epoch, (max_batches * args.batch_splits))
        logger.log("Truncate epoch @ batches: {}".format(num_batches_per_epoch))
    # print(num_batches_per_epoch)
    i = 0
    miniters = int(num_batches_per_epoch / 10)
    tqdm_bar = tqdm(data_iter, total=num_batches_per_epoch, miniters=miniters)
    for batch in data_iter:
        if batch is None:
            warnings.warn('Empty batch')
            continue
        if tqdm_bar.n > num_batches_per_epoch:
            # TODO: only tentative, fix tqdm later
            break

        batch = prepare_batch(batch, args)
        logger.newline()
        logger.log("prepare data")

        step_results = model_step(batch, models, optimizers, train_model, args)

        loss, batch_preds, batch_probs, batch_golds, batch_patient_golds, batch_exams, batch_pids, batch_censors, batch_days_to_censor,batch_dates = step_results
        batch_loss += loss.cpu().data.item()
        logger.log("model step")

        if train_model:
            if (i + 1) % args.batch_splits == 0:
                optimizers['model'].step()
                optimizers['model'].zero_grad()

        logger.log("model update")
        if (i + 1) % args.batch_splits == 0:
            losses.append(batch_loss)
            batch_loss = 0

        preds.extend(batch_preds)
        probs.extend(batch_probs)
        golds.extend(batch_golds)
        patient_golds.extend(batch_patient_golds)
        dates.extend(batch_dates)
        censor_times.extend(batch_censors)
        days_to_final_censors.extend(batch_days_to_censor)
        exams.extend(batch_exams)
        pids.extend(batch_pids)
        logger.log("saving results")

        i += 1
        if i > num_batches_per_epoch + 1 and args.num_workers > 0:
            data_iter.__del__()
            break
        logger.update()
        tqdm_bar.update()

    avg_loss = np.mean(losses)

    return avg_loss, golds, patient_golds, preds, probs, exams, pids, censor_times, days_to_final_censors, dates


def prepare_batch(batch, args):
    keys_of_interest = ['x', 'y','y_seq','y_mask', 'time_seq', 'age', 'ks', 'sex', 'bmi', 'age_seq', 'dx', 'dx_seq', 'time_at_event', 'outcome', 'days_to_censor']
    if args.use_char_embedding:
        keys_of_interest += ['char_x']

    for key in batch.keys():
        if key in keys_of_interest:
            batch[key] = batch[key].to(args.device)
    return batch


def eval_model(eval_data, name, models, args):
    '''
        Run model on test data, and return test stats (includes loss
        accuracy, etc)
    '''
    logger_eval = time_logger(1, hierachy = 2, model_name=args.model_name, logger_dir=args.log_dir, log_name=args.log_name) if args.time_logger_verbose>=2 else time_logger(0)
    logger_eval.log("Evaluating model")

    if not isinstance(models, dict):
        models = {'model': models}
    models['model'] = models['model'].to(args.device)
    batch_size = args.eval_batch_size // args.batch_splits
    eval_stats = init_metrics_dictionary(modes=[name])
    logger_eval.log("Load model")

    data_loader = get_dataset_loader(args, eval_data)
    logger_eval.log('Load eval data')

    loss, golds, patient_golds, preds, probs, exams, pids,censor_times, days_to_final_censors, dates = run_epoch(
        data_loader,
        train_model=False,
        truncate_epoch=(not args.exhaust_dataloader and eval_data.split_group!='test'),
        models=models,
        optimizers=None,
        args=args)
    logger_eval.log('Run eval epoch')

    log_statement, eval_stats, eval_preds = compute_eval_metrics(
                            args, loss,
                            golds, patient_golds, preds, probs, exams, pids, dates,
                            censor_times, days_to_final_censors, eval_stats, name)
    print(log_statement)
    logger_eval.log('Compute eval metrics')
    logger_eval.update()

    return eval_stats, eval_preds
