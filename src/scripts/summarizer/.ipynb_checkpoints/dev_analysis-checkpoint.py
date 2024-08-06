import matplotlib
import pickle as pkl
import pprint as pp
from collections import OrderedDict
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os


def print_stats(stats_fname):
    with open(stats_fname, 'rb') as f:
        stats = pkl.load(f)
    print("\nAUROCS:")
    aurocs = OrderedDict({k: stats[k] for k in stats if k[-7:] == 'auroc_c'})
    pp.pprint(aurocs)
    print("\nAUPRCS:")
    auprcs = OrderedDict({k: stats[k] for k in stats if k[-7:] == 'auprc_c'})
    pp.pprint(auprcs)

def get_params(result_path):
    with open(result_path, 'rb') as f:
        results = pkl.load(f)
    params = ['model_name', 'num_layers', 'num_heads', 'exclusion_interval', 'init_lr', "weight_decay", 'pool_name', 
    'no_random_sample_eval_trajectories', 'pad_size', 'epochs', 'train', 'dev', 'test', 'cuda', 
    'train_batch_size', 'eval_batch_size', 'max_batches_per_train_epoch', 'max_batches_per_dev_epoch', 
    'max_events_length', 'max_eval_indices', 'eval_auroc', 'eval_auprc', 'eval_c_index']
    configs = [{param: results[param]} for param in params]
    return configs

def plot_train_metric(epoch_stats_path, metric_keys):
    with open(epoch_stats_path, 'rb') as f:
        epoch_stats = pkl.load(f)
    plt.figure(figsize=(10, 7))
    lines = []
    labels = []
    for metric_key in metric_keys:
        # print(cls)
        # print(fprs[cls])
        num_of_epoch = len(epoch_stats[metric_key])
        line, = plt.plot(list(range(1, num_of_epoch + 1)), epoch_stats[metric_key], lw=1)
        lines.append(line)
        labels.append(metric_key)
    plt.xticks(np.arange(0, 21, 1))
    plt.xlim([0, 21])
    plt.xlabel('Epoch', size=20)
    plt.ylabel('Score', size=20)
    plt.legend(lines, labels, loc='best', prop=dict(size=12))
    plt.title("Training process for transformer model", size=16)

def plot_train_metrics(result_path, model, metrics):
    epoch_path_pat = result_path + '.epoch_stats'
    out_path_pat = 'results\\training\\epochs\\{}_{}.png'
    epoch_path = epoch_path_pat.format(model)
    for metric in metrics:
        outpath = out_path_pat.format(model, metric)
        if metric == "auroc":
            plot_train_metric(epoch_path, ['train_36month_auroc_c', 'dev_36month_auroc_c'])  
        if metric == "auprc":
            plot_train_metric(epoch_path, ['train_36month_auprc_c', 'dev_36month_auprc_c'])
        if metric == "loss":
            plot_train_metric(epoch_path, ['train_loss', 'dev_loss'])
        plt.savefig(outpath)

def get_training_summary(model, log_dir, test = False):
    result_fname = model + '.results'
    result_path = os.path.join(log_dir, result_fname)
    dev_stat_path = result_path + '.dev_stats'
    params = get_params(result_path)
    pp.pprint(params)
    print_stats(dev_stat_path)
    if test:
        test_stat_path = result_path + '.test_stats'
        print_stats(test_stat_path)
    plot_train_metrics(result_path, model, metrics)

def get_test_summary(logid, log_dir):
    result_fname = logid + '.results'
    result_path = os.path.join(log_dir, result_fname)
    test_stat_path = result_path + '.test_stats'
    params = get_params(result_path)
    pp.pprint(params)
    print_stats(test_stat_path)


#########################################################
# ---------- Model training results ---------------------
#########################################################
metrics = ['auroc', 'auprc', 'loss']
result_path_pat_3m = 'logs_training_transformer_va3m/{}.results'
result_path_pat_1m = 'logs_training_transformer_va1m_part0/{}.results'

# -----------------------------------------------------------------------
# --- Non-random sampling combinded with learning rate and weight decay -
# -----------------------------------------------------------------------

# The first model trained from full VA dataset
result_path = 'logs/all_epoch_10_w1_batch_12816_devlen10.results'
params = get_params(result_path)
test_path = 'logs/all_epoch_10_w1_batch_12816_devlen10.results.test_stats'
print_stats(test_path)
epoch_path = 'logs/all_epoch_10_w1_batch_12816_devlen10.results.epoch_stats'
with open(epoch_path, 'rb') as f:
    epochs = pkl.load(f)

# ----- Learning rate and weight decay ------
# 0.0005, 0.001
result_path = 'logs_training_transformer_va1m_part0/c3846d5054d9fcd81ff3acba2e9e5c91.results'
params = get_params(result_path)
test_path = 'logs_training_transformer_va1m_part0/c3846d5054d9fcd81ff3acba2e9e5c91.results.dev_stats'
print_stats(test_path)

# 0.0001, 0.001
result_path = 'logs_training_transformer_va1m_part0/1d5b7da64830c4b49138d29e5f966454.results'
params = get_params(result_path)
test_path = 'logs_training_transformer_va1m_part0/1d5b7da64830c4b49138d29e5f966454.results.dev_stats'
print_stats(test_path)

epoch_path = 'logs_training_transformer_va1m_part0/1d5b7da64830c4b49138d29e5f966454.results.epoch_stats'
plot_train_metric(epoch_path, ['train_36month_auroc_c', 'dev_36month_auroc_c', 'train_loss', 'dev_loss' ])
plt.savefig('results/training/epochs/1d5b7da64830c4b49138d29e5f966454.png')

# 0.0005, 0.005
result_path = 'logs_training_transformer_va1m_part0/1f82a48e932440088453c0db28b0935c.results'
params = get_params(result_path)
test_path = 'logs_training_transformer_va1m_part0/1f82a48e932440088453c0db28b0935c.results.dev_stats'
print_stats(test_path)

# 0.0001, 0.005
model = '549579a85b8d27d8a0061e51e4664bcd'
result_path = result_path_pat_1m.format(model)
dev_stat_path = result_path + '.dev_stats'
params = get_params(result_path)
pp.pprint(params)
print_stats(dev_stat_path)
plot_train_metrics(model, metrics)


epoch_path = 'logs_training_transformer_va1m_part0/549579a85b8d27d8a0061e51e4664bcd.results.epoch_stats'
plot_train_metric(epoch_path, ['train_36month_auroc_c', 'dev_36month_auroc_c', 'train_loss', 'dev_loss' ])
plt.savefig('results/training/epochs/549579a85b8d27d8a0061e51e4664bcd.png')
# -------End of config -----

# 0.0001, 0.01 - Not learn
model = '63bf6e06e27a7093e4a898ffc0d778d2'
get_training_summary(model)

# 0.0002, 0.01
model = '7710d9f12c96c56450cfe028f622d960'
get_training_summary(model)

# ------ grid_search_va04.json --more traning and eval samples
# 0.001, 0.005, train batch: 4000, eval_batch: 100 
model = '33e17528c2ecece94854aee870d03b58'
get_training_summary(model)

# 0.0001, 0.005, train batch: 4000, eval_batch: 100 
model = '13a2a0dfb9fd46ffe20cbde87349d232'
get_training_summary(model)

# -----------------------------------------------------------------------
# ---- Random sampling combinded with learning rate and weight decay ----
# -----------------------------------------------------------------------
# -- begin grid_search_va05 ---
# 0.001, 0.001, train batch: 4000, eval_batch: 2000, eval_indices: 10
model = '397af42985332d19ef6000f8d800a218'
get_training_summary(model)

# 0.0001, 0.001, train batch: 4000, eval_batch: 2000, eval_indices: 10
model = 'eca0b4c65c4eac5f46d70eca22010fb6'
get_training_summary(model)

# 0.001, 0.005, train batch: 4000, eval_batch: 2000, eval_indices: 10
model = '2a931b8dc890a6ab01d7ea8cd6e9f7d2'
get_training_summary(model)

# 0.0001, 0.005, train batch: 4000, eval_batch: 2000, eval_indices: 10
model = '58ba98d86abc056b5e7fa2e4ee613e86'
get_training_summary(model)
# -- end 

# -- begin grid_search_va12 ---
# train: 10/40/1000; dev: random, 10/8/100; 0.0001/0.001;
model = 'ce47006b061767ae9c62db626102a69b'
get_training_summary(model)
# train: 10/40/1000; dev: random, 10/8/100; 0.0001/0.005;
model = 'afe0980dc13983d422e2cc78383bb232'
get_training_summary(model)
# -- end

# -- begin grid_search_va13 ---
# train: 1/40/4000; dev: non-random, 250/8/100; 0.001/0.005; auprc
model = '88b4cb6b7df405a0451d606eea131198'
get_training_summary(model)
logid = '1ea35a74edc346217a8b2ca038651d9f'
log_dir = 'logs_test_transformer_va1m_part0'
get_test_summary(logid, log_dir)
# train: 1/40/4000; dev: non-random, 250/8/100; 0.001/0.002; auprc
model = 'afe0980dc13983d422e2cc78383bb232'
get_training_summary(model)
# train: 1/40/4000; dev: non-random, 250/8/100; 0.001/0.001; auprc
# -- end

# -- begin grid_search_va14 ---
# train: 1/40/4000; dev: non-random, 250/8/100; 0.0001/0.001; auprc
model = '21c9a6920a8e8a8030d0f984ca5f9d21'
get_training_summary(model)
# train: 1/40/4000; dev: non-random, 250/8/100; 0.0001/0.002; auprc
model = 'afe0980dc13983d422e2cc78383bb232'
get_training_summary(model)
# train: 1/40/4000; dev: non-random, 250/8/100; 0.0001/0.005; auprc
# -- end


# -----------------------------------------------------------------------
# ----------------------- 3 Million model results -----------------------
# -----------------------------------------------------------------------
log_dir_train = 'logs_training_transformer_va3m'
log_dir_test = 'logs_test_transformer_va3m'
metrics = ['auroc', 'auprc', 'loss']
# Transformer model - Data exclusion: 0
model = '564ca19fc9cb08fa1ba512cbbf5d5982'
get_training_summary(model, log_dir_train)
# randlom-100
logid = '646397ef78b3aa2c49f96a80b2db1cfa'
get_test_summary(logid, log_dir_test)
# Non-random-100
logid = '1b63a742f50db2ccb4b93072082af16a'
get_test_summary(logid, log_dir_test)
# Random-200
logid = '1f2921038990df8911522a413b0cfcba'
get_test_summary(logid, log_dir_test)

# Transformer model - Data exclusion: 3
model = '2ba7a93c54a871ab82a18fb6a0e969a4'
get_training_summary(model, log_dir_train)
logid = 'dfcf89629bf66aff23b0647bd3a7f0ac'
get_test_summary(logid, log_dir_test)

# Transformer model - Data exclusion: 6
model = '1abbbe7900e10b1544fb64afe9ce1efe'
get_training_summary(model, log_dir_train)
logid = 'acfc03c55f5e3567c8ce95a98127954e'
get_test_summary(logid, log_dir_test)


# -----------------------------------------------------------------------
# ---------------- 3 Million model results (subgroup) -------------------
# -----------------------------------------------------------------------
log_dir_train = 'logs_training_transformer_va3m'
log_dir_test = 'logs_test_transformer_va3m'
metrics = ['auroc', 'auprc', 'loss']
# Transformer model - Data exclusion: 0
# model = '564ca19fc9cb08fa1ba512cbbf5d5982'
# Female-randlom-100
logid = '8edf0caec4793a23ce3d71f13af4dad8'
get_test_summary(logid, log_dir_test)
# Male-randlom-100
logid = '54e95d52a486108aefcb486bf3e7705f'
get_test_summary(logid, log_dir_test)
# Age > 50 - random-100
logid = '2dc9f02924cb5310808ec36bc43d99ac'
get_test_summary(logid, log_dir_test)

# Transformer model - Data exclusion: 3
# model = '2ba7a93c54a871ab82a18fb6a0e969a4'


# ------------------------------------------------------------------------


# Model trained from 3M VA dataset
result_path = 'logs_training_transformer_va3m/eca0b4c65c4eac5f46d70eca22010fb6.results'
params = get_params(result_path)
test_path = 'logs_training_transformer_va3m/eca0b4c65c4eac5f46d70eca22010fb6.results.dev_stats'
print_stats(test_path)

result_path = 'logs_training_transformer_va3m/397af42985332d19ef6000f8d800a218.results'
params = get_params(result_path)
test_path = 'logs_training_transformer_va3m/397af42985332d19ef6000f8d800a218.results.dev_stats'
print_stats(test_path)

model = '58ba98d86abc056b5e7fa2e4ee613e86'
metrics = ['auroc', 'auprc', 'loss']
result_path_pat_3m = 'logs_training_transformer_va3m/{}.results'
result_path = result_path_pat_3m.format(model)
dev_stat_path = result_path + '.dev_stats'
params = get_params(result_path)
pp.pprint(params)
print_stats(dev_stat_path)
plot_train_metrics(model, metrics)

logid = '99aae8baa52c1e434c62100a2a478084'
log_dir = 'logs_test_transformer_va3m'
get_test_summary(logid, log_dir)


result_path = 'logs_training_transformer_va3m/2a931b8dc890a6ab01d7ea8cd6e9f7d2.results'
params = get_params(result_path)
test_path = 'logs_training_transformer_va3m/2a931b8dc890a6ab01d7ea8cd6e9f7d2.results.dev_stats'
print_stats(test_path)














dev_stats_fname = "logs_2/cpu_va_10000_test.results.dev_stats"
print_stats(dev_stats_fname)

test_stats_fname = "logs_2/cpu_va_10000_test.results.dev_stats"
print_stats(test_stats_fname)


test_results_fname = 'logs_2/cpu_va_10000_test.results'
with open(test_results_fname, 'rb') as f:
    results = pkl.load(f)

test_results_fname = 'snapshot_danish/ac3f725f4cdefda0a4a4f0db6a3bf838.results'
with open(test_results_fname, 'rb') as f:
    results = pkl.load(f)

pp.pprint(results)

test_stats_fname = "logs_cross_test/fe353b08454ff122f6d84efac77814e3.results.test_stats"
print_stats(test_stats_fname)

test_stats_fname2 = "logs_cross_test_10000/942a0d0bae5096b6a911268961709c11.results.test_stats"
print_stats(test_stats_fname2)

results_fname = 'logs_cross_test_10000/942a0d0bae5096b6a911268961709c11.results'
with open(results_fname, 'rb') as f:
    results = pkl.load(f)

# Model 686
test_stats_fname = 'logs_cross_test_10000/ce966cafd6f664b8edd1b9f1a885864d.results.test_stats'
print_stats(test_stats_fname)

# Model 734
test_stats_fname = 'logs_cross_test_10000/aea983d9545c4e5294d9f26c1062fe31.results.test_stats'
print_stats(test_stats_fname)

# Model 847
test_stats_fname = 'logs_cross_test_10000/4c740b9ea216cb0748464325a8047449.results.test_stats'
print_stats(test_stats_fname)

# 100K patients
# Model 686
test_stats_fname = 'logs_cross_test_100k/f99b51a40277824f9e3957f9885b8b47.results.test_stats'
print_stats(test_stats_fname)

test_stats_fname = "G:\FillmoreCancerData\chunlei\pancpred\code\pancnet\logs\part00_epoch_20.results.test_stats"
print_stats(test_stats_fname)


results_fname = 'G:\FillmoreCancerData\chunlei\pancpred\code\pancnet\snapshot\model_part00_epoch_20_optim.pt'
with open(results_fname, 'rb') as f:
    results = pkl.load(f)

epoch_stats_fname = r'G:\FillmoreCancerData\chunlei\pancpred\code\pancnet\snapshot_va10000\va_10000_w4_batch_4016_devinx10_stats.p'
with open(epoch_stats_fname, 'rb') as f:
    epoch_stats = pkl.load(f)

epoch_stats_fname = r'G:\FillmoreCancerData\chunlei\pancpred\code\pancnet\snapshot\part0_epoch_10_w4_batch_4004_devlen10_at_stats.p'
with open(epoch_stats_fname, 'rb') as f:
    epoch_stats = pkl.load(f)

param_dict_fname = r'G:\FillmoreCancerData\chunlei\pancpred\code\pancnet\snapshot\part0_epoch_10_w4_batch_4004_devlen10_at_param.p'
with open(param_dict_fname, 'rb') as f:
    param_dict = pkl.load(f)




test_stats_fname = 'code/pancnet/logs/part0_epoch_10_w4_batch_4004_devlen10_at.results.part_25.test_stats'
with open(test_stats_fname, 'rb') as f:
    stats = pkl.load(f)
print_stats(test_stats_fname)



test_stats_fname = 'code/pancnet/logs/va_10000_w4_batch_4004_devinx10.results.test_attribution'
with open(test_stats_fname, 'rb') as f:
    stats = pkl.load(f)

#########################################################
#---------------- snapshot files ------------------------
#########################################################
model_1_stat_fname = 'snapshot/all_epoch_10_w1_batch_12816_devlen10_stats.p'
with open(model_1_stat_fname, 'rb') as f:
    stats = pkl.load(f)
model_1_para_fname = 'snapshot/all_epoch_10_w1_batch_12816_devlen10_param.p'
with open(model_1_para_fname, 'rb') as f:
    paras = pkl.load(f)
model_1_model_fname = 'snapshot/model_all_epoch_10_w1_batch_12816_devlen10_model.pt'
with open(model_1_model_fname, 'rb') as f:
    model1 = pkl.load(f)




