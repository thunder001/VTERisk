# A transformer model to predict VTE outcome of cancer patients based on their disease trajectories

## Pre-requirements: 
Environment setup 
Data file in a json format

## Usage
### Data structure
The model need a json file having the following structure to run.
Note: one extra step, data file needs to pre-split into train/train.json, dev/dev.json and dev/test.json; 

```json
{
    "PID_0":{
        "birtdate":"1900-01-01",
        "end_of_data":"2022-01-01",
        "events":[
            {
                "admdate":"2001-01-01",
                "admid":"00000000",
                "codes":"E10",
            },
            {
                "admdate":"2005-01-01",
                "admid":"00000001",
                "codes":"C25",
            }
        ],
        "indexdate": "2003-01-01",
        "split_group": "train"
    }
}
```
### Run scripts
For code to run, go to src directory, enter the following in command line:
```python scripts/schedulers/expr_dispatcher.py –experiment-config-path configs/grid_search_test.json```

The config file is in json format, please replace ```{}``` with your absoulte path for your data file.
{
    "search_space": {
      "train_data_dir": ["{}/train"],
      "dev_data_dir": ["{}/dev"],
      "test_data_dir": ["{}/test"],
      "model_name": ["transformer"],
      "num_layers":[1],
      "num_heads":[16],
      "exclusion_interval": [0],
      "baseline_diseases": [false],
      "dropout":[0],
      "init_lr":[1e-03],
      "pool_name": ["Softmax_AttentionPool"],
      "use_time_embed": [true],
      "use_age_embed": [true],
      "no_random_sample_eval_trajectories": [true],
      "pad_size": [100],
      "epochs":[10],
      "train":[true],
      "dev":[true],
      "test":[true],
      "cuda":[false],
      "device":["cpu"],
      "num_workers":[0],
      "optimizer": ["adam"],
      "train_batch_size": [128],
      "eval_batch_size": [128],
      "max_batches_per_train_epoch": [1000],
      "max_batches_per_dev_epoch": [500],
      "max_events_length": [50],
      "max_eval_indices": [10],
      "eval_auroc": [true],
      "eval_auprc": [true],
      "eval_c_index": [true],
      "tuning_metric":["6month_auroc_c"],
      "model_dir":["../snapshot_vte"],
      "log_dir":["../logs_transformer_vte"]
    },
    "available_gpus": [1]
  }

If run successfully, you will see a log director under the project folder, which saves model training/evaluation process and results, and snapshot folder, which save the trained model.









