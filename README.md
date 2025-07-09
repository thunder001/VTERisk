## A Deep Learning Model to Dynamically Predict Cancer Associated Thromboembolism using Electronic Health Records from the Veteran’s Health Administration     

## Pre-requirements: 
Environment setup 

Data file in a json format

## Usage
### Data structure
The model need a json file having the following structure to run.  
Note: data file needs to pre-split into train/train.json, dev/dev.json and dev/test.json.


## Introduction
The repository contains the code implementation used for the paper "A Deep Learning Model to Dynamically Predict Cancer Associated Thromboembolism using Electronic Health Records from the Veteran’s Health Administration". 
We used disease trajectories from EHR to calculate the risk of VTE at different intervals after the assessment. The repository supports different deep learning models.

## Usage
### Data structure

 In case you want to reproduce the results on another dataset you need to generate a json file having the same structure. Phecode events and lab events (e.g. BUN0 (NA), BUN1 (below normal range), BUN2 (within normal range), BUN3 (above normal range)) are combined into single trajectories for every patient. Timestamped outcome "VTE" needs to be encoded within the events (like the third example).

Note the example file is only for the train split set, dev and test sets also need to be constructed.

```json
{
    "PID_0":{
        "birthdate":"1900-01-01",
        "end_of_data":"2022-01-01",
        "gender":1,
        "indexdate": "2014-01-30",
        "split_group": "train",
        "BMI":3,
        "Race":7,
        "dxdate":"2013-01-04",
        "events":[
            {
                "admdate":"1990-01-01",
                "admid":"00000000",
                "codes":"740.1",
            },
            {
                "admdate":"2015-01-01",
                "admid":"00000001",
                "codes":"Glucose3",
            }
        ]
    },
    "PID_1111":{
        "birthdate":"1950-01-01",
        "end_of_data":"2021-01-01",
        "gender":2,
        "indexdate": "2013-01-30",
        "split_group": "train",
        "BMI":2,
        "Race":4,
        "dxdate":"2009-01-04",
        "events":[
            {
                "admdate":"1980-01-01",
                "admid":"00000002",
                "codes":"250.2",
            },
            {
                "admdate":"2010-01-03",
                "admid":"00000004",
                "codes":"ALT2",
            }
        ]
    },
    "PID_9999":{
            "birtdate":"1970-01-01",
            "end_of_data":"2022-01-01",
            "gender":1,
            "indexdate": "2020-01-30",
            "split_group": "train",
            "BMI":1,
            "Race":4,
            "dxdate":"2019-01-04",
            "events":[
                {
                    "admdate":"2010-01-03",
                    "admid":"00020002",
                    "codes":"196",
                },
                {
                    "admdate":"2010-01-03",
                    "admid":"00020004",
                    "codes":"BUN0",
                },
                {
                    "admdate":"2021-01-03",
                    "admid":"00020014",
                    "codes":"VTE",
                }
            ]
        }

}
```
### Run scripts
For code to run, go to repo root directory (VTERISK), enter the following in command line:   
```python src/scripts/schedulers/expr_dispatcher.py –experiment-config-path configs/Main.json```

The config file is in json format, please replace ```{}``` with your absoulte path for your data file.
```json
{
    "search_space": {
      "train_data_dir": ["{}/train"],
      "dev_data_dir": ["{}/dev"],
      "test_data_dir": ["{}/test"],
      "model_name": ["transformer"],
      "start_at_dx_100":[true],
	  "start_noise_days":[true],        
	  "start_noise_len":[20],
      "max_days_before_index":[1000],
	  "filter_max_len":[true],
	  "max_events_length": [800],
      "min_events_length":[3],
      "num_layers":[1],
      "num_heads":[8],
      "exclusion_interval": [0],
      "baseline_diseases": [false],
      "dropout":[0],
      "init_lr":[1e-03],
      "pool_name": ["Softmax_AttentionPool"],
      "time_seq_cos": [true],
      "use_time_embed": [true],
      "use_age_embed": [true],
      "use_dxtime_embed": [true],
	  "use_index_embed": [true],
	  "add_sex_neuron": [true],
	  "add_bmi_neuron": [true],
	  "add_race_neuron": [true],
	  "ageseq_event": [true],
	  "dxseq_cos": [true],
      "indseq_cos": [true],
      "pad_size": [100],
      "epochs":[10],
      "train":[true],
      "dev":[true],
      "test":[true],
      "cuda":[false],
      "device":["cpu"],
      "num_workers":[0],
      "optimizer": ["adam"],
      "train_batch_size": [256],
      "eval_batch_size": [128],
      "max_batches_per_train_epoch": [1000],
      "max_batches_per_dev_epoch": [500],
      "eval_auroc": [true],
      "eval_auprc": [true],
      "eval_c_index": [true],
      "tuning_metric":["3day_auroc_c"],
      "model_dir":["snapshot_vte"],
      "log_dir":["logs_transformer_vte"]
    },
    "available_gpus": [1]
  }
```

If run successfully, you will see a log director under the project folder, which saves model training/evaluation process and results, and snapshot folder, which save the trained model.









