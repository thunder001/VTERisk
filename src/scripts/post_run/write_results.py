import json
from collections import Counter
import pickle
import numpy as np
import pandas as pd
import os
import argparse
from os.path import dirname, realpath
import sys

import math
import hashlib
import datetime
import datetime as datetime
 
import torch
from torch import nn, optim
from torch.nn import functional as F

from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
sys.path.append(dirname(dirname(realpath(os.getcwd()))))

from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

sys.path.append("./src/")

 
def md5(key):
    return hashlib.md5(repr(key).encode()).hexdigest()


def write_results_to_csv(keypath, trainpath, devpath, testpath,
                         TRAIN_OUT, DEV_OUT, TEST_OUT):


    testDF = pd.read_csv(keypath)
    pat_ids = (testDF['patient_id'] ).astype(int)
    testDF['pids'] = pat_ids.apply(md5)

    tdf = pd.DataFrame({'patient_id': (testDF['patient_id'] ),    
                        'pids': (testDF['pids'] ),    
                        'dob': (testDF['dob'] ),  
                        'outcome_date': (testDF['outcome_date'] ), 
                        'obs_time_end': (testDF['obs_time_end'] ),  
                        'index_date': (testDF['index_date'] ),  
                        'diag_date': (testDF['diag_date'] ),  
                        'outcome': (testDF['outcome'])    })
    with open(trainpath, 'rb') as f:
        R = pickle.load(f)
        p = np.array(R['probs'])

    Df = pd.DataFrame.from_dict(R)
    Df['probs'] = Df['probs'].astype(float)
    Df['exams'] = Df['exams'].astype(int)
    M = pd.merge(tdf, Df)
    M.to_csv(TRAIN_OUT) 
    
    with open(devpath, 'rb') as f:
        R_d = pickle.load(f)
    Df_dev = pd.DataFrame.from_dict(R_d)
    Df_dev['probs'] = Df_dev['probs'].astype(float)
    M_dev = pd.merge(tdf, Df_dev)
    M_dev.to_csv(DEV_OUT) 
    
    with open(testpath, 'rb') as f:
        R_t = pickle.load(f)

    Df_test = pd.DataFrame.from_dict(R_t)
    Df_test['probs'] = Df_test['probs'].astype(float)
    M_test = pd.merge(tdf, Df_test)
    M_test.to_csv(TEST_OUT) 
