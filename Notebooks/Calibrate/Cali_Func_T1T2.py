import os
import sys
import argparse
from os.path import dirname, realpath

sys.path.append("./src/")
sys.path.append(dirname(dirname(realpath(os.getcwd()))))
 

import json
from collections import Counter
import pickle
import numpy as np
import pandas as pd
import os
import sys
import argparse
from os.path import dirname, realpath
import math
import hashlib
import datetime
import datetime as datetime
sys.path.append("./src/")
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
src_path = "G:\\FillmoreCancerData\\markhe\\VTERisk" 
src_path2 = "G:\\FillmoreCancerData\\markhe\\VTERisk - Copy" 

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_regression
from sklearn.isotonic import IsotonicRegression
import ml_insights as mli


import numpy as np
from scipy import stats as stats

# first load 12_15a for T1, T2
M1 = pd.read_csv('Cali_25_1212/Cali_Mats/M1_Cali_Beta.csv') 
M2 = pd.read_csv('Cali_25_1212/Cali_Mats/M2_Cali_Beta.csv') 
M1 = M1.loc[:, ~M1.columns.str.contains('^Unnamed')]
M2 = M2.loc[:, ~M2.columns.str.contains('^Unnamed')]

X1 = M1.to_numpy().transpose()
X2 = M2.to_numpy().transpose()

def trim_upper(data):
    # proportiontocut=.2
    sorted_data = np.sort(data)
    n = len(sorted_data)
    k = int(n * .2)
    return np.quantile(sorted_data[0:n-k], .75)

def trim_lower(data):
    sorted_data = np.sort(data)
    n = len(sorted_data)
    k = int(n * .2)
    return np.quantile(sorted_data[k:n], .25)
 
def trim_upper_b(data):
    sorted_data = np.sort(data)
    n = len(sorted_data)
    k = int(n * .2)
    return np.quantile(sorted_data[0:n-k], .2)

def trim_lower_b(data):
    sorted_data = np.sort(data)
    n = len(sorted_data)
    k = int(n * .2)
    return np.quantile(sorted_data[k:n], .8)


def trim_upper_c(data):
    # proportiontocut=.2
    sorted_data = np.sort(data)
    n = len(sorted_data)
    k = int(n * .3)
    return np.quantile(sorted_data[0:n-k], .9)

def trim_lower_c(data):
    sorted_data = np.sort(data)
    n = len(sorted_data)
    k = int(n * .3)
    return np.quantile(sorted_data[k:n], .9)


def trim_upper_d(data):
    # proportiontocut=.2
    sorted_data = np.sort(data)
    n = len(sorted_data)
    k = int(n * .6)
    return np.quantile(sorted_data[0:n-k], .9)

def trim_lower_d(data):
    sorted_data = np.sort(data)
    n = len(sorted_data)
    k = int(n * .6)
    return np.quantile(sorted_data[k:n], .9)