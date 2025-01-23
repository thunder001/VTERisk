import numpy as np


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