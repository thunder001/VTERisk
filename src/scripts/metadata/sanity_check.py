import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy

def data_sanity_check(data_path):
    '''
    Check inconsistency between index dates and end of data dates.
    Index date is defined as the date that a cancer patient recieved the first treatment. 
    End of data date is defined as either last follow up date or death date, whichever first.
    The logic is that end of data dates should be alwary bigger than index dates. 
    '''
    data = json.load(open(data_path, 'r'))
    # p0 = data['1003444654']
    # distribution of event length
    event_lengths = []
    for patient in data.keys():
        pat_dat = data[patient]
        event_length = len(pat_dat['events'])
        event_lengths.append(event_length)

    # fig, ax = plt.subplots() 
    # ax.hist(event_lengths)
    # plt.show()

    # obtain all idex dates
    index_dates = []
    for patient in data.keys():
        pat_dat = data[patient]
        index_date = pat_dat['indexdate']
        index_dates.append(index_date)

    # obtain all end of data dates (Last followups or death dates)
    end_of_data_dates = []
    for patient in data.keys():
        pat_dat = data[patient]
        end_of_data_date = pat_dat['end_of_data']
        end_of_data_dates.append(end_of_data_date)

    new_data = pd.DataFrame({'indexdates':index_dates, 'enddates':end_of_data_dates})
    prob_data = new_data[new_data['indexdates'] > new_data['enddates']]
    print(f'{prob_data.shape[0]} of {new_data.shape[0]} patients are invalid since their index dates are bigger than end of data dates!')
    return prob_data


# ---------------------------------------------------------------
# ---------------------------------------------------------------
# ---------------------------------------------------------------

# cohort_data_path = r"F:\tmp_pancreatic\temp_fst\global\raw\analytic_final_2000-2021.feather"
# cohort = pd.read_feather(cohort_data_path)

def main():
    pass


# test_data_fpath = "F:\\tmp_pancreatic\\temp_json\\test\\vte\\train-10000\\train.json"
train_data_fpath = "F:\\tmp_pancreatic\\temp_json\\test\\vte\\train\\train.json"
dev_data_fpath = "F:\\tmp_pancreatic\\temp_json\\test\\vte\\dev\\dev.json"
test_data_fpath = "F:\\tmp_pancreatic\\temp_json\\test\\vte\\test\\test.json"
# death status

data_problem_train = data_sanity_check(train_data_fpath)
data_problem_dev = data_sanity_check(dev_data_fpath)
data_problem_test = data_sanity_check(test_data_fpath)

data_path = r'F:\tmp_pancreatic\temp_json\global\vte\data-081623.json'
data = data_sanity_check(data_path)

data_path = r'F:\tmp_pancreatic\temp_json\global\vte\data-081823.json'
data = data_sanity_check(data_path)