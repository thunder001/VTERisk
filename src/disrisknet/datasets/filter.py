from disrisknet.utils.date import parse_date
MIN_FOLLOWUP_YEAR_IF_NEG = 0

def get_avai_trajectory_indices(patient, events, feat_subgroup, split_group, args):

    valid_split, criterion = is_valid_subgroup(patient, feat_subgroup, split_group, args)
    if not valid_split:
        return [], None

    if '-' in criterion: # range of patient age
        min_age, max_age = map(int, criterion.split('-'))
        def is_valid_age(time):
            age = (time-parse_date(feat_subgroup['birth_date'])).days//365
            return (age in range(min_age, max_age) and split_group == 'test') or \
                   (age not in range(min_age, max_age) and split_group != 'test')
    elif criterion!='NA': # range of database age
        year_line = int(criterion)
        def is_valid_age(time):
            year = time.year
            return (year > year_line and split_group == 'test') or \
                   (year <= year_line and split_group != 'test')
    else:
        is_valid_age = lambda time: True

    candidate_indices = []
    valid_indices = []
    y = False

    for idx in range(len(events)):
        if patient['outcome'] and \
                (patient['outcome_date'] - events[idx]['admit_date']).days <= 30 * args.exclusion_interval:
            continue

        if is_valid_trajectory(events[:idx+1], patient['outcome_date'],
                                patient['outcome'], patient['index_date'], args):
            if is_valid_age(events[idx]['admit_date']):
                candidate_indices.append(idx)
                days_to_censor = (patient['outcome_date']-events[idx]['admit_date']).days
                y = (days_to_censor < ( max(args.month_endpoints) * 30) and patient['outcome']) or y
        if len(candidate_indices) > 0:
            valid_indices.append(candidate_indices[-1])
    if not valid_indices:
        return [], None
    else:
         return valid_indices, y


def is_valid_trajectory(events_to_date, outcome_date, outcome, index_date, args):

    if events_to_date[-1]['admit_date'] > index_date:
        return False
    
    enough_events_counted =  len(events_to_date) >= args.min_events_length
    if not enough_events_counted:
        return False
    
    is_pos_in_time_horizon = (outcome_date - events_to_date[-1]['admit_date']).days < max(args.month_endpoints) * 30
    is_pos_pre_cancer = events_to_date[-1]['admit_date'] < outcome_date

    is_valid_pos = outcome and is_pos_pre_cancer and  is_pos_in_time_horizon
    is_valid_neg = not outcome and \
                   (outcome_date - events_to_date[-1]['admit_date']).days // 365 > MIN_FOLLOWUP_YEAR_IF_NEG
    return (is_valid_neg or is_valid_pos)


def is_valid_subgroup(patient, feat_subgroup, split_group, args):

    include = False
    criterion = 'NA'
    try:
        key, value = args.subgroup_validation.split('_')
    except: #random partition
        key = 'random'
        value = -1

    if key == 'random':
        if patient['split_group'] == split_group:
            include = True
    elif key == 'age' or key == 'before':
        if patient['split_group'] == split_group:
            include = True
        criterion = value
    else: # other sub-group
        if feat_subgroup[key] == value and split_group == 'test':
            include = True
        elif feat_subgroup[key] != value and split_group != 'test':
            include = True

    return include, str(criterion)
