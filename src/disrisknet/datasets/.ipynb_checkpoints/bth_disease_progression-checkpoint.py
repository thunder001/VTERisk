import datetime
import tqdm
import numpy as np
import random
import json
from collections import Counter
from disrisknet.datasets.factory import RegisterDataset, UNK_TOKEN, PAD_TOKEN, NO_OP_TOKEN
from disrisknet.datasets.filter import get_avai_trajectory_indices
from torch.utils import data
from disrisknet.utils.date import parse_date
from disrisknet.utils.parsing import md5
import pdb
from bisect import bisect_left
from itertools import compress

END_OF_TIME_DATE = datetime.datetime(2020, 12, 31, 0, 0)
OUTCOME_CODE = ['VTE']
BASELINE = []

UNK_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'
MAX_TIME_EMBED_PERIOD_IN_DAYS = 12 * 365
MIN_TIME_EMBED_PERIOD_IN_DAYS = 10

SUMMARY_MSG = "Contructed BTH Disease Progression DiseaseRisk {} dataset with {} records, {} patients, and the following class balance \n {}"


@RegisterDataset("bth_disease_progression")
class BTH_Disease_Progression_Dataset(data.Dataset):
    """
        Dataset for survival analysis based on bth disease progression alone.
    """

    def __init__(self, metadata, args, split_group):
        super(BTH_Disease_Progression_Dataset, self).__init__()
        '''
            Dataset flattens admissions from the same patients.
        '''
        self.args = args
        self.split_group = split_group
        self.PAD_TOKEN = PAD_TOKEN
        self.metadata = metadata
        self.patients = np.array([])
        shard_path = None
        count_missing_date = 0
        self.shard = True if type(metadata) is list else False
        miniters = int(len(metadata) / 10) # 10% of the total iters
        metadata_tqdm = tqdm.tqdm(metadata, miniters=miniters)

        for patient in metadata_tqdm:
            if self.shard:
                (patient_not_encoded, current_shard_path), = patient.items()

                patient_id = md5(patient_not_encoded)
                if shard_path != current_shard_path:
                    shard_path = current_shard_path
                    patient_metadata = json.load(open(shard_path, 'r'))
                patient_dict = {'patient_id': patient_id, 'shard_path': shard_path}
            else:
                patient_id = md5(patient)
                patient_metadata = {patient_id: metadata[patient]}
                patient_dict = {'patient_id': patient_id}

            # For cross evaluation, splitting data is not needed and all data will be used for evaluation
            if self.split_group != 'all' and patient_metadata[patient_id]['split_group'] != split_group:
                continue
            if self.split_group == 'all':
                patient_metadata[patient_id]['split_group'] = 'test'
            if 'end_of_data' not in patient_metadata[patient_id]:
                count_missing_date += 1
                continue

            obs_time_end = parse_date(patient_metadata[patient_id]['end_of_data'])
            dob = parse_date(patient_metadata[patient_id]['birthdate'])            
            index_date = parse_date(patient_metadata[patient_id]['indexdate'])
            dx_date = parse_date(patient_metadata[patient_id]['dxdate'])
            ks = patient_metadata[patient_id]['ks_mod_score']
            sex = patient_metadata[patient_id]['gender']
            bmi = patient_metadata[patient_id]['BMI']
            race = patient_metadata[patient_id]['Race']

            events = self.process_events(patient_metadata[patient_id]['events'], obs_time_end)
            outcome, outcome_date = self.get_outcome_date(events, index_date, 
                                                                     end_of_date=obs_time_end)
            patient_dict.update({'outcome': outcome,
                                 'dob': dob,
                                 'outcome_date': outcome_date,
                                 'split_group': patient_metadata[patient_id]['split_group'],
                                 'obs_time_end': obs_time_end,
                                 'index_date': index_date,
                                 'dx_date': dx_date,
                                 'ks': ks,
                                 'race':race,
                                 'sex': sex,
                                 'bmi': bmi})
            
            feat_subgroup = {'birth_date': patient_metadata[patient_id]['birthdate']}

            if args.cross_eval or self.split_group == 'all':
                split_group = 'test'
            avai_indices, gold = get_avai_trajectory_indices(patient_dict, events, feat_subgroup, split_group, args)

            patient_dict.update({'avai_indices': avai_indices, 'y': gold})

            if not self.shard:
                patient_dict.update({'events': events})

            if avai_indices:  
                self.patients = np.append(self.patients, patient_dict)

        print("Number of patients with missing end of data is: {}".format(count_missing_date))
        self.class_count()
        total_positive = sum([p['y'] for p in self.patients])
        print("Number of positive patients  in {} is: {}".format(self.split_group, total_positive))
        self.class_count()

    def process_events(self, events, end_date=None):

        for event in events:
            if event is not None:
                # test = event['admdate']
                event['admit_date'] = parse_date(event['admdate'])
            else:
                events.remove(event)
        if end_date is not None and self.args.use_no_op_token:
            start_date, start_admid = min([(e['admit_date'], e['admid']) for e in events])
            year_range = (end_date - start_date).days // 365
            for year in range(year_range):
                events.append(self.get_no_op_event(start_date, year, start_admid))

        events = sorted(events, key=lambda event: event['admit_date'])

        if self.args.baseline:
            for e in events:
                if e['codes'] not in BASELINE:
                    e['codes'] = PAD_TOKEN
        return events

    def get_no_op_event(self, start_date, year, admid_base):
        code = NO_OP_TOKEN
        new_date = start_date + datetime.timedelta(365 * year)
        event = {'admit_date': new_date, 'codes': code, 'hospital': 'n/a', 'admid': '{}_{}'.format(admid_base, year)}

        return event
    

    def get_trajectory(self, patient):

        if self.shard:
            patient['events'] = self.process_events(patient['events'], patient["obs_time_end"])
        
        events = patient['events']
        ev_dates = [events[i]['admit_date'] for i in range(len(events))]
                 
        def find_closest_date_before(reference_date, dates):
            index = bisect_left(dates, reference_date)
            if index == 0:
                return None  # No date before the reference date
            return index - 1
        
    
        if self.args.sensitivity:
            lookback = patient['index_date'] + datetime.timedelta( int(   self.args.days )  )
            selected_idx = [find_closest_date_before(lookback, ev_dates)]
        elif self.args.multi_traj:
            
            t1 = patient['index_date'] + datetime.timedelta( 30 ) 
            t2 = patient['index_date'] + datetime.timedelta( 90 )  
            t3 = patient['index_date'] + datetime.timedelta( 180 )  
            t4 = patient['index_date'] + datetime.timedelta( 270 )  
    
            # need to subset traj by outcome date
            # if 
            
            sel_i =  [find_closest_date_before(t1, ev_dates), 
                      find_closest_date_before(t2, ev_dates),
                      find_closest_date_before(t3, ev_dates),
                      find_closest_date_before(t4, ev_dates),]    
            valid =  [i< patient['outcome_date'] for i in [t1,t2,t3,t4]]
            selected_idx = list( compress(sel_i, valid))
            
        else:
            forward_noise = random.choice(np.arange( self.args.days0  ,   self.args.days   ))
            lookback = patient['index_date'] + datetime.timedelta( int(forward_noise)  )
                
                
        samples = []
        for idx in selected_idx:
            events_to_date = patient['events'][:idx + 1]

            codes = [e['codes'] for e in events_to_date]
            _, time_seq = self.get_time_seq(events_to_date, events_to_date[-1]['admit_date'])
            age, age_seq = self.get_event_seq( events_to_date[-1]['admit_date'] , patient['dob'])
 
            if self.args.dxseq_event: 
                dx, dx_seq = self.get_event_seq(patient['dx_date'],patient['dob'])            
            else:
                dx, dx_seq = self.get_event_seq( events_to_date[-1]['admit_date']  ,patient['dx_date'])                    
            if self.args.indseq_event: 
                ind, ind_seq= self.get_event_seq(patient['index_date'],patient['dx_date'])            
            else:
                 ind, ind_seq = self.get_event_seq( events_to_date[-1]['admit_date'] , patient['index_date'])        

            y, y_seq, y_mask, time_at_event, days_to_censor = self.get_label(patient, idx)
            samples.append({'events': events_to_date,
                            'y': y,
                            'y_seq': y_seq,
                            'y_mask': y_mask,
                            'time_at_event': time_at_event,
                            'outcome': patient['outcome'],
                            'patient_id': patient['patient_id'],
                            'days_to_censor': days_to_censor,
                            'time_seq': time_seq,
                            'dx_seq': dx_seq,
                            'dx': dx,
                            'ind_seq': ind_seq,
                            'ind': ind,
                            'age_seq': age_seq,
                            'age': age,
                            'ks': patient['ks'],
                            'sex': patient['sex'],
                            'bmi': patient['bmi'],
                            'race': patient['race'],
                            'admit_date': events_to_date[-1]['admit_date'].isoformat(),
                            'exam': str(events_to_date[-1]['admid'])})

        return self.add_noise(samples)

    def get_time_seq(self, events, reference_date):
        deltas = np.array([abs((reference_date - event['admit_date']).days) for event in events])
        multipliers = 2 * np.pi / (np.linspace(start=MIN_TIME_EMBED_PERIOD_IN_DAYS, stop=MAX_TIME_EMBED_PERIOD_IN_DAYS,
                                               num=self.args.time_embed_dim))

        deltas, multipliers = deltas.reshape(len(deltas), 1), multipliers.reshape(1, len(multipliers))
        positional_embeddings = np.cos(deltas * multipliers)
        return max(deltas), positional_embeddings
     
    def get_event_seq(self, event, reference_date):
        deltas = np.array([abs((reference_date - event).days)])
        multipliers = 2 * np.pi / (np.linspace(start=MIN_TIME_EMBED_PERIOD_IN_DAYS, stop=MAX_TIME_EMBED_PERIOD_IN_DAYS,
                                               num=self.args.time_embed_dim))

        deltas, multipliers = deltas.reshape(len(deltas), 1), multipliers.reshape(1, len(multipliers))
        positional_embeddings = np.cos(deltas * multipliers)        
        return max(deltas), positional_embeddings
    
    def class_count(self):
        # Implement for class balance
        ys = [patient['y'] for patient in self.patients]
        label_counts = Counter(ys)
        weight_per_label = 1. / len(label_counts)
        label_weights = {
            label: weight_per_label / count for label, count in label_counts.items()
        }
        if self.args.class_bal:
            print("Label weights are {}".format(label_weights))
        self.weights = [label_weights[d] for d in ys]

    def get_label(self, patient, idx):
        '''
            outcome_date: outcome occurance or END_OF_TIME_DATE
            time_at_event: years to outcome_data/END_OF_TIME_DATE
            y_seq: zero array unless ever_develops_outcome => y_seq[time_at_event:]=1
                    * Used as golds in cumulative_probability_layer
            y_mask: how many years left in the disease window,
                    ([1] for 0:time_at_event years and [0] for the rest)
                    (without linear interpolation, y_mask looks like complement of y_seq)
                    * Used for mask loss in cumulative_probability_layer
            Ex1:  a patient has the outcome in 3 years
                    time_at_event: 2,
                    y_seq: [0, 0, 1, 1, 1]
                    y_mask: [1, 1, 1, 0, 0]
            Ex2:  a patient never gets outcome
                    y_seq: [0, 0, 0, 0, 0]
                    y_mask: [1, 1, 0, 0, 0]
                    time_at_event: 1,
        '''
        event = patient['events'][idx]
        days_to_censor = (patient['outcome_date'] - event['admit_date']).days
        if self.args.pred_day:
            num_time_steps, max_time = len(self.args.day_endpoints), max(self.args.day_endpoints)
            y = days_to_censor < max_time and patient['outcome']
            y_seq = np.zeros(num_time_steps)
            time_at_event = min([i for i, mo in enumerate(self.args.day_endpoints)
                                if days_to_censor < (mo * 30)]) if days_to_censor < (max_time * 30) else num_time_steps - 1
            if y:
                y_seq[time_at_event:] = 1
            y_mask = np.array([1] * (time_at_event + 1) + [0] * (num_time_steps - (time_at_event + 1)))
        else:
            num_time_steps, max_time = len(self.args.month_endpoints), max(self.args.month_endpoints)
            y = days_to_censor < (max_time * 30) and patient['outcome']
            y_seq = np.zeros(num_time_steps)
            time_at_event = min([i for i, mo in enumerate(self.args.month_endpoints)
                                if days_to_censor < (mo * 30)]) if days_to_censor < (max_time * 30) else num_time_steps - 1
            if y:
                y_seq[time_at_event:] = 1
            y_mask = np.array([1] * (time_at_event + 1) + [0] * (num_time_steps - (time_at_event + 1)))

        assert time_at_event >= 0 and len(y_seq) == len(y_mask)
        return y, y_seq.astype('float64'), y_mask.astype('float64'), time_at_event, days_to_censor

    def get_outcome_date(self, events, index_date, end_of_date=END_OF_TIME_DATE):
        '''
        Looks through events to find date of outcome
        If multiple outcomes occur, use the first occurance date.
        args:
        - events: List of event dicts. Each dict must have a CODE and admit_date.
        '''
        outcome_events = [e for e in events if any(icd == e['codes'] for icd in OUTCOME_CODE)]

        if len(outcome_events) > 0:
            ever_develops_outcome = True
            time = min([e['admit_date'] for e in outcome_events])
        else:
            ever_develops_outcome = False
            time = end_of_date
        return ever_develops_outcome, time

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, index):
        if self.shard:
            patient_tmp = self.patients[index]
            patient = json.load(open(patient_tmp['shard_path'], 'r'))[patient_tmp['patient_id']]
            patient.update(patient_tmp)
        else:
            patient = self.patients[index]

        samples = self.get_trajectory(patient)
        items = []
        for sample in samples:
            codes = [e['codes'] for e in sample['events']]
            code_str = " ".join(codes)
            # code_str = np.array(code_str).astype(np.string_)
            x = [self.get_index_for_code(e, self.args.code_to_index_map) for e in sample['events']]
            char_x = [self.get_chars_for_code(code, self.args.char_to_index_map) for code in
                      sample['codes']] if self.args.use_char_embedding else [-1]
            time_seq = sample['time_seq'].tolist()
            age_seq = sample['age_seq'].tolist()
            dx_seq = sample['dx_seq'].tolist()
            ind_seq = sample['ind_seq'].tolist()

            item = {
                'x': self.pad_arr(x, self.args.pad_size, 0),
                'time_seq': self.pad_arr(time_seq, self.args.pad_size, np.zeros(self.args.time_embed_dim)),
                'age_seq': self.pad_arr(age_seq, self.args.pad_size, np.zeros(self.args.time_embed_dim)), 
                'dx_seq': self.pad_arr(dx_seq, self.args.pad_size, np.zeros(self.args.time_embed_dim)),
                'ind_seq': self.pad_arr(ind_seq, self.args.pad_size, np.zeros(self.args.time_embed_dim)),
                'code_str': code_str
            }
            for key in ['y', 'y_seq', 'y_mask', 'time_at_event', 'admit_date', 'exam', 'age', 'dx', 'ind', 'ks', 'sex', 'bmi','race', 'outcome',  'days_to_censor', 'patient_id']:
                item[key] = sample[key]
            items.append(item)
        return items

    def add_noise(self, samples):

        if self.args.confusion == 'replace':
            def confuse(x, library, prob):
                if np.random.random() < self.args.confusion_strength:
                    return np.random.choice(library, p=prob)
                else:
                    return x

            for sample in samples:
                sample['codes'] = [confuse(disease, self.args.all_codes, self.args.all_codes_p) for disease in
                                   sample['codes']]

        if self.args.confusion == 'outcome':
            for sample in samples:
                if np.random.random() < self.args.confusion_strength:
                    sample['y'] = True
                    sample['y_seq'] = np.ones(sample['y_seq'].shape)

        if self.args.confusion == 'dropout':
            for sample in samples:
                n_total = len(sample['codes'])
                n_drop_avai = n_total - self.args.min_events_length
                drop_idx = random.choices(range(n_total), k=n_drop_avai)
                drop_idx = [i for i in drop_idx if np.random.random() < self.args.confusion_strength]
                sample['codes'] = [sample['codes'][i] for i in drop_idx]

        return samples

    def get_index_for_code(self, event, code_to_index_map):
        code = self.get_code(self.args, event)
        pad_index = len(code_to_index_map)
        if code == PAD_TOKEN:
            return pad_index
        if code in code_to_index_map:
            return code_to_index_map[code]
        else:
            return code_to_index_map[UNK_TOKEN]

    def get_chars_for_code(self, code, char_to_index_map):
        code = self.get_code(self.args, code, char=True)
        chars = list(code)
        vec = np.zeros(len(chars))
        for i, c in enumerate(chars):
            vec[i] = char_to_index_map[c] if c in char_to_index_map else char_to_index_map[UNK_TOKEN]
        return vec

    def get_code(self, args, event, char=False):

        if type(event) is dict:
            code = event['codes']
            if 'code_type' in event.keys():
                code_type = event['code_type']
            else:
                code_type = 'other'
        else:
            code = event

        if code_type == 'icd':
            if char:
                trunc_level = max(args.icd8_level, args.icd10_level) + 1
                return '-'*(trunc_level-len(code)) + code[:trunc_level]
            code = code.replace('.', '') 
            if len(code) > 1 and code[0] == 'D' and not code[1].isdigit(): #this means it is a SKS code
                return code[:args.icd10_level +1] # TODO: check replacement before truncation or after?

            elif code.isdigit():
                return code[:args.icd8_level]
            elif (len(code) > 1 and (code[0] == 'Y' or code[0] == 'E')): # TODO: separate SKS or RPDR code by the data class not by filtering
                return code[:args.icd8_level +1]
            
        if code_type in ['drug', 'phe_drug', 'lab', 'phe_lab', 'other']:
            return code

    def pad_arr(self, arr, max_len, pad_value):
        return np.array([pad_value] * (max_len - len(arr)) + arr[-max_len:])
