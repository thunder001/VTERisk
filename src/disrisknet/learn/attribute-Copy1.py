import disrisknet.learn.train as train
from copy import deepcopy
from disrisknet.utils.learn import init_metrics_dictionary, get_dataset_loader, get_train_variables
from collections import defaultdict
from disrisknet.utils.parsing import CODE2DESCRIPTION
from tqdm import tqdm
from captum.attr import LayerIntegratedGradients, LayerGradientShap,  TokenReferenceBase, visualization
import torch
import pandas as pd
import numpy as np
from functools import partial

torch.backends.cudnn.enabled=False

def compute_attribution(attribute_data, model, args, month_idx):
    # print("model:{}".format(model))
    model = model.to(args.device)
    test_data_loader = get_dataset_loader(args, attribute_data)
    token_reference = TokenReferenceBase(reference_token_idx=0)
    reference_indexes = token_reference.generate_reference(args.pad_size, device=args.device).unsqueeze(0)
    # print("model.model: {}".format(model.model))
    # print(model.model['model'].code_embed)
    lig_code = LayerIntegratedGradients(model, model.model.code_embed)
   
    if hasattr(model.model, 'a_embed_add_fc') and hasattr(model.model, 'a_embed_scale_fc'):
        age_embeddings_layers = [model.model.a_embed_add_fc, model.model.a_embed_scale_fc]
        lig_age = LayerIntegratedGradients(model, age_embeddings_layers)
    else:
        lig_age = None
    # print("lig_age".format(lig_age))
    # print(lig_age)
    test_iterator = iter(test_data_loader)
    word2attr = defaultdict(list)
    word2censor_att = defaultdict(partial(defaultdict,list))
  
    try:
        for i, batch in enumerate(tqdm(test_iterator)):
            batch = train.prepare_batch(batch, args)
            # print(torch.cuda.memory_snapshot())
            # print(batch)
            codes, attr, ages, add_attr_ages, scale_attr_ages, combined_add_ages = attribute_batch(
                lig_code, lig_age, batch, reference_indexes, args, month_idx)
      
            for patient_codes, patient_attr, gold, days in zip(codes, attr, batch['y'], batch['days_to_censor']):
                patient_codes = patient_codes.split()
                time_bin = int(days//30)
                for c,a in zip(patient_codes, patient_attr[-len(patient_codes):]):
                    code = get_code(args, c)
                    word2attr[code].append(a)
                    if gold:
                        word2censor_att[time_bin][code].append(a)
            for patient_age,patient_age_attr in zip(ages, add_attr_ages):
                word2attr["Add-Age-{}".format(patient_age)].append(patient_age_attr)
            for patient_age,patient_age_attr in zip(ages, scale_attr_ages):
                word2attr["Scale-Age-{}".format(patient_age)].append(patient_age_attr)
            for patient_age,patient_age_attr in zip(ages, combined_add_ages):
                word2attr["Combined-Age-{}".format(patient_age)].append(patient_age_attr)
            if i>= args.max_batches_per_dev_epoch:
                break
    except Exception as e:
            print(e)
    return word2attr, word2censor_att


def attribute_batch(explain_code, explain_age, batch, reference_indexes, args, month_idx):
    # import pdb; pdb.set_trace()
    batch_age = deepcopy(batch)
    if explain_code is not None:

        attributions_code = explain_code.attribute(inputs=(batch['x'], batch['age_seq'], batch['time_seq']),
                                        baselines=None, #(reference_indexes, torch.zeros_like(batch['age_seq']).to(args.device), torch.zeros_like(batch['time_seq']).to(args.device)),
                                        n_steps=2,
                                        return_convergence_delta=False,
                                        target=month_idx,
                                        additional_forward_args=batch)
        # print('attribution_code {}'.format(attributions_code))
        attributions_code = attributions_code.sum(dim=2).squeeze(0)
        attributions_code = attributions_code / torch.norm(attributions_code)
        attributions_code = attributions_code.cpu().detach().numpy()
        # print('attribution_code {}'.format(attributions_code))
    else:
        attribution_code = []

    if explain_age:
        # import pdb; pdb.set_trace()
        # print("explain age...")
        attributions_age = explain_age.attribute(inputs=(batch_age['x'],batch_age['age_seq'], batch_age['time_seq']), 
                                    baselines=None,#(reference_indexes, torch.zeros_like(batch['age_seq']).to(args.device), torch.zeros_like(batch['time_seq']).to(args.device)),
                                    n_steps=2, 
                                    return_convergence_delta=False,
                                    target=month_idx,
                                    attribute_to_layer_input=True,
                                    additional_forward_args=(batch_age))


        attributions_age[0] = attributions_age[0].sum(dim=(-1,-2)).squeeze()
        attributions_age[0] = attributions_age[0]/torch.norm(attributions_age[0])
        attributions_age[1] = attributions_age[1].sum(dim=(-1,-2)).squeeze()
        attributions_age[1] = attributions_age[1]/torch.norm(attributions_age[1])

        age_attribution_add = attributions_age[0].cpu().detach().numpy()
        age_attribution_scale = attributions_age[1].cpu().detach().numpy()
        age_attribution_combined = (attributions_age[0] + attributions_age[1]).cpu().detach().numpy()
    else:
        age_attribution_add = []
        age_attribution_scale = []
        age_attribution_combined = []
    
    return batch['code_str'], attributions_code, (batch_age['age']//365).squeeze().tolist(), age_attribution_add, age_attribution_scale, age_attribution_combined


def get_code(args, event, char=False):

        if type(event) is dict:
            code = event['codes']
            if 'code_type' in event.keys():
                code_type = event['code_type']
            else:
                code_type = 'phe_drug'
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
            
        if code_type in ['drug', 'phe_drug', 'phe_lab']:
            return code