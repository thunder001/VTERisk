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
import pdb

torch.backends.cudnn.enabled=False

ATTRIBUTION_METHODS = {
    "layer_ig": LayerIntegratedGradients
}
PAD_INDEX = 0
def get_baseline_zero_tensor(args, inputs):
        return tuple([torch.zeros((input_tensor.shape[0],*input_tensor.shape[1:])).to(args.device) for input_tensor in inputs])


def compute_attribution(attribute_data, model, args, month_idx, neg=False, both = False):
    #pdb.set_trace()
    model = model.to(args.device)
    test_data_loader = get_dataset_loader(args, attribute_data)
    token_reference = TokenReferenceBase(reference_token_idx=0)
    reference_indexes = token_reference.generate_reference(args.pad_size, device=args.device).unsqueeze(0)
    
    # lig_code = LayerIntegratedGradients(model, model.model.code_embed)
    print("model.model: {}".format(model.model))
    lig_code = ATTRIBUTION_METHODS["layer_ig"](model, model.model.code_embed)
    if hasattr(model.model, 'a_embed_add_fc') and hasattr(model.model, 'a_embed_scale_fc'):
        age_embeddings_layers = [model.model.a_embed_add_fc, model.model.a_embed_scale_fc]
        lig_age = ATTRIBUTION_METHODS["layer_ig"](model, age_embeddings_layers)
     
    else:
        lig_age = None    
    #if hasattr(model.model, 'i_embed_add_fc') and hasattr(model.model, 'i_embed_scale_fc'):
    #    age_embeddings_layers = [model.model.i_embed_add_fc, model.model.i_embed_scale_fc]        
    test_iterator = iter(test_data_loader)
    word2attr = defaultdict(list)
    word2censor_att = defaultdict(partial(defaultdict,list))
    print('line51')
    try:
        for i, batch in enumerate(test_iterator):
            batch = train.prepare_batch(batch, args)
        
            codes, attr = attribute_batch(lig_code, lig_age, 
                                          batch, reference_indexes, args, month_idx)
            for patient_codes, patient_attr, gold, days in zip(codes, attr, batch['y'], batch['days_to_censor']):
                patient_codes = patient_codes.split()                
                time_bin = int(days//args.days_bin)
                for c,a in zip(patient_codes, patient_attr[-len(patient_codes):]):
                    code = get_code(args, c)
                    word2attr[code].append(a)        
                    #word2censor_att[time_bin][code].append(a)
                    if both:
                        word2censor_att[time_bin][code].append(a)
                    if neg:
                        if not gold:
                            word2censor_att[time_bin][code].append(a)
                    else:
                        if gold:
                            word2censor_att[time_bin][code].append(a)
            if i>= args.max_batches_per_dev_epoch:
                break
    except Exception as e:
            print(e)
    return word2attr, word2censor_att


def attribute_batch(explain_code, explain_age, 
                    batch, reference_indexes, args, month_idx):
    
    if explain_code is not None:
        inputs=(batch['x'], batch['age_seq'], batch['time_seq'], batch['dx_seq'], batch['ind_seq'], batch['sex'], batch['race'], batch['bmi'])
        ATTRIBUTION_PARAMS = {
            "layer_ig": {"n_steps": 10, "return_convergence_delta": False, "baselines": None}
        }
        attributions_code = explain_code.attribute(inputs=inputs,
                                        target=month_idx,  **ATTRIBUTION_PARAMS["layer_ig"])
        attributions_code = attributions_code.sum(dim=2).squeeze(0)
    else:
        attribution_code = []
    
    return batch['code_str'], attributions_code



def get_code(args, event, char=False):
        if type(event) is dict:
            code = event['codes']
            if 'code_type' in event.keys():
                code_type = event['code_type']
            else:
                code_type = 'other'

            if code_type == 'icd':
                if char:
                    trunc_level = max(args.icd8_level, args.icd10_level) + 1
                    return '-'*(trunc_level-len(code)) + code[:trunc_level]
                code = code.replace('.', '') 
                if len(code) > 1 and code[0] == 'D' and not code[1].isdigit():  
                    return code[:args.icd10_level +1]  

                elif code.isdigit():
                    return code[:args.icd8_level]
                elif (len(code) > 1 and (code[0] == 'Y' or code[0] == 'E')): # TODO: separate SKS or RPDR code by the data class not by filtering
                    return code[:args.icd8_level +1]
            #if code_type in ['drug', 'phe_drug', 'lab', 'phe_lab', 'other']:
            if code_type in ['drug', 'phe_drug']:
                    return code
        else:
             return event