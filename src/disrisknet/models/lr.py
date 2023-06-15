import math
import torch
import torch.nn as nn
import pdb
import numpy as np
from disrisknet.models.pools.factory import get_pool
from disrisknet.models.factory import RegisterModel
from disrisknet.models.utils import One_Hot_Layer
import torch.nn.functional as F
from disrisknet.models.abstract_risk_model import AbstractRiskModel


class OneHotRiskModel(AbstractRiskModel):
    """
        A basic logistic regression model that uses one hot embedding.
    """

    def __init__(self, args):
        
        super(OneHotRiskModel, self).__init__(args)
        self.args = args
        if not args.use_char_embedding:
            self.code_embed = One_Hot_Layer(num_classes=self.vocab_size, padding_idx = 0)
        else:
            raise NotImplementedError("Char embedding is not supported in cox models!")
        num_features_in = self.vocab_size + 1 if args.use_age_in_cox else self.vocab_size
        self.add_module('linear_layer', nn.Linear(num_features_in, 1, bias=False))

        self.pool = get_pool(args.pool_name)(args)

    def forward(self, x, batch=None):
        # Overrides forward() and skip char and time embedding 
        embed_x = self.code_embed(x)
        seq_hidden = self.pool(embed_x.transpose(1,2))
        if self.args.use_age_in_cox:
            age_in_year = batch['age']/365.
            seq_hidden = torch.cat((seq_hidden, age_in_year), axis=1)
        seq_hidden = self.dropout(seq_hidden)
        seq_hidden = self._modules['linear_layer'](seq_hidden)
        logit = self.get_cox_logit(seq_hidden)

        aux_loss = {}
        if self.args.pred_mask:
            obs_seq = x != pad_token_indx
            obs_seq = x != pad_token_indx
            masked_x, is_mask = self.mask_input(embed_x, obs_seq)
            aux_loss['pred_mask_loss'] = self.get_pred_mask_loss(seq_hidden, x, is_mask)
        return logit, aux_loss


@RegisterModel("cox")
class ProportionalHazards(OneHotRiskModel):

    def __init__(self, args):

        assert args.pool_name != 'Softmax_AttentionPool', "COX models are not compatible with attention poolings"
        super(ProportionalHazards, self).__init__(args)
        self.add_module('linear_layer_hz', nn.Linear(1, len(self.args.month_endpoints), bias=False))
        self.exp = torch.exp
    

    def get_cox_logit(self, seq_hidden):
        '''
        seq_hidden (float): scalar input
        '''
        hazard_ratio = self.exp(seq_hidden)
        hazards = self._modules['linear_layer_hz'](hazard_ratio)
        return hazards


@RegisterModel("add_cox")
class AdditiveHazards(OneHotRiskModel):

    def __init__(self, args):
        
        assert args.pool_name != 'Softmax_AttentionPool', "COX models are not compatible with attention poolings"
        super(AdditiveHazards, self).__init__(args)
        self.baseline_hazard = nn.Parameter(torch.zeros(len(self.args.month_endpoints)))
        self.register_parameter('baseline_hazard', self.baseline_hazard)
    

    def get_cox_logit(self, seq_hidden):
        '''
        seq_hidden (float): scalar input
        '''
        hazards = self.baseline_hazard + seq_hidden
        return hazards
