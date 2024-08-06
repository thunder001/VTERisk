import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from disrisknet.models.pools.factory import get_pool
from disrisknet.models.utils import Cumulative_Probability_Layer
from disrisknet.models.factory import RegisterModel

class AbstractRiskModel(nn.Module):
    """
        A neural risk model with a discrete time survival objective.
    """

    def __init__(self, args):

        super(AbstractRiskModel, self).__init__()

        self.args = args

        if args.use_char_embedding:
            self.vocab_size = len(args.char_to_index_map) + 1
            self.char_embed = nn.Embedding(len(args.char_to_index_map) + 1, args.char_dim)
            self.char_rnn = nn.GRU(input_size= args.char_dim, hidden_size=args.hidden_dim, num_layers=1, bidirectional=False, batch_first=True)
            
        else:
            self.vocab_size = len(args.code_to_index_map) + 1
            # print(self.vocab_size)
            # print(type(self.vocab_size))
            self.code_embed = nn.Embedding(self.vocab_size, args.hidden_dim, padding_idx = 0)

        kept_token_vec = torch.nn.Parameter(torch.ones([1,1,1]),
            requires_grad=False)
        self.register_parameter('kept_token_vec', kept_token_vec)

        self.pool = get_pool(args.pool_name)(args)
        self.dropout = nn.Dropout(p=args.dropout)
        if args.model_name not in ['cox', 'add_cox']:
            hidden_dim =  args.hidden_dim
            if self.args.add_age_neuron:
                hidden_dim = hidden_dim+1  
            if self.args.add_ks_neuron: 
                hidden_dim = hidden_dim+1 
            if self.args.add_sex_neuron: 
                hidden_dim = hidden_dim+1 
            if self.args.add_bmi_neuron: 
                hidden_dim = hidden_dim+1 
            self.prob_of_failure_layer = Cumulative_Probability_Layer(hidden_dim, len(args.day_endpoints), args)

        if args.use_time_embed:
            self.t_embed_add_fc = nn.Linear(args.time_embed_dim, args.hidden_dim)
            self.t_embed_scale_fc = nn.Linear(args.time_embed_dim, args.hidden_dim)

        if args.use_age_embed:
            self.a_embed_add_fc = nn.Linear(args.time_embed_dim, args.hidden_dim)
            self.a_embed_scale_fc = nn.Linear(args.time_embed_dim, args.hidden_dim)
        
        if args.use_dxtime_embed:
            self.d_embed_add_fc = nn.Linear(args.time_embed_dim, args.hidden_dim)
            self.d_embed_scale_fc = nn.Linear(args.time_embed_dim, args.hidden_dim)    
            

        if args.pred_mask:
            self.mask_embedding = torch.nn.Embedding(2, args.hidden_dim, padding_idx=1)
            if args.use_char_embedding:
                raise NotImplementedError("Pred mask support when using only char embeddings is not supported!")
            self.pred_masked_fc = nn.Linear(args.hidden_dim, self.vocab_size)


    def mask_input(self, x, obs_seq):
        B, N, _ = x.size()
        mask_prob = self.args.mask_prob if self.training else 0
        is_mask = torch.bernoulli( self.kept_token_vec.expand([B,N,1]) * mask_prob ) #0 is not masked, 1 is masked
        # Don't mask out any PAD tokens
        is_mask = is_mask * (obs_seq).unsqueeze(-1).float() # sets all not viewed as 0
        is_kept = 1 - is_mask
        x = x * is_kept + self.mask_embedding(is_kept.squeeze(-1).long())
        return x, is_mask

    def condition_on_pos_embed(self, x, embed, embed_type='time'):
        if embed_type == 'time':
            return self.t_embed_scale_fc(embed) * x + self.t_embed_add_fc(embed)
        if embed_type == 'age':
            return self.a_embed_scale_fc(embed) * x + self.a_embed_add_fc(embed)
        if embed_type == 'dxtime':
            return self.d_embed_scale_fc(embed) * x + self.d_embed_add_fc(embed)
        else:
            raise NotImplementedError("Embed type {} not supported".format(embed_type))


    def get_pred_mask_loss(self, seq_hidden, x, is_mask):
        if is_mask.sum().item() == 0:
            return 0
        seq_hidden = seq_hidden.transpose(1,2) # CHANGED FROM 2 to 3
        B, N, D_n = seq_hidden.size()
        hidden_for_mask = torch.masked_select(seq_hidden, is_mask.byte()).view(-1, D_n)
        pred_x = self.pred_masked_fc(hidden_for_mask)
        x_for_mask = torch.masked_select(x.unsqueeze(-1), is_mask.byte())
        return F.cross_entropy(pred_x, x_for_mask)
     
    def get_embeddings(self, x, batch=None):
        if self.args.use_char_embedding:
           char_x = self.char_embed(batch['char_x'].long()) # B, L_SEQ, L_CODE, D
           B, L_seq, L_code, D = char_x.size()
           char_x = char_x.view( (B*L_seq, L_code, D))
           h, _ = self.char_rnn(char_x) # B*L_SEQ, L_CODE, H
           h = h[:, -1, :]
           token_embed = h.view(B, L_seq, -1)
        else:
            token_embed = self.code_embed(x.long())
        return token_embed

    def forward(self, x, batch=None):
        pad_token_indx = 0 # Note, change this to be -1 or something not as terrible.
        embed_x = self.get_embeddings(x, batch)
        obs_seq = x != pad_token_indx

        if self.args.use_time_embed:
            time = batch['time_seq'].float()
            embed_x = self.condition_on_pos_embed(embed_x, time, 'time')

        if self.args.use_age_embed:
            age = batch['age_seq'].float()
            embed_x = self.condition_on_pos_embed(embed_x, age, 'age')
       
        if self.args.use_dxtime_embed:
            dxtime = batch['dx_seq'].float()
            embed_x = self.condition_on_pos_embed(embed_x, dxtime, 'dxtime')

        if self.args.pred_mask:
            raise Exception ("Prediction mask is not enabled.")
            masked_x, is_mask = self.mask_input(embed_x, obs_seq)

        seq_hidden = self.encode_trajectory(embed_x, batch)
        seq_hidden = seq_hidden.transpose(1,2)
                    # changed from 2 to 3
        
        hidden = self.dropout(self.pool(seq_hidden))
        if self.args.add_age_neuron:
            age_in_year = batch['age']/365.
            hidden = torch.cat((hidden, age_in_year), axis=-1)
       
        # pdb.set_trace()
        if self.args.add_ks_neuron:
            if self.args.neuron_norm:
                ks = F.normalize(batch['ks'], p=1.0, dim=0)
            ks = batch['ks'].int()
            ks = ks.reshape(len(ks), 1)
            hidden = torch.cat((hidden, ks), axis=-1)
            
        if self.args.add_sex_neuron:
            if self.args.neuron_norm:
                sex = F.normalize(batch['sex'], p=1.0, dim=0)
            sex = batch['sex'].int()
            sex = sex.reshape(len(sex), 1)
            hidden = torch.cat((hidden, sex), axis=-1)            
            
        if self.args.add_bmi_neuron:
            if self.args.neuron_norm:
                bmi = F.normalize(batch['bmi'], p=1.0, dim=0)
            bmi = batch['bmi'].int()
            bmi = sex.reshape(len(bmi), 1)
            hidden = torch.cat((hidden, bmi), axis=-1)                   
            
        logit = self.prob_of_failure_layer(hidden)

        aux_loss = {}
        if self.args.pred_mask:
            aux_loss['pred_mask_loss'] = self.get_pred_mask_loss(seq_hidden, x, is_mask)
        return logit, aux_loss
