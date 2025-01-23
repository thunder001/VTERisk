import numpy as np
import math
import sklearn.metrics
import torch
import torch.nn.functional as F
import pdb


def get_model_loss(logit, batch, args):
    y_seq  = batch['y_seq']
    y_mask  = batch['y_mask']
    if args.loss_f == 'binary_cross_entropy_with_logits':
        loss = F.binary_cross_entropy_with_logits(logit, y_seq.float(), weight=y_mask.float(), reduction='sum')/ torch.sum(y_mask.float())
    elif args.loss_f == 'mse':
        loss = F.mse_loss(logit, y_seq.float(), size_average=False, reduction='mean')
    else:
        raise Exception('Loss function is illegal or not found.')
    return loss

def model_step(batch, models, optimizers, train_model,  args):
    '''
        Single step of running model on the a batch x,y and computing
        the loss. Backward pass is computed if train_model=True.
        Returns various stats of this single forward and backward pass.
        args:
        - batch: whole batch dict, can be used by various special args
        - models: dict of models. The main model, named "model" must return logit, hidden, activ
        - optimizers: dict of optimizers for models
        - train_model: whether or not to compute backward on loss
        - args: various runtime args such probability to as batch_split etc
        returns:
        - loss: scalar for loss on batch as a tensor
        - reg_loss: scalar for regularization loss on batch as a tensor
        - preds: predicted labels as numpy array
        - probs: softmax probablities as numpy array
        - golds: labels, numpy array version of arg y
        - exams: exam ids for batch if available
        - censor_times: feature rep for batch
    '''
    logit, aux_loss_dict = models['model'](batch['x'], batch)
    loss = get_model_loss(logit, batch, args)

    if args.pred_mask:
        loss += args.pred_mask_lambda * aux_loss_dict['pred_mask_loss']

    loss /= args.batch_splits

    if train_model:
        loss.backward()
    # changed cpu to cuda to see if that fixes the neuron issue
    probs = torch.sigmoid(logit).cpu().data.numpy()#Shape is B, len(args.month_endpoints)
    preds = probs > .5
    golds = batch['y'].data.cpu().numpy()
    patient_golds = batch['outcome'].data.cpu().numpy()
    exams = batch['exam']
    pids = batch['patient_id']
    censor_times = batch['time_at_event'].cpu().numpy()
    days_to_censor = batch['days_to_censor'].cpu().numpy()
    dates = batch['admit_date']

    return  loss, preds, probs, golds, patient_golds, exams, pids, censor_times, days_to_censor, dates
