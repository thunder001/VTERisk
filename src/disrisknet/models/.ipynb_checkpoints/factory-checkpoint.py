import torch
from torch import nn
import pdb
import disrisknet.learn.state_keeper as state

MODEL_REGISTRY = {}

NO_MODEL_ERR = 'Model {} not in MODEL_REGISTRY! Available models are {} '
NO_OPTIM_ERR = 'Optimizer {} not supported!'

def RegisterModel(model_name):
    """Registers a configuration."""

    def decorator(f):
        MODEL_REGISTRY[model_name] = f
        return f

    return decorator


def get_model(args):
    return get_model_by_name(args.model_name, True, args)


def get_model_by_name(name, allow_wrap_model, args):
    '''
        Get model from MODEL_REGISTRY based on args.model_name
        args:
        - name: Name of model, must exit in registry
        - allow_wrap_model: whether or not override args.wrap_model and disable model_wrapping.
        - args: run ime args from parsing

        returns:
        - model: an instance of some torch.nn.Module
    '''
    if not name in MODEL_REGISTRY:
        raise Exception(
            NO_MODEL_ERR.format(
                name, MODEL_REGISTRY.keys()))


    model = MODEL_REGISTRY[name](args)
    return model

def load_model(path, args, do_wrap_model = True):
    print('\nLoading model from [%s]...' % path)
    try:
        model = torch.load(path, map_location=torch.device('cpu'))
        if isinstance(model, dict):
            model = model['model']

        if isinstance(model, nn.DataParallel):
            model = model.module.cpu()
    except:
        try:  # TODO: test if we can remove the exception and keep a single version
            state_keeper = state.StateKeeper(args)
            # state_keeper.identifier = path
            models, *_ = state_keeper.load()
            model=models['model']
        except:
            raise Exception(
                    "Sorry, snapshot {} does not exist and could not resume any model from state keeper".format(path))
    
    return model



def get_params(model):
    '''
    Helper function to get parameters of a model.
    '''

    return model.parameters()


def get_optimizer(model, args):
    '''
    Helper function to fetch optimizer based on args.
    '''
    params = [param for param in model.parameters() if param.requires_grad]
    if args.optimizer == 'adam':
        return torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adagrad':
        return torch.optim.Adagrad(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        return torch.optim.SGD(params,
                              lr=args.lr,
                              weight_decay=args.weight_decay,
                              momentum=args.momentum )
    else:
        raise Exception(NO_OPTIM_ERR.format(args.optimizer))

