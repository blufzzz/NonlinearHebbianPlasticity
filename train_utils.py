import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from collections import defaultdict
from IPython.core.debugger import set_trace

from input_utils import *
from metric_utils import *

'''
inp - [d1,T], layer input 
out - [d2,T], layer output
W - [d2, d1]
|--------|
|--w(i)--|
|--------|
|--------|
'''

def g_krotov(tot_input, k=2, delta=0.4):
    
    '''
    Implicit WTA
    most active - get activation 1
    k-th active - get -delta
    others - 0
    '''
    
    h, batch_size = tot_input.shape
    
    # applying activation
    y = torch.argsort(tot_input, dim=0)
    yl = torch.zeros((h, batch_size), dtype=tot_input.dtype, device=tot_input.device) # [h,T]

    batch_arange = torch.arange(batch_size, device=tot_input.device)
    yl[y[h-1,:], batch_arange] = 1.0 # the  before-last neuron takes it all
    yl[y[h-k], batch_arange] = -delta # the k-th from the end 
    
    return yl


def krotov_rule_linear(inpt, outpt, W):
    
    # WTA
    Wv = W@inpt
    yl = g_krotov(outpt)
    
    xx=(yl*Wv).sum(1) # sum over batch, get [h,]
    dW = yl@inpt.T - xx.unsqueeze(1)*W
    
    return dW


def krotov_rule(inpt, outpt, W):
    
    h, batch_size = outpt.shape
    d = inpt.shape[0]
    
    # WTA
    Wv = W@inpt
    yl = g_krotov(outpt)
    
    xx=(yl*outpt).sum(1) # sum over batch, get [h,]
    dW = yl@inpt.T - xx.unsqueeze(1)*W
    
    return dW
      
    
def oja_rule_linear(inp, out, W):
    d1,T = inp.shape
    Wv = W@inp
    # equation (4) for quadratic error
    # minimizes quadratic representation error $J(W) = ||X - W^Tf(WX)||_2$
    dW = out@(inp.T - Wv.T@W)/T # [d2,:]@([:,d1] - [:,d2]@[d2,d1]) 
    return dW
    
    
def oja_rule(inp, out, W):
    d1,T = inp.shape
    # equation (4) for quadratic error
    # minimizes quadratic representation error $J(W) = ||X - W^Tf(WX)||_2$
    dW = out@(inp.T - out.T@W)/T # [d2,:]@([:,d1] - [:,d2]@[d2,d1]) 
    return dW

def hebb_rule(inp, out, W, weight_decay=0):
    d1,T = inp.shape
    dW = (out@inp.T)/T - weight_decay*W
    return dW


def criterion_rule(inp, out, W):
    # equation (3)
    if isinstance(inp, torch.Tensor):
        d1,T = inp.shape
        device = W.device
        I = torch.eye(d1, device=device) 
    else:
        d1,T = inp.shape
        I = np.eye(d1) # [d1,d1]
        
    dW = (out@inp.T)@(I - W.T@W)/T # [d2,:]@[:,d1]@([d1,d1] - [d1,d1]) 
    return dW


def GHA_rule(inp, out, W):
    d1,T = inp.shape
    i_u, j_u = np.triu_indices(d2, k=1)
    L_out = out@out.T
    L_out[i_u, j_u] = 0
    dW = (out@inp.T - (L_out @ W))/T # [d2,:]@([:,d1] - [d2,d2]@[d2,d1])
    return dW


def bruteforce_projection(n_grid_samples, criterion, X, f=None, w_min=-1.5, w_max=1.5):
    
    '''
    Maximizing projection criterion, by bruteforce
    X - [d,T]
    '''
    
    W_grid = np.stack(np.meshgrid(np.linspace(w_min, w_max, n_grid_samples), 
                                  np.linspace(w_min, w_max, n_grid_samples), indexing='ij'), -1)
    
    W_grid /= np.linalg.norm(W_grid, axis=-1, keepdims=True)

    crit_map = np.empty((n_grid_samples, n_grid_samples))
    for i in range(n_grid_samples):
        for j in range(n_grid_samples):
            w = W_grid[i,j]
            
            if f is not None:
                s = f(w@X)
            else:
                s = w@X
                
            crit = criterion(s)
            crit_map[i,j] = crit
            
    return crit_map, W_grid



def check_nan(*args):
    '''
    input - lists of lists of tensor
    '''
    out = False
    for arg in args:
        if isinstance(arg, torch.Tensor):
            if torch.isnan(arg).any():
                out = True
                break
        elif isinstance(arg, list):
            out = check_nan(*arg)
        else:
            pass
            
    return out

            

def get_grad_params(params):
    return list(filter(lambda x: x.requires_grad, params))

def train(network, 
          opt, 
          criterion,
          criterion_kwargs,
          parameters,
          train_dataloader,
          val_dataloader,
          metric_dict=None,
          val_metrics=None,
          ):
    
    '''
    Versatile function for training different combinations of models, criteria and learning
    '''
    
    TP = parameters

    if metric_dict is None:
        metric_dict = defaultdict(list)
        
    if not hasattr(TP, 'maxiter') or TP.maxiter is None:
        TP.maxiter = np.inf
        
    if hasattr(TP, 'val_metrics'):
        val_metrics = TP.val_metrics
    else:
        val_metrics = None
        
    device = TP.device
    
    enable_grad_train = TP.enable_grad_train if hasattr(TP, 'enable_grad_train') else False
    enable_grad_val = TP.enable_grad_val if hasattr(TP, 'enable_grad_val') else False
    
    autograd_context_train = torch.autograd.enable_grad if enable_grad_train else torch.autograd.no_grad
    autograd_context_val = torch.autograd.enable_grad if enable_grad_val else torch.autograd.no_grad
    
    iterator = tqdm(range(TP.epochs), position=0, leave=True) if TP.progress_bar else range(TP.epochs)
    
    # iterate over epoches
    for epoch in iterator:
        
        #########
        # TRAIN #
        #########
        network.train()
        metric_dict_train = defaultdict(list)
        # iterate over batches
        for itr, input_batch in enumerate(train_dataloader):
            
            # early_stopping
            if itr >= TP.maxiter:
                break
            
            with autograd_context_train():

                network.train()
                
                # common call for all models
                input_batch = input_batch.to(device)
                output_batch = network.forward(input_batch)
                
                if check_nan(*output_batch[0]):
                    raise RuntimeError('NaN in `output_batch`!')
                
                if not criterion_kwargs['skip_train']:
                    criterion_train = criterion(output_batch, input_batch, **criterion_kwargs['train'])
                    metric_dict_train['criterion_train'].append(criterion_train.item())
                
                
                ##################
                # WEIGHTS UPDATE #
                ##################
                if TP.backprop_learning:
                    opt.zero_grad()
                    criterion_train.backward()
                    
                    if TP.calculate_grad:
                        metric_dict_train['grad_norm_train'].append(calc_gradient_norm(get_grad_params(network.parameters())))
                    
                    if TP.clip_grad_value is not None:
                        nn.utils.clip_grad_norm_(network.parameters(), TP.clip_grad_value)
                    
                    opt.step()
                    
                if TP.hebbian_learning:
                    # tuple: (list of layer's outputs, final output)
                    network.hebbian_learning_step(*output_batch, readout=None)
                    
                
        
        # end of epoch
        ######################################################################################
        if TP.weight_saver is not None:
             metric_dict['weights'].append(TP.weight_saver(network))
        
        # saving train per-epoch mean statistics
        for k,v in metric_dict_train.items():
            metric_dict[k].append(np.mean(v))

        ##############
        # VALIDATION #
        ##############
        
        if val_dataloader is None:
            continue
            
        network.eval()
        metric_dict_val = defaultdict(list)
        with autograd_context_val():
            for input_batch in val_dataloader:

                output_batch = network.forward(input_batch)

                if not criterion_kwargs['skip_val']:
                    criterion_val = criterion(output_batch, input_batch, **criterion_kwargs['val'])
                    metric_dict_val['criterion_val'].append(criterion_val.item())

                # calculate val metrics
                if val_metrics is not None:
                    for val_metric_name, val_metric in val_metrics.items():
                        v = val_metric(output_batch, input_batch)
                        metric_dict_val[metric_name + '_val'] = v.item()
            
            # saving val per-epoch mean statistics 
            for k,v in metric_dict_val.items():
                metric_dict[k].append(np.mean(v))

            if hasattr(TP, 'stopping_criterion') and TP.stopping_criterion is not None:
                stopping_criterion_val = TP.stopping_criterion(output_batch, input_batch)

                # early stopping
                assert stopping_criterion_val > -1e-5, '`stopping_criterion_val` must be positive!'
                if stopping_criterion_val < TP.tol:
                    print('STOPPING!')
                    break
            else:
                # no stopping criteria, continue training
                continue

        
    return network, opt, metric_dict
    
    
def plot_weights_hist(network, suptitle=None):

    N_p = len(list(network.parameters()))
    fig, axes = plt.subplots(ncols=N_p, nrows=1, figsize=(N_p*5,5))

    for i,(name,p) in enumerate(list(network.named_parameters())):
        axes[i].hist(to_numpy(p.data.flatten()), bins=50)
        axes[i].set_title(name)
        
    fig.suptitle(suptitle, color='blue')

    plt.show()
    
def get_capacity(model):
    s_total = 0
    for param in model.parameters():
        s_total+=param.numel()
    return round(s_total,2)


def calc_gradient_norm(parameters):
    total_norm = 0.0
    for i,p in enumerate(parameters):
        try:
            param_norm = p.grad.data.norm(2)
        except:
            set_trace()
        total_norm += param_norm.item()
    total_norm = total_norm 
    return np.mean(total_norm)