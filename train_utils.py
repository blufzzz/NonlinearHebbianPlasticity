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

def NonlinearGHA(inp, out, W):
    raise NotImplementedError()
#             dW = torch.zeros_like(W)
#             errs = torch.zeros_like(W)
#             errs[0] = inp
#             for i in range(d2):
#                 y_i = w_i @ n
#                 f_i = f(y_i)
#                 g_i = torch.autograd.grad(f_i, y_i)
#                 w_i = W[i]
#                 dW[i] = 
        
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
    for arg in args:
        if torch.isnan(arg).any():
            return True


def train(network, 
          opt, 
          criterion,
          train_params,
          gata_generator,
          val_data,
          criterion_kwargs=None,
          metric_dict=None
          ):
    
    TP = train_params

    if metric_dict is None:
        metric_dict = defaultdict(list)
        
    if hasattr(TP, 'val_metrics'):
        val_metrics = TP.val_metrics
    else:
        val_metrics = None
        
    
    enable_grad_train = TP.enable_grad_train if hasattr(TP, 'enable_grad_train') else False
    enable_grad_val = TP.enable_grad_val if hasattr(TP, 'enable_grad_val') else False
    
    autograd_context_train = torch.autograd.enable_grad if enable_grad_train else torch.autograd.no_grad
    autograd_context_val = torch.autograd.enable_grad if enable_grad_val else torch.autograd.no_grad
    
    
    # val data
    input_torch_val = val_data['inpt']
    output_torch_val = val_data['outpt']
    
    iterator = tqdm(range(TP.epochs)) if TP.progress_bar else range(TP.epochs)
    
    # iterate over epoches
    for i in iterator:

        # train
        network.train()
        data_input, data_output, _ = gata_generator() # ([d_input,T], [d_output,T])
        random_indexes = np.arange(data_input.shape[1]) # shuffle over time
        if TP.shuffle:
            np.random.shuffle(random_indexes)
        
        input_torch = torch.tensor(data_input[:,random_indexes], dtype=torch.float).to(TP.device)
        output_torch = torch.tensor(data_output[:,random_indexes], dtype=torch.float).to(TP.device)

        T = input_torch.shape[1]
        criterion_train_list = []
        for t in range(0,T,TP.batch_size):
            with autograd_context_train():

                network.train()
                
                # create batches
                input_torch_batch = input_torch[:,t:t+TP.batch_size]
                output_torch_batch = output_torch[:,t:t+TP.batch_size]
                
                # common call for all models
                output_pred = network.forward(input_torch_batch)
                if check_nan(*output_pred):
                    set_trace()
                
#                 set_trace()
                
                criterion_train = criterion(output_pred[-1], 
                                            output_torch_batch,
                                            **criterion_kwargs['train'])
                
                criterion_train_list.append(criterion_train.item())
                
                ##################
                # WEIGHTS UPDATE #
                ##################
                if TP.learning_type=='BP':
                    opt.zero_grad()
                    criterion_train.backward()
                    
                    if TP.calculate_grad:
                        metric_dict['grad_norm'].append(calc_gradient_norm(filter(lambda x: x[1].requires_grad, 
                                                        network.named_parameters())))
                    
                    if TP.clip_grad_value is not None:
                        nn.utils.clip_grad_norm_(network.parameters(), TP.clip_grad_value)
                    opt.step()
                    
                else:
                    network.hebbian_learning_step(input_torch[:,t:t+TP.batch_size], 
                                                  output_pred, # list of layer activations
                                                  output_torch[:,t:t+TP.batch_size],
                                                  learning_type=TP.learning_type,
                                                  learning_rate=TP.lr,
                                                  weight_decay=TP.wd)
                    
                if TP.weight_saver is not None:
                     metric_dict['weight'].append(TP.weight_saver(network))
        
        # end of epoch
        ######################################################################################
        
        metric_dict['criterion_train'].append(np.mean(criterion_train_list))
        
        
        # validation
        with autograd_context_val():
            
            network.eval()
            output_pred_val = network.forward(input_torch_val)
            
            criterion_val = criterion(output_pred_val[-1], 
                                      output_torch_val,
                                      **criterion_kwargs['val'])
            
            metric_dict['criterion_val'].append(criterion_val.item())
            metric_dict['outpt_val'] = output_pred_val
            
            
            # calculate val metrics
            if val_metrics is not None:
                for val_metric_name, val_metric in val_metrics.items():
                    v = val_metric(output_pred_val, output_torch_val)
                    metric_dict[val_metric_name + '_val'].append(v.item())
                    
            if hasattr(TP, 'stopping_criterion') and TP.stopping_criterion is not None:
                stopping_criterion_val = TP.stopping_criterion(output_pred_val, output_torch_val)
        
                # early stopping
                assert stopping_criterion_val > -1e-5, '`stopping_criterion_val` must be positive!'
                if stopping_criterion_val < TP.tol:
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


def calc_gradient_norm(named_parameters):
    total_norm = 0.0
    for name, p in named_parameters:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item()
    total_norm = total_norm 
    return np.mean(total_norm)