import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from collections import defaultdict
from IPython.core.debugger import set_trace


from input_utils import *
from metric_utils import *


def train(network, 
          opt, 
          criterion,
          train_params,
          data_params,
          val_data,
          metric_dict=None
          ):

    if metric_dict is None:
        metric_dict = defaultdict(list)
        
    TP = train_params
    
    enable_grad_train = TP.enable_grad_train if hasattr(TP, 'enable_grad_train') else False
    enable_grad_val = TP.enable_grad_val if hasattr(TP, 'enable_grad_val') else False
    
    autograd_context_train = torch.autograd.enable_grad if enable_grad_train else torch.autograd.no_grad
    autograd_context_val = torch.autograd.enable_grad if enable_grad_val else torch.autograd.no_grad
    
    if hasattr(TP, 'val_metrics'):
        val_metrics = TP.val_metrics
    else:
        val_metrics = None
    
    input_torch_val = val_data['inpt']
    output_torch_val = val_data['outpt']
    
    if metric_dict is None:
        metric_dict = defaultdict(list)
    
    iterator = tqdm(range(TP.epochs)) if TP.progress_bar else range(TP.epochs)
    
    for i in iterator:

        # train
        network.train()
        data_input, data_output = create_data(**data_params) # ([d_input,T], [d_output,T])
        random_indexes = np.arange(data_input.shape[1]) # shuffle over time
        if TP.shuffle:
            np.random.shuffle(random_indexes)
        
        input_torch = torch.tensor(data_input[:,random_indexes], dtype=torch.float).to(TP.device)
        output_torch = torch.tensor(data_output[:,random_indexes], dtype=torch.float).to(TP.device)

        T = input_torch.shape[1]
        for t in range(0,T,TP.batch_size):
            with autograd_context_train():

                # train
                network.train()
                output_pred = network.forward(input_torch[:,t:t+TP.batch_size])
                loss = criterion(output_pred[-1], output_torch[:,t:t+TP.batch_size])
                metric_dict['loss_train'].append(loss.item())
                metric_dict['outpt_train'] = output_pred
                
                if TP.learning_type=='BP':
                    opt.zero_grad()
                    loss.backward()
                    
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

        # validation
        with autograd_context_val():
            
            network.eval()
            output_pred_val = network.forward(input_torch_val)
            loss_val = criterion(output_pred_val[-1], output_torch_val)
            metric_dict['loss_val'].append(loss_val.item())
            metric_dict['outpt_val'] = output_pred_val
            
            # calculate val metrics
            if val_metrics is not None:
                for val_metric_name, val_metric in val_metrics.items():
                    v = val_metric(output_pred_val, output_torch_val)
                    metric_dict[val_metric_name + '_val'].append(v.item())
            
            if hasattr(TP, 'stopping_criterion'):
                stopping_criterion_val = TP.stopping_criterion(output_pred_val, output_torch_val)
            else:
                stopping_criterion_val = loss_val.item()
            
        if stopping_criterion_val < TP.tol:
            break
        
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