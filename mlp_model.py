import numpy as np
import torch
from torch import autograd
from torch import nn
from IPython.core.debugger import set_trace
from metric_utils import to_numpy
from models_utils import init_weights
from train_utils import *


class MLP_NonlinearEncoder(nn.Module):
    
    '''
    Feed-forward MLP without recurrent connections
    '''
    
    def __init__(self,**kwargs):
        
        super().__init__()
        
        for k, v in kwargs.items():
            setattr(self, k, v)
            
        # initialize
        if self.set_seed:
            torch.manual_seed(self.seed)
            
        # need gradients only for BP algorithm
        W_s_grad = self.W_requires_grad
        
        # for parameters
        W_s = []
        # for activations
        self.f_s = []
        # for batch-normalization
        if self.add_bn:
            BN_s = []
        
        # fill layers
        for layer in range(self.layers_number):
            
            out_dim = self.embedding_dim if layer==self.layers_number-1 else self.hidden_dim
            in_dim = self.input_dim if layer==0 else self.hidden_dim
                
            W_s.append(nn.Parameter(torch.zeros(out_dim, in_dim), requires_grad=W_s_grad))
            
            if self.add_bn:
                BN_s.append(nn.BatchNorm1d(out_dim, affine=False, track_running_stats=False))
            
            # add activation function
            if not self.final_nonlinearity and layer == self.layers_number-1:
                self.f_s.append(nn.Identity())
            else:
                self.f_s.append(self.create_f(out_dim))
            
            
        if self.add_readout:
            self.W_out = nn.Parameter(torch.zeros(1, out_dim), requires_grad=W_s_grad)
        
        # create parameter lists
        self.W_s = nn.ParameterList(W_s)
        if self.parametrized_f:
            self.f_s = nn.ModuleList(self.f_s)
        if self.add_bn:
            self.BN_s = nn.ModuleList(BN_s)
        
        init_weights(self)


    def create_f(self, input_dim):
        if self.parametrized_f:
            return self.nonlinearity(input_dim, **self.f_kwargs)
        else:
            # default
            return self.f_kwargs['function']

        
    def hebbian_learning_step(self, X_s, readout=None):
        
        '''
        X_s: [[d_1,], ..., [d_k,T]] - layer activations
        readout: [d,T] - ground-truth output
        '''
        
        inp = X_s[0] 
        # hebbian update for intermediate layers
        for i, W in enumerate(self.W_s, start=1):
            
            out = X_s[i]
            
            dW = self.hebbian_update(inp, out, W.data)
            W.data = W.data + self.lr_hebb*dW
            
            if self.normalize_hebbian_update:
                # helps to avoid weight explosion
                W.data = W.data / torch.norm(W.data, dim=1, keepdim=True)
            
            inp = out
            
        if self.add_readout:
            # delta-rule update for the readout layer
            delta = readout - X_s[-1]  # [1,T]
            dW_out = delta@inp.T
            self.W_out.data = self.W_out.data + self.lr*dW_out
        
    def forward(self,X):
        
        '''
        X input batch - [T,d]
        '''
        
        batch_size, dim = X.shape
        X = X.T # to make it [d,T]
        X_s = [X]
        
        for i, W in enumerate(self.W_s):
            
            if self.add_bn and batch_size > 1:
                # transpose to [T,d] for BS layer and back to [d,T]
                X = self.BN_s[i](X.T).T 
            
            if self.inplace_update:
                Y = self.f_s[i](W@X)
                dW = self.hebbian_update(X, Y, W)
                X = self.f_s[i]((W + self.lr_hebb*dW)@X) 
            else:
                X = self.f_s[i](W@X) # [d,T]

            X_s.append(X) 
            
        # single ouput readout
        if self.add_readout:
            X = self.W_out@X
            X_s.append(X)

        return X_s

