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
            
        # for parameters
        self.W_s = []
        # for activations
        self.f_s = []
        # for batch-normalization
        if self.add_bn:
            self.BN_s = []
        # for recurrent connections
        if self.add_recurrent_connections:
            self.W_r_s = []
        if self.add_recurrent_nonlinearity:
            self.f_s_r = []
        
        # fill layers
        for layer in range(self.layers_number):
            
            out_dim = self.embedding_dim if layer==self.layers_number-1 else self.hidden_dim
            in_dim = self.input_dim if layer==0 else self.hidden_dim
                
            # adding feed-forward connections
            self.W_s.append(nn.Parameter(torch.zeros(out_dim, in_dim), 
                                    requires_grad=self.W_requires_grad))
            # adding recurrent connections
            if self.add_recurrent_connections:
                self.W_r_s.append(nn.Parameter(torch.zeros(out_dim, out_dim), 
                                        requires_grad=self.W_r_requires_grad))
            
            if self.add_bn:
                self.BN_s.append(nn.BatchNorm1d(out_dim, affine=False, track_running_stats=False))
            
            # add activation function
            if not self.final_nonlinearity and layer == self.layers_number-1:
                self.f_s.append(nn.Identity())
                if self.add_recurrent_nonlinearity:
                    self.f_s_r.append(nn.Identity())
            else:
                self.f_s.append(self.create_f(out_dim))
                if self.add_recurrent_nonlinearity:
                    self.f_s_r.append(self.create_f(out_dim))
            
        if self.add_readout:
            self.W_out = nn.Parameter(torch.zeros(1, out_dim), 
                                      requires_grad=self.W_requires_grad)
        
        # create parameter lists
        self.W_s = nn.ParameterList(self.W_s)
        if self.parametrized_f:
            self.f_s = nn.ModuleList(self.f_s)
            if self.add_recurrent_nonlinearity:
                self.f_s_r = nn.ModuleList(self.f_s_r)
        
        if self.add_bn:
            self.BN_s = nn.ModuleList(self.BN_s)
        
        if self.add_recurrent_connections:
            self.W_r_s = nn.ParameterList(self.W_r_s)
        
        init_weights(self)

    def create_f(self, input_dim):
        '''
        Creating (possibly parametrized) non-linearity
        '''
        if self.parametrized_f:
            return self.nonlinearity(input_dim, **self.f_kwargs)
        else:
            return self.nonlinearity
        
        
    def hebbian_learning_step(self, layer_outputs, out, readout=None):
        
        '''
        layer_outputs: [[X, Y, Y_f, Y_fr, Y_frf,]_1, 
                        ...., 
                        [X, Y, Y_f, Y_fr, Y_frf,]_k]
                 
        readout: [d,T] - ground-truth output
        '''
         
        # hebbian update for intermediate layers
        for layer_number in range(self.layers_number):
            
            X, Y, Y_f, Y_fr, Y_frf = layer_outputs[layer_number]
            batch_size = X.shape[1]
            
            if self.add_recurrent_connections:
                dW = self.hebbian_update(X, Y_fr, self.W_s[layer_number])
                
                # anti-hebbian update
                self.W_r_s[layer_number] += self.lr_hebb*(-1.0*Y_frf@Y_frf.T)/batch_size
            
            else:
                dW = self.hebbian_update(X, Y_f, self.W_s[layer_number])
                self.W_s[layer_number] += self.lr_hebb*dW
            
            if self.normalize_hebbian_update:
                # helps to avoid weight explosion
                self.W_s[layer_number] /= torch.norm(self.W_s[layer_number], dim=1, keepdim=True)
                if self.add_recurrent_connections:
                    self.W_r_s[layer_number] /= torch.norm(self.W_r_s[layer_number], dim=1, keepdim=True)
            
        if self.add_readout:
            # delta-rule update for the readout layer
            delta = readout - out  # [1,T]
            dW_out = delta@inp.T
            self.W_out.data = self.W_out.data + self.lr*dW_out
    
    def single_layer_forward(self, X, layer_number):
        
        '''
        Outputs results of consecutive operations in the layer
        X - input
        Y - output after synapses W layer
        Y_f - after nonlinearity f
        Y_fr - after recurrent connections
        Y_frf - after nonlinearity f
        '''
        
        batch_size = X.shape[1]
        
        # normalize batch of [d,T] if needed
        if self.add_bn and batch_size > 1:
            # transpose to [T,d] for BS layer and back to [d,T]
            X = self.BN_s[layer_number](X.T).T 
        
        Y = self.W_s[layer_number]@X # [d,T]
        Y_f = self.f_s[layer_number](Y)
        
        Y_fr = None
        Y_frf = None
        if self.add_recurrent_connections:
            Y_fr = self.W_r_s[layer_number]@Y_f
            if self.add_recurrent_nonlinearity:
                Y_frf = self.f_s_r[layer_number](Y_fr)
                
        return [X, Y, Y_f, Y_fr, Y_frf]
    
    def hebbian_update(self, ):
        
        return self.hebbian_update_fun()
        
    def forward(self,X):
        
        '''
        X input batch - [T,d]
        '''
        
        batch_size, dim = X.shape
        X = X.T # to make it [d,T]
        layer_outputs = []
        
        for layer_number in range(self.layers_number):
            
            layer_output = self.single_layer_forward(X, layer_number)
            # activation to pass to the next layer
            X = layer_output[-1] if self.add_recurrent_connections else layer_output[1]
            
            if self.inplace_update:
                  raise NotImplementedError  
                    
#                 layer_output = self.single_layer_forward(X, layer_number)
#                 X = layer_output[-1] if self.add_recurrent_connections else layer_output[1]
#                 # do the update based on current layer `inpt`-`outpt`
#                 dW = self.hebbian_update(, self.W_s[layer_number])
#                 dW_r = self.antihebb()

            layer_outputs.append(layer_output)
            
        # single ouput readout
        if self.add_readout:
            X = self.W_out@X

        return [layer_outputs, X]

