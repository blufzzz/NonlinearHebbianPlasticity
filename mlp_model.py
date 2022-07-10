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
        
# breaking non-singularity of embedding
#         init_constant(self)
#         nn.init.xavier_normal_(self.W_out)

    def create_f(self, input_dim):
        if self.parametrized_f:
            return self.nonlinearity(input_dim, **self.f_kwargs)
        else:
            # default
            return self.f_kwargs['function']
    
    def hebbian_update(self, W, inp, out, learning_type='Oja', weight_decay=1e-1):
        
        '''
        inp - [d1,T], layer input 
        out - [d2,T], layer output
        W - [d2, d1]
        |--------|
        |--w(i)--|
        |--------|
        |--------|
        '''
        d1,T = inp.shape
        d2,T = out.shape
        device = W.device
        
        if learning_type=='Oja':
            # equation (4) for quadratic error
            # minimizes quadratic representation error $J(W) = ||X - W^Tf(WX)||_2$
            dW = out@(inp.T - out.T@W)/T # [d2,:]@([:,d1] - [:,d2]@[d2,d1]) 
            
        elif learning_type=='Hebb':
            dW = (out@inp.T)/T - weight_decay*W
            
        elif learning_type=='Criterion':
            # equation (3)
            I = torch.eye(d1, device=device) 
            dW = (out@inp.T)@(I - W.T@W)/T # [d2,:]@[:,d1]@([d1,d1] - [d1,d1]) 
        
        elif learning_type=='GHA':
            i_u, j_u = np.triu_indices(d2, k=1)
            L_out = out@out.T
            L_out[i_u, j_u] = 0
            dW = (out@inp.T - (L_out @ W))/T # [d2,:]@([:,d1] - [d2,d2]@[d2,d1])
            
        else:
            raise RuntimeError('Only ["BP", "Hebb", "Oja", "Criterion", "GHA"] rules are supported!')
                
        return dW
        
    def hebbian_learning_step(self, 
                              X_s, 
                              readout=None,
                              learning_type='Oja', 
                              learning_rate=1e-1, 
                              weight_decay=1e-1):
        
        '''
        X_s: [[d_1,], ..., [d_k,T]] - layer activations
        readout: [d,T] - ground-truth output
        '''
        
        inp = X_s[0] 
        # hebbian update for intermediate layers
        for i, W in enumerate(self.W_s, start=1):
            out = X_s[i]
            
            dW = self.hebbian_update(W.data, inp, out, learning_type=learning_type, weight_decay=weight_decay)
            W.data = W.data + learning_rate*dW
#             W.data = W.data / torch.norm(W.data, dim=1, keepdim=True)
            inp = out
            
        if self.add_readout:
            # delta-rule update for the readout layer
            delta = readout - X_s[-1]  # [1,T]
            dW_out = delta@inp.T
            self.W_out.data = self.W_out.data + learning_rate*dW_out
        
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
            
            X = self.f_s[i](W@X) # [d,T]
            
            if self.inplace_update:
#                     self.W_2 = self.W_2 + self.λ*dW2
#         # updated pass
#         Y2 = self.f_2(self.W_2@Y1)

            X_s.append(X)
            
        # single ouput readout
        if self.add_readout:
            X = self.W_out@X
            X_s.append(X)

        return X_s




class MLP_EncoderDecoder(nn.Module):

    def __init__(self,**kwargs):

        super(MLP_EncoderDecoder, self).__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        # initialize
        if self.set_seed:
            torch.manual_seed(self.seed)
            
    
    def forward(self, X):

        '''
        X - [d,T], input data
        '''

        encoder_output = self.encoder(X)
        Z = encoder_output[-1].T
        X_pred = self.decoder(Z)

        return encoder_output + [X_pred.T]

class MLP_NonlinearDecoder(nn.Module):
    
    '''
    Here serve the purpose to create the loss for the backpropagation
    '''
    
    def __init__(self,**kwargs):
        
        super(MLP_NonlinearDecoder, self).__init__()
        
        for k, v in kwargs.items():
            setattr(self, k, v)
            
        # initialize
        if self.set_seed:
            torch.manual_seed(self.seed)
            
        hidden_layers = []
        input_dim = self.input_dim
        for layer in range(self.hidden_layers_number):
            
            hidden_layers.append(nn.Linear(input_dim, self.hidden_dim, bias=self.bias))
            if self.add_bn:
                hidden_layers.append(nn.BatchNorm1d(self.hidden_dim, 
                                                    affine=False, 
                                                    track_running_stats=False))
            input_dim = self.hidden_dim
            
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim, bias=self.bias)
        
        # create parameter lists
    
    def forward(self,I):

        '''
        I - [d,T], input data
        '''

        X = I
        for layer in self.hidden_layers:
            X = self.default_nonlinearity(layer(X))
    
        X = self.output_layer(X)

        return X
