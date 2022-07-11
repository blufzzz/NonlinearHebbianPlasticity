import numpy as np
import torch
from torch import autograd
from torch import nn
from IPython.core.debugger import set_trace
from metric_utils import to_numpy


def initialize_nonlinearities(network, state_dict):
    for k,v in network.named_parameters():
        if 'f_s' in k:
            print(f'f_s from {k} loaded')
            v.data = state_dict[k].to(v.data.device)


def init_constant(self, C=1.):
    for p in self.parameters():
        nn.init.constant_(p, C)
    
def sigmoid(x):
    return (1./(1+torch.exp(-x))) - 0.5

def init_weights(self):
    for p in self.parameters():
        nn.init.xavier_normal_(p)

class dJ_criterion:
    
    '''
    example
    J_θ_s = [f_theta(1) for _ in range(3)]
    J_θ = J_criterion(J_θ_s)
    J_θ(X) -> Y
    '''
    
    def __init__(self, J_s, reduce='mean'):
        
        # list of differentiable scalar functions: J:R^T -> R
        self.J_s = J_s 
        self.dim = len(J_s)
        self.reduce=reduce
        
    def __call__(self, X):
        
        # X - [d,T]
        d,T = X.shape
        output = []
        if self.reduce=='mean':
            mult = T
        elif self.reduce=='sum':
            mult = 1
        else:
            raise RuntimeError('Wrong self.reduce type!')
            
        assert len(X) == self.dim
        for i in range(d):
            X_i = X[i].clone().detach().requires_grad_(True)
            y = self.J_s[i](X_i)*mult
            
            output.append(torch.autograd.grad(y, 
                                              X_i, 
                                              retain_graph=True, 
                                              allow_unused=True, 
                                              create_graph=True)[0])
            
        return torch.stack(output, dim=0)


class gained_function(nn.Module):
    
    def __init__(self, input_dim, function, bias=True, **kwargs):
        
        super().__init__()
        
        # nonlinearity
        self.function = function
        self.bias = bias
        self.input_dim = input_dim
        self.requires_grad = kwargs['requires_grad']
        
        self.theta = nn.Parameter(torch.zeros(self.input_dim, 1), requires_grad=self.requires_grad)
        if self.bias:
            self.theta_bias = nn.Parameter(torch.zeros(self.input_dim, 1), requires_grad=self.requires_grad)
        else:
            self.theta_bias = 0.
        
        init_weights(self)

    def forward(self, x):
        
        '''
        x - [d,T]
        '''
        
        x = self.function(x*self.theta + self.theta_bias)
        
        return x
    

class universal_approximator(nn.Module):
    
    def __init__(self, input_dim, hidden_dim=10, **kwargs):
        
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.requires_grad = kwargs['requires_grad']
        
        self.theta1 = nn.Parameter(torch.zeros(self.hidden_dim, self.input_dim), requires_grad=self.requires_grad)
        self.bias1 = nn.Parameter(torch.zeros(self.hidden_dim, self.input_dim), requires_grad=self.requires_grad)
        self.theta2 = nn.Parameter(torch.zeros(1, self.hidden_dim), requires_grad=self.requires_grad)
        
        init_weights(self)
        
    def forward(self, x):
        
        '''
        x - [d, T]
        '''
        
        x = torch.sigmoid(x.unsqueeze(0) * self.theta1.unsqueeze(-1) + self.bias1.unsqueeze(-1))
        x = torch.einsum('mh,hdt->mdt', self.theta2, x).squeeze(0)
        
        return x
    
