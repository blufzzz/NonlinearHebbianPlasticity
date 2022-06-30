import torch
from torch import nn
from IPython.core.debugger import set_trace

class J_criterion:
    
    '''
    example
    J_θ_s = [f_theta(1) for _ in range(3)]
    J_θ = J_criterion(J_θ_s)
    J_θ(X) -> Y
    '''
    
    def __init__(self, J_s):
        
        # list of differentiable scalar functions: J:R^T -> R
        self.J_s = J_s 
        self.dim = len(J_s)
        
    def __call__(self, X):
        # X - [d,T]
        output = []
#         X.requires_grad_(True)
        assert len(X) == self.dim
        for i,X_i in enumerate(X):
            X_i = X_i.clone().detach().requires_grad_(True)
            y = self.J_s[i](X_i)
            output.append(torch.autograd.grad(y, 
                                              X_i, 
                                              retain_graph=True, 
                                              allow_unused=True, 
                                              create_graph=True)[0])
            
        return torch.stack(output, dim=0)


def init_weights(self):
    for p in self.parameters():
        nn.init.xavier_normal_(p)

class universal_approximator(nn.Module):
    
    def __init__(self, input_dim, hidden_dim=100):
        
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.theta1 = nn.Parameter(torch.zeros(self.hidden_dim, self.input_dim, requires_grad=True))
        self.bias1 = nn.Parameter(torch.zeros(self.hidden_dim, self.input_dim, requires_grad=True))
        self.theta2 = nn.Parameter(torch.zeros(1, self.hidden_dim, requires_grad=True))
        
        init_weights(self)
        
    def forward(self, x):
        
        '''
        x - [d, T]
        '''
        
        x = torch.sigmoid(x.unsqueeze(0) * self.theta1.unsqueeze(-1) + self.bias1.unsqueeze(-1))
        x = torch.einsum('mh,hdt->mdt', self.theta2, x).squeeze(0)
        
        return x