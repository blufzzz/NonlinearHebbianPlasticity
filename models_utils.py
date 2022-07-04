import numpy as np
import torch
from torch import autograd
from torch import nn
from IPython.core.debugger import set_trace
from metric_utils import to_numpy

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
    
    def __init__(self, input_dim, function):
        
        super().__init__()
        
        # nonlinearity
        self.function = function
        self.input_dim = input_dim
        
        self.theta = nn.Parameter(torch.zeros(self.input_dim, requires_grad=True))
        
        init_weights(self)

    def forward(self, x):
        
        '''
        x - [d,T]
        '''
        
        x = self.function(x*self.theta.unsqueeze(1))
        
        return x
    

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
    
    
# Code below from: https://github.com/HanchenXiong/deep-tsne-embedding
def Hbeta(D, beta):
    '''
    D - list of distances from i to j'th
    beta - 1/2(σ^2)
    '''
    P = np.exp(-D * beta) 
    sumP = np.sum(P)
    H = np.log(sumP) + beta * np.sum(np.multiply(D, P)) / sumP
    P = P / sumP
    return H, P


def x2p(X, u=15, tol=1e-4, print_iter=2500, max_tries=50, verbose=0):
    
    
    '''
    X - [n,d]: T samples of d-dimensional data
    '''
    
    # Initialize some variables
    n = X.shape[0]                     # number of instances
    P = np.zeros((n, n))               # empty probability matrix
    beta = np.ones(n)                  # empty precision vector
    logU = np.log(u)                   # log of perplexity (= entropy)
    
    # compute pairwise distances
    if verbose > 0: print('Computing pairwise distances...')
    sum_X = np.sum(np.square(X), axis=1) # ||x_i||^2
    
    # create euclidean distance matix [n,n]
    D = sum_X + sum_X[:,None] - 2*X.dot(X.T)  # ||x_i - x_j||^2

    # run over all datapoints
    if verbose > 0: print('Computing P-values...')
    for i in range(n):
        
        if verbose > 1 and print_iter and i % print_iter == 0:
            print('Computed P-values {} of {} datapoints...'.format(i, n))
        
        # Set minimum and maximum values for precision
        betamin = float('-inf')
        betamax = float('+inf')
        
        # Compute the Gaussian kernel and entropy for the current precision
        indices = np.concatenate((np.arange(0, i), np.arange(i + 1, n))) # all except i-th
        Di = D[i, indices] # distances from i to other points
        H, thisP = Hbeta(Di, beta[i])
        
        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while abs(Hdiff) > tol and tries < max_tries:
            
            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i]
                if np.isinf(betamax):
                    beta[i] *= 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i]
                if np.isinf(betamin):
                    beta[i] /= 2
                else:
                    beta[i] = (beta[i] + betamin) / 2
            
            # Recompute the values
            H, thisP = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1
        
        # Set the final row of P
        P[i, indices] = thisP
        
    if verbose > 0: 
        print('Mean value of sigma: {}'.format(np.mean(np.sqrt(1 / beta))))
        print('Minimum value of sigma: {}'.format(np.min(np.sqrt(1 / beta))))
        print('Maximum value of sigma: {}'.format(np.max(np.sqrt(1 / beta))))
    
    return P, beta

def compute_joint_probabilities(samples, d=2, perplexity=30, tol=1e-5, verbose=0):
    
    # Initialize some variables
    n = samples.shape[0]
    
    # Precompute joint probabilities for all batches
    if verbose > 0: print('Precomputing P-values...')
      
    P, beta = x2p(samples, perplexity, tol, verbose=verbose) # compute affinities using fixed perplexity
    P[np.isnan(P)] = 0                                       # make sure we don't have NaN's
    P = (P + P.T)                                            # make symmetric
    P = P / P.sum()                                          # obtain joint probabilities
    P = np.maximum(P, np.finfo(P.dtype).eps)

    return P

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 50 epochs"""
    lr = 0.1 * (0.1 ** (epoch // 150))
    lr = max(lr, 1e-3)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def tsne_loss(P, activations):
    n = activations.size(0)
    device=activations.device
    alpha = 1
    eps = 1e-15
    sum_act = torch.sum(torch.pow(activations, 2), 1)
    Q = sum_act + sum_act.view([-1, 1]) - 2 * torch.matmul(activations, torch.transpose(activations, 0, 1))
    Q = Q / alpha
    Q = torch.pow(1 + Q, -(alpha + 1) / 2)
    Q = Q * autograd.Variable(1 - torch.eye(n), requires_grad=False).to(device)
    Q = Q / torch.sum(Q)
    C = torch.log((P + eps) / (Q + eps))
    C = torch.sum(P * C)
    return C



def tsne_criterion(out_pred, out, **kwargs):
    
    '''
    out_pred - [d2,T], embedding
    out - [d1,T], original data
    d2 < d1
    '''
    device = out_pred.device
    
    perplexity = kwargs['perplexity']
    
    if 'P' in kwargs:
        # we have pre-calculated P (used for validation)
        P_tensor = kwargs['P']
    else:
        out_np = to_numpy(out).T
        P = compute_joint_probabilities(out_np,
                                        perplexity=perplexity,
                                        verbose=0)
        P_tensor = autograd.Variable(torch.Tensor(P), requires_grad=False).to(device)
        
    loss = tsne_loss(P_tensor, out_pred.T)

    return loss