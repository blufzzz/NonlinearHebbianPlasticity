import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import r2_score
import torch
from IPython.core.debugger import set_trace


def get_k_th_moment(k, **k_th_moment_kw):
    def wrapper(X):
        return k_th_moment(X, k, **k_th_moment_kw)
    return wrapper

def variance(X, μ=0):
    
    '''
    X - [,T] or [1,T]
    '''
    
    ndim = X.ndim
    dim = 0 if ndim == 1 else 1 
    
    if μ is None:
        μ = X.mean()
        
    K = (X - μ)**2 # [,T]
    
    # expectation
    return K.mean()



def k_th_moment(X, k, μ=0, σ=None):
    
    '''
    X - [,T] or [1,T]
    '''
    ndim = X.ndim
    dim = 0 if ndim == 1 else 1 
#     set_trace()
    if μ is None:
        μ = X.mean()
    if σ is None: 
        σ = X.std()
        
    K = (X - μ)**k / (σ**k) # [,T]
    
    # expectation
    return K.mean()


def cosine_sim(x,y):
    return x@y / (np.linalg.norm(x)*np.linalg.norm(y))

def get_index(metric, index):
    def wrapper(Y_pred, Y_true):
        return metric(Y_pred[index], Y_true)
    return wrapper

def numpy_metric(metric):
    def wrapper(*args):
        args = [to_numpy(arg) for arg in args]
        return metric(*args)
    return wrapper

def to_numpy(X):
    return X.detach().cpu().numpy()

def r2_score_torch(Y_pred, Y_true):
    '''
    Y_pred - [T,d]
    Y_true - [T,d]
    '''
    assert Y_pred.shape[1] == Y_true.shape[1] == 1
    return r2_score(to_numpy(Y_true.squeeze(1)), to_numpy(Y_pred.squeeze(1)))
    
    
def l2_loss(Y_pred, Y_true):
    '''
    Y_pred - [T,d]
    Y_true - [T,d]
    '''
    return torch.pow(torch.norm(Y_pred - Y_true, dim=1), 2).mean()


def strain(X, Z):
    
    '''
    X - [T,d1]
    Z - [T,d2]
    d2 < d1
    '''
    
    return ((Z@Z.T - X@X.T)**2).mean()



def coranking_matrix(high_data, low_data):
    
    # from https://github.com/samueljackson92/coranking/blob/master/coranking/_coranking.py
    """Generate a co-ranking matrix from two data frames of high and low
    dimensional data.
    :param high_data: DataFrame containing the higher dimensional data.
    :param low_data: DataFrame containing the lower dimensional data.
    :returns: the co-ranking matrix of the two data sets.
    """
    n, m = high_data.shape
    high_distance = squareform(pdist(high_data))
    low_distance = squareform(pdist(low_data))

    high_ranking = high_distance.argsort(axis=1).argsort(axis=1)
    low_ranking = low_distance.argsort(axis=1).argsort(axis=1)

    Q, xedges, yedges = np.histogram2d(high_ranking.flatten(),
                                       low_ranking.flatten(),
                                       bins=n)

    Q = Q[1:, 1:]  # remove rankings which correspond to themselves
    return Q


def calculate_Q_metrics(X, Z):
    '''
    Calculates co-ranking matrix based metrics Q_loc and Q_glob
    X: np.ndarray [N,d1] - data
    Z: np.ndarray [N,d2] - embedding (d2 <= d)
    '''
    
    d1 = X.shape[1]
    d2 = Z.shape[1]
    
    assert X.shape[0] == Z.shape[0]
    assert d2 <= d1
    
    Q = coranking_matrix(X, Z)
    
    m = X.shape[0]
    UL_cumulative = 0 
    Q_k = []
    LCMC_k = [] 
    for k in range(0, Q.shape[0]):
        r = Q[k:k+1,:k+1].sum()
        c = Q[:k,k:k+1].sum()
        UL_cumulative += (r+c)
        Qnk = UL_cumulative/((k+1)*m) 
        Q_k.append(Qnk)
        LCMC_k.append(Qnk - ((k+1)/(m-1)))
    
    argmax_k = np.argmax(LCMC_k)
    k_max = np.arange(1.,m)[argmax_k]
    Q_loc = (1./k_max)*np.sum(Q_k[:argmax_k+1])
    Q_glob = (1./(m-k_max))*np.sum(Q_k[argmax_k+1:])
    
    return Q_loc, Q_glob