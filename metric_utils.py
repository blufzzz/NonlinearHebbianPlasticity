import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import r2_score
import torch

def get_pred_index(metric, index):
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
    Y_pred - [d,T]
    Y_true - [d,T]
    '''
    assert Y_pred.shape[0] == Y_true.shape[0] == 1
    return r2_score(to_numpy(Y_true.squeeze(0)), to_numpy(Y_pred.squeeze(0)))
    
    
def l2_loss(Y_pred, Y_true):
    '''
    Y_pred - [d,T]
    Y_true - [d,T]
    '''
    return torch.pow(torch.norm(Y_pred - Y_true, dim=0), 2).mean()





def strain(Y_pred, Y_true):
    
    '''
    Y_pred - [d,T]
    Y_true - [d,T]
    '''
    
    return ((Y_pred.T@Y_pred - Y_true.T@Y_true)**2).mean()



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