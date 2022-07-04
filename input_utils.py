import numpy as np
from scipy.signal import  convolve
from scipy.signal.windows import blackman, gaussian
from scipy.spatial.transform import Rotation as R

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from IPython.core.debugger import set_trace


def make_kurtosis_first_direction(n_samples, dim=2, ktype='pos', offset=None, replace_ratio=0.25):
    
    X = np.random.randn(dim, n_samples)
    Y = X[0]
    
    if ktype == 'pos':
        assert replace_ratio <= 0.5
        replace_number = int(n_samples*replace_ratio)
        rand_subsample = np.random.choice(np.arange(n_samples), size=replace_number, replace=False)
        Y[rand_subsample] = (np.random.random_sample(replace_number) - 0.5)*0.25
        
    elif ktype=='neg':
        if offset is None:
            offset = 1 + np.random.random_sample()
        rand_subsample = np.random.choice([True,False], size=n_samples, replace=True)
        Y[rand_subsample] += offset
        Y[~rand_subsample] -= offset
    
    else:
        raise RuntimeError('Wrong `stype`, only pos and neg are supported')
    
    return X.T, np.zeros((n_samples,))


    
def make_skewness_first_direction(n_samples, dim=2, stype='pos', k=None):
    
    X = np.random.randn(dim, n_samples)
    Y = X[0]
    
    if k is None:
        k = 1+np.random.random_sample()
    
    if stype == 'pos':
        Y[Y>0] = Y[Y>0]*k
    elif stype=='neg':
        Y[Y<0] = Y[Y<0]*k
    else:
        raise RuntimeError('Wrong `stype`, only pos and neg are supported')
    
    return X.T, np.zeros((n_samples,))



def make_bicluster(n_samples, distance=1, n_clusters=2, corr=0.99, angle=np.pi/4, random_state=42):
    
    dy = dx = np.sqrt(distance)
    means = np.array([[0,0], 
                      [-dx,dy]])
    
    cov = np.array([
                    [[1,corr],
                     [corr,1]],
                    [[1,corr],
                     [corr,1]]
                   ])
    
    
    N_0 = n_samples//2
    N_1 = n_samples - n_samples//2
    mv_0 = np.random.multivariate_normal(means[0], cov[0], size=N_0)
    mv_1 = np.random.multivariate_normal(means[1], cov[1], size=N_1)

    I = np.concatenate([mv_0, mv_1]).T
    theta = np.concatenate([np.zeros(N_0), np.ones(N_1)])
    
    # rotation angles
    U = np.array([[np.cos(angle), np.sin(angle)],
                  [-np.sin(angle), np.cos(angle)]])

    I = U@I
    
    return I.T, theta


def make_random_affine(n_samples, cov, W, noise=1e-1, random_state=42):
     
    n_pc = cov.shape[0]
    X = np.random.randn(n_samples, n_pc)
    X = X@cov
    
    _, v = np.linalg.eigh(cov)
    
    theta = X@v[-1]

    X = X@W
    
    noise_ = np.random.randn(*X.shape)
    noise_ /= np.linalg.norm(noise_, axis=0, keepdims=True)
    noise_ *= noise
    X += noise_
    
    return X, theta


def generate_random_tiling(N_I, T, receptive_field_dt):
    
    I = np.zeros((N_I,T))
    for neuron in range(N_I):
        if receptive_field_dt == 1:
            firing_time = neuron
            I[neuron, firing_time: firing_time+1] = 1
        else:
            receptive_field_sample = np.random.choice(np.arange(2, receptive_field_dt*2), size=1)[0]
            drf = receptive_field_sample//2
            firing_time = neuron + np.random.randint(-receptive_field_dt, receptive_field_dt, size=1)[0]
            firing_time = max(0, firing_time)
            I[neuron, max(0,firing_time-drf): min(T,firing_time+drf)] = 1
    return I


def whiten(X):

    T,d = X.shape
    C = X.T@X / (T-1) # [d,d]
    w,v = np.linalg.eig(C)
    w_argsort = np.argsort(w)[::-1]
    w = np.sqrt(w[w_argsort])
    v = v[:,w_argsort]
    X_ = X@v / w[None,:]
    
    return X_

class DataGenerator:
    
    def __init__(self, **kwargs):
        
        '''
        generator output should be (X, y) or (X, None) tuple!
        '''
        
        # defaults
        self.normalization = False
        self.normalize_output = False
        self.whitening = False
        self.pca_whitener = None
        self.use_outpt_color = False
        
        for k, v in kwargs.items():
            setattr(self, k, v)
        
    def __call__(self):
        
        color = None
        inpt, outpt = self.generator(**self.generator_kwargs)
        
        if self.use_outpt_color:
            color = MinMaxScaler((-1,1)).fit_transform(outpt[:,None]).flatten()

        T,d = inpt.shape
        assert T > d

        if self.scaler is not None:
            inpt = self.scaler.fit_transform(inpt)
            
        if self.whitening:
            if self.pca_whitener is None:
                self.pca_whitener = PCA(whiten=True, random_state=42)
                self.pca_whitener.fit_transform(inpt)
            else:
                inpt = pca_whitener.transform(inpt)
            # assert identity covariance
            assert np.isclose((inpt.T @ inpt) / inpt.shape[0], np.eye(d), atol=1e-1).all()

        inpt = inpt.T # to obtrain [d,N]

        if self.unsupervised:
            # use encoding-decoding paradigm
            outpt = inpt.copy()
        else:
            if self.normalize_output:
                # use regression
                outpt -= np.mean(outpt) # outpt - [N,]
                outpt = MinMaxScaler((-1,1)).fit_transform(outpt[:,None]).flatten() # outpt - [N,]
                
            outpt = outpt[None,:] # outpt - [1,N]
            
        return inpt, outpt, color

def create_data_tiling(**kwargs):
    
    N_I = kwargs['N_I']
    N_indep = kwargs['N_indep']
    receptive_field_dt = kwargs['receptive_field_dt']
    T = kwargs['T']
    sygma = kwargs['sygma']
    redundancy = kwargs['redundancy']
    kernel_conv = kwargs['kernel_conv']
    add_noise = kwargs['add_noise']
    normalize = kwargs['normalize']
    
    I_s = []
    for _ in range(redundancy):
        I = generate_random_tiling(N_I, T, receptive_field_dt)
        if add_noise:
            I += np.random.randn(*I.shape).clip(-sygma,sygma)
        if kernel_conv > 0:
            conv_filter = np.blackman(kernel_conv)[None,:]
            I = convolve(I, conv_filter, mode='same')
        I_s.append(I)
        
    I = np.concatenate(I_s, axis=0)

    if normalize:
        I = (I - I.mean(1)[:,None]) / I.std(1)[:,None]

    I = I[np.argsort(I.argmax(1))]
    
    T = N_indep
    theta = np.linspace(0,1,num=T)
    
    return I, theta