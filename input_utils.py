import numpy as np
from scipy.signal import  convolve
from scipy.signal.windows import blackman, gaussian
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from IPython.core.debugger import set_trace


def make_kurtosis_bicluster(n_samples, n_clusters=2, corr=0.99):
    
 
    means = np.array([[0,0], 
                      [-1,1]])
    
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
    randind = np.arange(n_samples)
    np.random.shuffle(randind)
    I = I[:,randind]
    theta = theta[randind]
    
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


def create_data(**kwargs):
    '''
    generator output should be (X, y) or (X, None) tuple!
    '''
    gen = kwargs['generator']
    gen_params = kwargs['generator_kwargs']
    unsupervised = kwargs['unsupervised']
    scaler = kwargs['scaler']
    
    inpt, outpt = gen(**gen_params)
    inpt -= inpt.mean(0, keepdims=True)
    if scaler is not None:
        inpt = scaler.fit_transform(inpt)
    inpt = inpt.T # to obtrain [d,N]
    
    if unsupervised:
        # use encoding-decoding paradigm
        outpt = inpt.copy()
    else:
        # use regression
        outpt -= np.mean(outpt) # outpt - [N,]
        outpt = MinMaxScaler((-1,1)).fit_transform(outpt[:,None]).flatten() # outpt - [N,]
        outpt = outpt[None,:] # outpt - [1,N]
        
    return inpt, outpt

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