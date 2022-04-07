import numpy as np
import ssm
from scipy import linalg
from scipy import stats
from sklearn.decomposition import SparsePCA, DictionaryLearning


###############################
#                             #
# different competing methods #
#                             #
###############################


#################################
#                               #
# M4 sparse dictionary learning #
#                               #
#################################
def sparse_dict_learning(X, n_com, alpha=1.0):
    N = X.shape[0]
    T = X.shape[1]
    covar = np.zeros((T,X.shape[2],X.shape[2]))
    X = X.reshape((-1,X.shape[2]))

    dl_model = DictionaryLearning(n_components=n_com, alpha=alpha)
    y = dl_model.fit_transform(X)

    y = y.reshape((N,T,-1))
    v_com = dl_model.components_

    for t in range(T):
        yy = np.einsum('ki,ij->kj', y[:,t,:], v_com)
        C = np.einsum('ki,kj->ij',yy, yy)/N
        covar[t,:,:] = C 
    return covar

def sparse_PCA(X, n_com, alpha=1.0):
    N = X.shape[0]
    T = X.shape[1]
    covar = np.zeros((T, X.shape[2], X.shape[2]))
    X = X.reshape((-1, X.shape[2]))

    sppca_model = SparsePCA(n_components=n_com, alpha=alpha)
    y = sppca_model.fit_transform(X)

    y = y.reshape((N,T,-1))
    v_com = sppca_model.components_

    for t in range(T):
        yy = np.einsum('ki,ij->kj', y[:,t,:], v_com)
        C = np.einsum('ki,kj->ij',yy, yy)/N
        covar[t,:,:] = C 
    #print(covar[t])
    return covar

def sliding_window_idx(t_len, w_len):
    """
    compute the sliding window index

    Parameters
    ----------
    t_len: time series length, scalar
    w_len: sliding-window length, scalar

    Returns
    ----------
    w_idx: sliding window index, list
    """
    assert(w_len <= t_len)
    stride = t_len - w_len + 1
    w_idx = []
    s_idx = 0 
    for i in range(stride):
        x = np.arange(s_idx,s_idx+w_len,1)
        w_idx.append(x)
        s_idx += 1
    return w_idx


def sliding_window_covaraince(X, w_len):
    """
    compute sliding window covariance

    Parameters
    ----------
    X: time series data, array-like, shape (N, T, D)
    w-len: sliding window length , integer

    Returns
    ---------
    s_cov: sliding-window sample covariance, shape(T, D, D)
    """
    assert (w_len <= X.shape[1])
    #compute sample covariance
    s_t = np.einsum('nti,ntj->ntij',X,X)
    w_idx = sliding_window_idx(X.shape[1],w_len)

    s_cov = np.array([np.mean(s_t[:,sub_i,:,:], axis=(0,1)) for sub_i in w_idx])
    return s_cov

#################################
#                               #
# M1 Slifing window PCA         #
#                               #
#################################

def SWPCA(X,k, w_len):
    """
    compute sliding window covariance + PCA

    Parameters
    ----------
    X: time series data, array-like, shape (N, T, D)
    w-len: sliding window length , integer

    Returns
    ---------
    s_cov: sliding-window sample covariance, shape(T, D, D)
    """
    s_cov = sliding_window_covaraince(X, w_len)
    for t in range(s_cov.shape[0]):
        w, vh = linalg.eigh(s_cov[t,:,:])
        
        w = w[::-1]
        
        vh = vh[:,::-1]
        new_cov = vh[:,0:k] @ np.diag(w[0:k]) @ vh[:, 0:k].T

        s_cov[t,:,:] = new_cov
    return s_cov



#################################
#                               #
# MS Spectral Initialization    #
#                               #
#################################
def spectral_initialization(X, k):
    """
    compute spectral initialization

    Parameters
    ----------
    X: time series data, array-like, shape (N, T, D)
    k: low-rank structure, integer

    Returns
    ---------
    s_cov: covariance, shape(T, D, D)
    """
    N = X.shape[0]
    T = X.shape[1]
    D = X.shape[2]

    S_N = np.einsum('nji,njk->ik', X, X) / N
    w, eigv = linalg.eigh(S_N)

    w = w[::-1] #shape(D) v
    eigv = eigv[:,::-1][:, 0:k] #shape(D, D)v
    S_T = np.einsum('ntk,ntj->tkj', X, X) / N #shape(T,D,D) v
    S_Tv = np.einsum('tkj,ji->tki', S_T, eigv) #shape(T,D,D) v
    vtS_Tv = np.einsum('ki,tkj->tij', eigv, S_Tv) # shape(T,D,D) v
    Ax = np.einsum('tii->ti', vtS_Tv) #v T*k*K
    A = np.array([np.diag(Ax[t,:]) for t in range(T)])
    cov = np.array([eigv @ A[t,:,:] @ eigv.T for t in range(T)])
    return cov, eigv, Ax


#################################
#                               #
# M2 Hidden Markov Model        #
#                               #
#################################


def HMM(X, k, method="sgd", n_iters=1000):
    """
    compute hidden Markov model

    Parameters
    ----------
    X: time series data, array-like, shape (N, T, D)
    k: number of discrete states, integer
    method: optimization method, choice [sgd, em]

    Returns
    ---------
    s_cov: covariance, shape(T, D, D)
    """    
    N = X.shape[0]
    T = X.shape[1]
    D = X.shape[2]

    
    hmm = ssm.HMM(k, D, observations="gaussian")
    if N==1:
        train_data = np.squeeze(X)
    else:
        train_data = [np.array(X[n, :, :]) for n in range(N)]
    
    hmm_lls = hmm.fit(train_data, method=method, num_iters = n_iters)
    states_cov = hmm.observations.Sigmas #shape(K,D,D)
    
    predicted_states = np.zeros((N,T), dtype=np.int)
    for i in range(N):
        predicted_states[i,:] = hmm.most_likely_states(X[i,:,:])
    #mode_predicted_states =  np.squeeze(stats.mode(predicted_states, axis = 0)[0])
    cov = np.zeros((T,D,D))
    for t in range(T):
        #cov[t,:,:]=states_cov[mode_predicted_states[t]]
        stat, count = np.unique(predicted_states[:,t], return_counts=True)
        for i in range(len(stat)):

            cov[t,:,:] += states_cov[stat[i]] * count[i]
        cov[t,:,:] /= N
    return cov, states_cov


#################################
#                               #
# M3 Autoregressiv              #
# Hidden Markov Model           #
#                               #
#################################

def ARHMM(X, k, method="sgd", n_iters=1000):
    """
    compute autoregressive observation hidden Markov model

    Parameters
    ----------
    X: time series data, array-like, shape (N, T, D)
    k: number of discrete states, integer
    method: optimization method, choice [sgd, em]

    Returns
    ---------
    s_cov: covariance, shape(T, D, D)
    """        
    N = X.shape[0]
    T = X.shape[1]
    D = X.shape[2]
    if N==1:
        train_data = np.squeeze(X)
    else:
        train_data = [np.array(X[n, :, :]) for n in range(N)]
    
    arhmm = ssm.HMM(k, D, observations="ar")

    hmm_lls = arhmm.fit(train_data, method=method, num_iters = n_iters)
    
    states_cov = arhmm.observations.Sigmas #shape(K,D,D)
    Aks = arhmm.params[2][0]
    n_cov = np.zeros((N,T,D,D))
    for n in range(N):
        id_states = arhmm.most_likely_states(X[n,:,:])
        for t in range(T):
            n_cov[n,t,:,:] = states_cov[id_states[t],:,:]
            if t !=0:
                n_cov[n,t,:,:] += Aks[id_states[t],:,:] @ n_cov[n,t-1,:,:] @ Aks[id_states[t],:,:].T
    cov = np.mean(n_cov, axis=0)
    return cov, states_cov


if __name__ == "__main__":
    import synth_data
    N = 10
    T = 50
    D = 10
    K = 3
    v = np.array([[1, -1, 1, -1, -1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, -1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]], dtype=np.float32)
    v /= np.sqrt(np.sum(v ** 2, axis=1))[:,np.newaxis]
    alphas=synth_data.synthesize_weights(K,T)
    C = np.einsum('ki,kj->kij', v, v)
    sd = synth_data.synthesize_data(alphas, C, N)
    n_cov, stat_cov = ARHMM(sd, 3, method="sgd", n_iters=100)
    synth_data.plot_components(stat_cov, fname='test.png')