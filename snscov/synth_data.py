import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from time import time
from snscov.hemodynamic_filter import Poisson_function
from scipy.stats import special_ortho_group
from  scipy.sparse import random as sparse_random
import numpy.linalg as linalg
from scipy import interpolate
from matplotlib.gridspec import GridSpec


import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
def generate_latent_splines(t, r, amp, n_knots = 6):
    """
    source code: https://github.com/modestbayes/LFGP_NeurIPS/blob/master/notebooks/simulation.ipynb
    Generate r random splines with t time points each.
    """
    
    splines = np.zeros((t, r))
    for i in range(r):
        x = np.arange(0, n_knots, 1) + np.random.uniform(-0.5, 0.5, n_knots)
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        y = np.random.uniform(0, amp, n_knots)
        s = interpolate.InterpolatedUnivariateSpline(x, y, k=3)
        xnew = np.linspace(0, 1, t)
        offset = min([0, np.min(s(xnew))])
        splines[:, i] = s(xnew)-offset
    return splines

def sparse_ortho_component(D,K,density,n_block):
    assert(n_block <= K)
    assert(K<=D)
    d_size = int(K/n_block)
    M = np.zeros(D**2).reshape((D,D))
    for i in range(n_block):
        if i != n_block-1:
            sparse_m = sparse_random(d_size, d_size, density=density).A
        else:
            if (K % n_block) != 0:
                dd_size = K - i*d_size
            else:
                dd_size = d_size
            sparse_m = sparse_random(dd_size, dd_size, density=density).A
      
        ortho_m,_ = linalg.qr(sparse_m)
       
        st = i*d_size
        if i != n_block-1:
            ed = min([(i+1)*d_size, K])
        else:
            ed = K
        print(st,ed)
        M[st:ed, st:ed] = ortho_m
    per_col = np.random.permutation(K)
    per_row = np.random.permutation(D)
    
    #M = M[:,per_col]
    #M = M[per_row,:]
    return(M)
def synthesize_switch(T, ith, K):
    sub_T = np.int(np.ceil(T/K))
    interval = np.zeros(T)+0.2
    interval[sub_T*ith:sub_T*(ith+1)] = 1.0
    return interval

def synthesize_switches(K, T):
    alphas = np.empty((K, T))
    for k in range(K):
        alphas[k] = synthesize_switch(T, k, K)
    return alphas
def synthesize_weight(T, ith):
    if ith == 0: 
        alpha_k = np.ones(T) * 0.8  
    elif ith == 1:
        alpha_k = np.arange(0, 1, step=1./T)
    elif ith == 2:
        t = np.linspace(0, 2*np.pi, num=T)
        alpha_k = 0.5*(np.cos(t-np.pi/8) + 1.2)
    elif ith == 3:
        t = np.linspace(0, 2*np.pi, num=T)
        alpha_k = 0.6*(np.sin(t) + 1.2)
    #elif ith == 4:
    #    sub_T = np.int(np.ceil(T/4))
    #    alpha_k = np.zeros(T)
    #    alpha_k[sub_T*2:sub_T*3] = 2.0
    elif ith == 4:
        t = np.linspace(0, 2*np.pi, num=T)
        alpha_k = 0.5*np.sin(t + ( np.pi *0. / 3.))  + 0.5
    elif ith == 5:
        t = np.linspace(0, 2*np.pi, num=T)
        alpha_k = (np.sin(2*t + (np.pi / 2)) + 1)/2 + .5
    #elif ith == 6:
    #    t = np.linspace(0, 2*np.pi, num=T)
    #    alpha_k = 1.2*np.cos(t + (np.pi/2)) + 1.2
    #elif ith == 7:
    #    t = np.linspace(0, 2*np.pi, num=T)
    #    alpha_k = np.cos(t) + 1.
    elif ith == 6:
        alpha_k = synthesize_switch(T, 0, 4)
    elif ith == 7:
        alpha_k = synthesize_switch(T, 1, 4)
    elif ith == 8:
        alpha_k = synthesize_switch(T, 2, 4)
    elif ith == 9:
        alpha_k = synthesize_switch(T, 3, 4)
    return alpha_k

def synthesize_weights(K, T):
    alphas = np.empty((K, T))
    for k in range(K):
        alphas[k] = synthesize_weight(T, k)
    return alphas

def synthesize_periodic_wave(T, max_p,ith):
    alpha_k = np.zeros(T)
    t = np.arange(T)
    if ith == 0:
        f = 2/max_p
        alpha_k = 0.5*np.sin(2*np.pi*f*t+ np.pi/4) + 0.6
    elif ith == 1:
        f = 4/max_p
        alpha_k = 0.5*np.cos(2*np.pi*f*t) + 0.7
    elif ith == 2:
        f = 1/max_p
        alpha_k = 1.2*np.cos(2*np.pi*f*t) + 1.3
    elif ith == 3:
        f = 8/max_p
        alpha_k = 0.3*np.cos(2*np.pi*f*t+np.pi) + 1.0    
    print(np.sum(alpha_k))
    return alpha_k

def synthesize_periodic_waves(T, max_p,K): 
    alphas = np.empty((K, T))
    for k in range(K):
        alphas[k] = synthesize_periodic_wave(T,max_p, k)
    return alphas

def synthesize_sine(T, ith):
    alpha_k = np.zeros(T)
    if ith == 0:
        t = np.linspace(0, 2*np.pi, num=T)
        alpha_k = 0.5*np.sin(t + ( np.pi *0. / 3.))  + 0.5
    elif ith == 1:
        t = np.linspace(0, 2*np.pi, num=T)
        alpha_k = (np.sin(2*t + (np.pi / 2)) + 1)/2 + .5
    elif ith == 2:
        t = np.linspace(0, 2*np.pi, num=T)
        alpha_k = 1.2*np.cos(t + (np.pi/2)) + 1.2
    elif ith == 3:
        t = np.linspace(0, 2*np.pi, num=T)
        alpha_k = np.cos(t) + 1. 
    return alpha_k

def synthesize_sines(K, T):
    alphas = np.empty((K, T))
    for k in range(K):
        alphas[k] = synthesize_sine(T, k)
    return alphas



def synthesize_switch2(T, ith, K):
    sub_T = np.int(np.ceil(T/K))
    interval = np.zeros(T)
    interval[sub_T*ith:sub_T*(ith+1)] = 1.0
    return interval

def synthesize_switches2(K, T):
    alphas = np.empty((K, T))
    for k in range(K):
        alphas[k] = synthesize_switch2(T, k, K)
    return alphas

def synthesize_group_sparse_component(K, D, group_par, prob=0.2):
    
    num = len(group_par)
    C = np.empty((K,D,D))
    v = np.zeros((K,D))
    for k in range(K):
        idx = np.mod(k, num)
        
        pos = group_par[idx][0]
        
        G = len(group_par[idx])
        
        vec = np.zeros((D,1))
        values = np.random.choice([0, 1], size=(G,1), p=[1-prob, prob])*np.random.normal(0, 1, (G,1))
        
        vec[pos:pos+G] = values
        vec /= np.sqrt(np.sum(vec ** 2)+1e-10)
        v[k,:] = vec[:,0]
        C[k, :, :] = vec * vec.T
    return C , v       

def synthesize_components(K, D):
    C = np.empty((K, D, D))
    for k in range(K):
        values = np.random.choice([0, 1], size=(D,1), p=[7./10, 3./10])*np.random.normal(0, 1, (D,1))
        values /= np.sqrt(np.sum(values ** 2))
        C[k, :, :] = values*values.T
    return C

def synthesize_no_overlap(K):
    components = np.array([[1, -1, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, -1, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, -1, 1]])
    C = np.empty((K, 10, 10))
    for k in range(K):
        C[k, :, :] = np.outer(components[k, :],components[k, :])
    return C

def synthesize_some_overlap(K):
    assert (K==3) or (K==6)
    components = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                           [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                           [0, 0, 0, 0, 0, 1, 1, 0, 0, 0]])
    C = np.empty((K, 10, 10))
    for k in range(K):
        C[k, :, :] = np.outer(components[k, :], components[k, :])
    return C

def plot(alphas, C, title_append, fname):
    K, T = alphas.shape
    fig = plt.figure(figsize=(10, 5))
    width_ratio = [((K+1)//2)*1.5]
    width_ratio += [1 for i in range(int((K+1)/2))]
    gs = GridSpec(nrows=2, ncols=1+np.int(np.ceil(K/2)), width_ratios=width_ratio)
    ax0 =  fig.add_subplot(gs[:, 0])
    #flatui = ["#9b59b6", "#3498db", "#e74c3c", "#34495e", "#2ecc71"]
    marker_set=['p','v','s','*','D','X','H','^','1','2','3','4','8','+','d'] 
    color_set = ['blue','red','green','black', 'yellow', 'purple', 'hotpink', 'sienna', 'orange', 'grey']
    for k in range(K):
        ax0.plot(alphas[k, :],marker=marker_set[k], color=color_set[k],label=str(k))
    ax0.legend(loc='upper right')
    ax0.set_xlabel('t')
    ax0.set_ylabel('amplitude')
    #ax0.title(wtitle_append, fontsize=14)
    

    K, D, _D = C.shape
    vmin = np.min(C)
    vmax = np.max(C)

    for k in range(K):
        if k == 0:
            ax = fig.add_subplot(gs[0, 1])
        else:
            ax = fig.add_subplot(gs[k%2, (k//2)+1])
        pos = ax.imshow(C[k], cmap='jet')
        ax.set_axis_off()
        ax.set_title("component {}".format(str(k)))
        pos.set_clim(vmin, vmax)
    
    fig.subplots_adjust(right=0.82, wspace=0.1, hspace=0.1)
    cbar_ax = fig.add_axes([0.86, 0.2, 0.03, 0.6])
    fig.colorbar(pos, cax=cbar_ax)
    fig.suptitle(title_append)
    if fname is not None:
        plt.savefig(fname,  bbox_inches='tight')


def plot_weights(alphas, fname=None, title_append=""):
    K, T = alphas.shape
    #flatui = ["#9b59b6", "#3498db", "#e74c3c", "#34495e", "#2ecc71"]
    marker_set=['p','v','s','*','D','X','H','^','1','2','3','4','8','+','d'] 
    color_set = ['blue','red','green','black', 'yellow', 'purple', 'hotpink', 'sienna', 'orange', 'grey']
    for k in range(K):
        plt.plot(alphas[k, :],marker=marker_set[k], color=color_set[k],label=str(k))
    plt.legend(loc='upper right', prop={'size': 14})
    plt.xlabel('t', fontsize=14)
    plt.ylabel('amplitude', fontsize=14)
    plt.title(title_append, fontsize=14)
    if fname is not None:
        plt.savefig(fname,  bbox_inches = 'tight', pad_inches = 0)
    plt.close()

def plot_weights_with_box(alphas, box, fname=None, title_append=""):
    K, T = alphas.shape
    max_am =  np.int(np.max(alphas))
    if max_am == 0:
        max_am = 1
    cm = ["k", "r","b","y", 'green']
    for i in range(box.shape[1]):
        color = cm[i]
        timestep =np.zeros(T)
        act = np.where(box[:,i]>0)
        timestep[act] = max_am
        plt.plot( timestep,'--', color=color)

    for k in range(K):
        plt.plot(alphas[k, :], label=str(k) + 'th component')
    plt.legend(loc='lower right')
    plt.xlabel('t')
    plt.ylabel('amplitude')
    plt.title(title_append)
    if fname is not None:
        plt.savefig(fname)

    plt.close()

def hemodynamic_weights(alphas, filter_type = "poisson", lam = 5, order = 10):
    K, T = alphas.shape
    if filter_type == "poisson":
        h = Poisson_function(lam, order)
    for k in range(K):
        a = np.convolve(h**2, alphas[k,:], mode='same')
        alphas[k,:] = a
    return alphas

def plot_hemodynamic_weights(alphas, fname=None, title_append="", filter_type = "poisson", lam = 5, order = 10):
    K, T = alphas.shape
    if filter_type == "poisson":
        h = Poisson_function(lam, order)
    plt.figure()
    for k in range(K):
        a = np.convolve(h**2, alphas[k,:], mode='same')

        plt.plot(a, marker='*', label=str(k) + 'th component')
    plt.legend(loc='lower right')
    #plt.ylim([0,2.])
    plt.xlabel('t')
    plt.ylabel('amplitude')
    plt.title("Hemodynamic response of time-series component weights for "+title_append)
    if fname is not None:
        plt.savefig(fname)
    #plt.show()
    plt.close()

def plot_weightsshade(alphas, var, fname=None, title_append=""):
    K, T = alphas.shape
    plt.figure()
    for k in range(K):
        plt.plot(alphas[k, :], label=str(k) + 'th component')
        upper, lower = alphas[k, :] + [2 * var[k, :], -2 * var[k, :]]
        plt.fill_between(np.arange(alphas.shape[1]), upper, lower, alpha=.1)
    plt.legend(loc='lower right')
    plt.title("Time-series component weights for "+title_append)
    if fname is not None:
        plt.savefig(fname)
    plt.show()
    plt.close()

def synthesize_data(alphas, C, N):
    K1, T = alphas.shape
    K2, D, _D = C.shape

    K = K1

    BETA_inv = 1e-2
    I = np.eye(D)
    covars = np.zeros((N, T, D, D))
    syn_data = np.empty((N, T, D))

    for n in range(N):
        for t in range(T):
            covars[n, t, :, :] += BETA_inv*I
            for k in range(K):
                covars[n, t, :, :] += alphas[k, t]*C[k]
            xnt = np.random.multivariate_normal(np.zeros(D), covars[n, t, :, :])
            syn_data[n, t, :] = xnt
    return syn_data

def synthesize_hemodynamic_data(alphas, C, N, filter_type = "poisson", lam = 5, order = 10):
    K1, T = alphas.shape
    K2, D, _D = C.shape

    K = K1

    BETA_inv = 1e-2 #noise
    I = np.eye(D)
    covars = np.zeros((N, T, D, D))
    syn_data = np.empty((N, T, D))
    if filter_type == "poisson":
        h = Poisson_function(lam, order)
    for n in range(N):
        for t in range(T):
            covars[n, t, :, :] += BETA_inv*I
            for k in range(K):
                covars[n, t, :, :] += alphas[k, t]*C[k]
            xnt = np.random.multivariate_normal(np.zeros(D), covars[n, t, :, :])
            syn_data[n, t, :] = xnt
    return syn_data


def synthesize_data_noiselevel(alphas, C, N, BETA_inv = 0.):
    K1, T = alphas.shape
    K2, D, _D = C.shape

    K = K1

    I = np.eye(D)
    covars = np.zeros((N, T, D, D))
    syn_data = np.empty((N, T, D))

    for n in range(N):
        for t in range(T):
            covars[n, t, :, :] += BETA_inv*I
            for k in range(K):
                covars[n, t, :, :] += alphas[k, t]*C[k]
            xnt = np.random.multivariate_normal(np.zeros(D), covars[n, t, :, :])
            syn_data[n, t, :] = xnt
    return syn_data

    
def plot_components(C, fname=None, title_append=""):
    K, D, _D = C.shape
    vmin = np.min(C)
    vmax = np.max(C)
    if K <= 2:
        f, axes = plt.subplots(np.int(np.ceil(K/2)), K, sharey="all")#, figsize=(3, 4))
    else:
        f, axes = plt.subplots(2, np.int(np.ceil(K/2)), sharey="all")#, figsize=(3, 4))
    
    for k in range(K):
        if np.int(np.ceil(K/2))==1:

            pos = axes[np.mod(k,2)].imshow(C[k], cmap='jet')
            axes[np.int(k/2)].set_title(str(k))
 
        
        else:    
            pos = axes[np.mod(k,2), np.int(k/2)].imshow(C[k], cmap='jet')
     
            axes[np.mod(k,2), np.int(k/2)].set_title(str(k))



        pos.set_clim(vmin, vmax)
    
    f.subplots_adjust(right=0.8, wspace=0.2, hspace=0.3)
    cbar_ax = f.add_axes([0.85, 0.2, 0.03, 0.7])
    f.colorbar(pos, cax=cbar_ax)
    f.suptitle(title_append, fontsize=14)
    if fname is not None:
        plt.savefig(fname,  bbox_inches='tight')
    #plt.show()
    plt.close()

if __name__ == "__main__":
    pass