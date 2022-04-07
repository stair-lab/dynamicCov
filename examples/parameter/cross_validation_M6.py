import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
import snscov.synth_data as synth_data
from  snscov.model_reg import Model6
from snscov.kernel import Matern52Kernel
from snscov.evaluate import *
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

import argparse

parser = argparse.ArgumentParser(description='Process some integers. ')
parser.add_argument('--waveform',     type=str, default='mixing',    help='select waveform')

args = parser.parse_args()
N = 200
K = 4
T = 50
D = 20

fullcov = np.zeros((K, D, D), dtype=np.float32)
v = np.zeros((4,20))
v[0,0:5] = np.array([1, -1, 1, -1, -1])
v[0,9:10] = np.array([1])

v[1,5:9] = np.array([1,-1, -1, 1])

v[2,10:13] = np.array([1, 1, -1])

v[3,13:16] = np.array([1, -1,  1])
v[3,17:20] = np.array([-1, -1, -1])


v /= np.sqrt(np.sum(v ** 2, axis=1))[:,np.newaxis]
C = np.einsum('ki,kj->kij', v, v)


if args.waveform == 'sine':
    alphas=synth_data.synthesize_sines(K,T)
    amp=4
    tol=0.001

elif args.waveform == 'square':
    alphas=synth_data.synthesize_switches2(K,T)
    amp=1.5
    tol=0.001
elif args.waveform == 'mixing':
    alphas=synth_data.synthesize_weights(K,T)
    #amp=2*np.max(alphas)
    tol=0.001
    amp=4.
amp=2*np.max(alphas)
def compute_log_likelihood(data,mean,cov):
    """
    data: shape (N,T,D)
    mean: shape (T,D)
    cov: (T,D,D)
    """
    T = data.shape[1]
    D = data.shape[2]
    sum_logpdf = 0.
    for t in range(T):
        logpdf = multivariate_normal.logpdf(data[:,t,:],mean=mean[t],cov=cov[t]).sum()
        sum_logpdf += logpdf
    
    return sum_logpdf



def aic_v(data, mean, cov, v,alphas):
    lln = compute_log_likelihood(data, mean, cov)
    v_0 = np.where(np.abs(v) >= 1e-5)[0].size
    a_0 = np.where(np.abs(alphas) >= 1e-5)[0].size
    return -2*lln + 2*(v_0+a_0)


def bic_v(data, mean, cov, v,alphas):
    lln = compute_log_likelihood(data, mean, cov)
    v_0 = np.where(np.abs(v) >= 1e-5)[0].size
    a_0 = np.where(np.abs(alphas) >= 1e-5)[0].size
    N = data.shape[0]
    return -2*lln + (v_0+a_0)*np.log(N)

def compute_sdp_sparse_threshold(true_covar,test_covar, v,lam=0.1):
    v_0 = np.where(np.abs(v) >= 1e-5)[0].size
    return 0.5*np.sum((true_covar-test_covar)**2)+lam*v_0
#tuning parameter

length_scale = np.array([5,10,50,100])
a_coef = np.array([ 0.0001, 0.01, 0.1]) #[100]
b_coef = np.array([ 0.01, 0.1])
lr_rate = np.array([1e-4, 1e-5])
K_set = np.array([2,4,6])


np.random.seed(8)
inv_noise_level = 1e-2
sd = synth_data.synthesize_data_noiselevel(alphas, C, N, BETA_inv = inv_noise_level)
# partition k_fold cross validation data

k_fold = 5
shuffle_idx = np.random.permutation(N)
#randomly shuflle data
sd = sd[shuffle_idx]
d_pfold = int(N/k_fold)



max_iter = 2000



oos_mse_m= np.zeros((len(lr_rate),len(b_coef),len(a_coef),len(length_scale), len(K_set)))

oos_lln_m = np.zeros(oos_mse_m.shape)
aic_v_n = np.zeros(oos_mse_m.shape)
bic_v_n = np.zeros(oos_mse_m.shape)
for lr in range(len(lr_rate)):
    for s in range(len(b_coef)):
        for g in range(len(a_coef)):
            for ls in range(len(length_scale)):
                for kk in range(len(K_set)):
                    kernel = Matern52Kernel(T, 2, length_scale[ls])
                    kernel = linalg.solve(kernel, np.eye(T))
                    acc_oos_mse = 0.
                    acc_oos_lln = 0.
                    acc_aic_v = 0.
                    acc_bic_v = 0.
                    if K_set[kk]<K:
                        a_amp = np.argsort(np.sum(alphas,axis=1))[::-1]
                        idx = a_amp[0:K_set[kk]]
                        feed_a = alphas[idx]
                        feed_v = v[idx]
                    else:
                        pad = K_set[kk]-K
                        pad_a = np.zeros((pad, T))
                        pad_v = np.zeros((pad, D))
                        feed_a = np.vstack((alphas,pad_a))
                        feed_v = np.vstack((v,pad_v))
                    temp_metric = np.zeros(2)
                    for k in range(k_fold):
                        val_index = np.arange(k*d_pfold,min((k+1)*d_pfold, N), 1)
                        train_index = np.hstack((np.arange(0,k*d_pfold,1),np.arange(min((k+1)*d_pfold, N),N,1)))
                        train_data = sd[train_index]
                        val_data = sd[val_index]

                        tkst_test = Model6(K_set[kk], T, D, lambda1 = a_coef[g], lambda2= b_coef[s], la_rate = lr_rate[lr],
                            ld_rate=lr_rate[lr], kernel = kernel, group=None,
                            max_iter = max_iter, a_method = 'temporal_kernel', d_method = 'sparse')


                        tkst_test.fit(train_data, true_a=feed_a, true_d=feed_v, evaluate=True)

                        tkst_re_dict, tkst_re_alphas = tkst_test.dict, tkst_test.alphas

                        est_C  = np.einsum('ki,kj->kij', tkst_re_dict,  tkst_re_dict)
                        est_covar  = np.einsum('ti,ijk->tjk',tkst_re_alphas ,  est_C)


                        S_n = np.einsum('ntj,ntk->tjk',val_data,val_data)/val_data.shape[0]
                        oos_mse = compute_sdp_sparse_threshold(est_covar, S_n, tkst_re_dict,lam=.1)
                        acc_oos_mse += oos_mse
                        est_covar += inv_noise_level*np.eye(D) 
                        oos_lln = compute_log_likelihood(val_data,np.zeros((T,D)), est_covar)

                        is_aic_v = aic_v(train_data, np.zeros((T,D)), est_covar,tkst_re_dict, tkst_re_alphas)
                        is_bic_v = bic_v(train_data, np.zeros((T,D)), est_covar,tkst_re_dict, tkst_re_alphas)

                        acc_oos_lln += oos_lln
                        acc_aic_v += is_aic_v

                        acc_bic_v += is_bic_v                    
                        print('fold {}|result: bcoef {}, acoef {},ls {},lr {},K {}'.format(k, b_coef[s], a_coef[g],length_scale[ls], lr_rate[lr],K_set[kk])) 
                  
                    oos_mse_m[lr,s,g,ls,kk] = acc_oos_mse
                    oos_lln_m[lr,s,g,ls] = acc_oos_lln
                    aic_v_n[lr,s,g,ls,kk] = acc_aic_v/k_fold
                    bic_v_n[lr,s,g,ls,kk] = acc_bic_v/k_fold
                    print('Accumulated OOS MSE', acc_oos_mse)
                    print('--------------------------------')   



print('minimum oos mse',np.min(oos_mse_m))
loc = np.where(oos_mse_m==np.min(oos_mse_m))
print('parameters',loc)
print('minimum oos bic_v',np.min(bic_v_n))
loc = np.where(bic_v_n==np.min(bic_v_n))
print('parameters',loc)
print('minimum oos aic_v',np.min(aic_v_n))
loc = np.where(aic_v_n==np.min(aic_v_n))
print('parameters',loc)
print('maximum oos lln',np.max(oos_lln_m))
loc = np.where(oos_lln_m==np.max(oos_lln_m))
print('parameters',loc)

np.savez('../../parameters/reg_{}_sparse_ms_fix.npz'.format(args.waveform),oos_lln=oos_lln_m,oos_mse=oos_mse_m, aic_v=aic_v_n,bic_v=bic_v_n)
