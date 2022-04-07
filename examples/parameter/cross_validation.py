import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
import snscov.synth_data as synth_data
from  snscov.model import snscov_model
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

def aic(data, mean, cov):
    lln = compute_log_likelihood(data, mean, cov)
    N = data.shape[0]
    param_n = np.where(np.abs(cov)>=1e-5)[0].size
    return -2*lln + 2*param_n

def bic(data, mean, cov):
    lln = compute_log_likelihood(data, mean, cov)
    N = data.shape[0]
    param_n = np.where(np.abs(cov)>=1e-5)[0].size
    return -2*lln + param_n*np.log(N)

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

def compute_rank_regress(data, v, alphas, test_v, lam):
    N = data.shape[0]
    matrix = np.einsum('ji,jk->ik',v,alphas).T
    temr1 = np.sum((data - matrix[np.newaxis,:,:])**2)/(2*N)
    v_0 = np.where(np.abs(v) >= 1e-5)[0].size
    return term1 + lam*v_0
def compute_sdp_sparse(true_covar,test_covar, v,lam=0.01):
    v_norm = np.sqrt(np.sum(v**2,axis=1))
    return 0.5*np.sum((true_covar-test_covar)**2)+lam*np.sum(v_norm)
def compute_sdp_sparse_threshold(true_covar,test_covar, v,lam=0.1):
    v_0 = np.where(np.abs(v) >= 1e-5)[0].size
    return 0.5*np.sum((true_covar-test_covar)**2)+lam*v_0
#tuning parameter
gamma = np.array([0.0001,0.01,0.1]) #[100]
length_scale = np.array([5,10,50,100,200,1000])
sparse_level = np.array([6])
lr_rate = np.array([1e-4])



np.random.seed(8)
inv_noise_level = 1e-2
sd = synth_data.synthesize_data_noiselevel(alphas, C, N, BETA_inv = inv_noise_level)
# partition k_fold cross validation data

k_fold = 5
shuffle_idx = np.random.permutation(N)
#randomly shuflle data
sd = sd[shuffle_idx]
d_pfold = int(N/k_fold)



max_iter = 50

aic_n = np.zeros((len(lr_rate),len(sparse_level),len(gamma),len(length_scale)))
oos_lln_m = np.zeros((len(lr_rate),len(sparse_level),len(gamma),len(length_scale)))
eva_metric = np.zeros((len(lr_rate),len(sparse_level),len(gamma),len(length_scale),2))
for lr in range(len(lr_rate)):
    for s in range(len(sparse_level)):
        for g in range(len(gamma)):
            for ls in range(len(length_scale)):
                    kernel = Matern52Kernel(T, 2, length_scale[ls])
                    kernel = linalg.solve(kernel, np.eye(T))
                    acc_oos_lln = 0.
        
                    temp_metric = np.zeros(2)
                    for k in range(k_fold):
                        val_index = np.arange(k*d_pfold,min((k+1)*d_pfold, N), 1)
                        train_index = np.hstack((np.arange(0,k*d_pfold,1),np.arange(min((k+1)*d_pfold, N),N,1)))
                        train_data = sd[train_index]
                        val_data = sd[val_index]
                    
                        tkst_test = snscov_model(K, T, D, la_rate = lr_rate[lr], amp=amp,tol=0.001,
                                                ld_rate=lr_rate[lr]/T, kernel = kernel, smooth=gamma[g],k_sparse=sparse_level[s],
                                                max_iter = max_iter, a_method = 'temporal_kernel', d_method = 'sparse')

                        tkst_test.fit(train_data, true_a=alphas, true_d=v, evaluate=False)

                        tkst_re_dict, tkst_re_alphas = tkst_test.best_dictionary, tkst_test.best_alphas
                        
                        est_C  = np.einsum('ki,kj->kij', tkst_re_dict,  tkst_re_dict)
                        est_covar  = np.einsum('ti,ijk->tjk',tkst_re_alphas ,  est_C)

                        metric = np.array([avg_LERM_cov((alphas).T, v, est_covar ),
                                                        dual_permutation((alphas).T, v, tkst_re_alphas, tkst_re_dict)])
                        temp_metric += metric
                        est_covar += inv_noise_level*np.eye(D) 
                        oos_lln = compute_log_likelihood(val_data,np.zeros((T,D)), est_covar)
               
                        acc_oos_lln += oos_lln
                      
                        print('fold {}|result: sparse {}, gamma {},ls {},lr {}'.format(k,sparse_level[s], gamma[g],length_scale[ls], lr_rate[lr]), metric) 
                    eva_metric[lr, s, g,ls,:] = temp_metric/k_fold
                    oos_lln_m[lr,s,g,ls] = acc_oos_lln
               
                    print('Accumulated OOS LLN', acc_oos_lln)
                    
                    print('--------------------------------')   
print('minimum avglog metric', np.min(eva_metric[:, :, :,:,0]))
loc = np.where(eva_metric[:, :, :,:,0]==np.min(eva_metric[:, :, :,:,0]))
print('parameter loc',loc)
print('minimum dist metric', np.min(eva_metric[:, :, :,:,1]))
loc = np.where(eva_metric[:, :, :,:,1]==np.min(eva_metric[:, :, :,:,1]))
print('parameter loc',loc)
print('maximum oos lln',np.max(oos_lln_m))
loc = np.where(oos_lln_m==np.max(oos_lln_m))
print('parameters',loc)

np.savez('../../parameters/{}_ms.npz'.format(args.waveform),oos_lln=oos_lln_m, metric=eva_metric)


gamma = np.array([0.01,0.1]) #[100]
length_scale = np.array([5,10,50,100,200])
sparse_level = np.array([4,6,8,10,12,14])

lr_rate = np.array([1e-4])
K_set = np.array([2,4,6])


oos_mse_m= np.zeros((len(lr_rate),len(sparse_level),len(gamma),len(length_scale), len(K_set)))

oos_lln_m = np.zeros(oos_mse_m.shape)
aic_n = np.zeros(oos_mse_m.shape)
bic_n = np.zeros(oos_mse_m.shape)
aic_v_n = np.zeros(oos_mse_m.shape)
bic_v_n = np.zeros(oos_mse_m.shape)
eva_metric = np.zeros((len(lr_rate),len(sparse_level),len(gamma),len(length_scale),len(K_set),2))
for lr in range(len(lr_rate)):
    for s in range(len(sparse_level)):
        for g in range(len(gamma)):
            for ls in range(len(length_scale)):
                for kk in range(len(K_set)):
                    kernel = Matern52Kernel(T, 2, length_scale[ls])
                    kernel = linalg.solve(kernel, np.eye(T))
                    acc_oos_mse = 0.
                    acc_aic = 0.
                    acc_bic = 0.
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
                    
                        tkst_test = snscov_model(K_set[kk], T, D, la_rate = lr_rate[lr], amp=amp,tol=0.001,
                                                ld_rate=lr_rate[lr]/T, kernel = kernel, smooth=gamma[g],k_sparse=sparse_level[s],
                                                max_iter = max_iter, a_method = 'temporal_kernel', d_method = 'sparse')

                        tkst_test.fit(train_data, true_a=feed_a, true_d=feed_v, evaluate=True)

                        tkst_re_dict, tkst_re_alphas = tkst_test.best_dictionary, tkst_test.best_alphas

                        est_C  = np.einsum('ki,kj->kij', tkst_re_dict,  tkst_re_dict)
                        est_covar  = np.einsum('ti,ijk->tjk',tkst_re_alphas ,  est_C)


                        metric = np.array([avg_LERM_cov((alphas).T, v, est_covar ),
                                                        dual_permutation(feed_a.T, feed_v, tkst_re_alphas, tkst_re_dict)])
                        temp_metric += metric
                        S_n = np.einsum('ntj,ntk->tjk',val_data,val_data)/val_data.shape[0]
                        oos_mse = compute_sdp_sparse_threshold(est_covar, S_n, tkst_re_dict,lam=.1)
                        acc_oos_mse += oos_mse
                        est_covar += inv_noise_level*np.eye(D) 
                        oos_lln = compute_log_likelihood(val_data,np.zeros((T,D)), est_covar)
                        is_aic = aic(train_data, np.zeros((T,D)), est_covar)
                        is_bic = bic(train_data, np.zeros((T,D)), est_covar)
                        is_aic_v = aic_v(train_data, np.zeros((T,D)), est_covar,tkst_re_dict, tkst_re_alphas)
                        is_bic_v = bic_v(train_data, np.zeros((T,D)), est_covar,tkst_re_dict, tkst_re_alphas)
                        acc_aic += is_aic

                        acc_bic += is_bic
                        acc_oos_lln += oos_lln
                        acc_aic_v += is_aic_v

                        acc_bic_v += is_bic_v                    
                        print('fold {}|result: sparse {}, gamma {},ls {},lr {},K{}'.format(k,sparse_level[s], gamma[g],length_scale[ls], lr_rate[lr],K_set[kk]), metric) 
                    eva_metric[lr, s, g,ls,kk,:] = temp_metric/k_fold
                    oos_mse_m[lr,s,g,ls,kk] = acc_oos_mse
                    aic_n[lr,s,g,ls,kk] = acc_aic/k_fold
                    bic_n[lr,s,g,ls,kk] = acc_bic/k_fold
                    oos_lln_m[lr,s,g,ls] = acc_oos_lln
                    aic_v_n[lr,s,g,ls,kk] = acc_aic_v/k_fold
                    bic_v_n[lr,s,g,ls,kk] = acc_bic_v/k_fold
                    print('Accumulated OOS MSE', acc_oos_mse)
                    print('--------------------------------')   
print('minimum avglog metric', np.min(eva_metric[:, :, :,:,:,0]))
loc = np.where(eva_metric[:, :, :,:,:,0]==np.min(eva_metric[:, :, :,:,:,0]))
print('parameter loc',loc)
print('minimum dist metric', np.min(eva_metric[:, :, :,:,:,1]))
loc = np.where(eva_metric[:, :, :,:,:,1]==np.min(eva_metric[:, :, :,:,:,1]))
print('parameter loc',loc)
print('minimum oos mse',np.min(oos_mse_m))
loc = np.where(oos_mse_m==np.min(oos_mse_m))
print('parameters',loc)
print('minimum oos aic',np.min(aic_n))
loc = np.where(aic_n==np.min(aic_n))
print('parameters',loc)
print('minimum oos bic',np.min(bic_n))
loc = np.where(bic_n==np.min(bic_n))
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

np.savez('../../parameters/{}_sparse_ms_fix.npz'.format(args.waveform),oos_lln=oos_lln_m,oos_mse=oos_mse_m, metric=eva_metric, aic=aic_n, bic=bic_n, aic_v=aic_v_n,bic_v=bic_v_n)
