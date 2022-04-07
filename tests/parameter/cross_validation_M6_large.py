import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
import snscov.synth_data as synth_data
from  snscov.model_reg_large import Model6_large
from snscov.kernel import Matern52Kernel
from snscov.evaluate import *
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from multiprocessing import Pool, cpu_count
import argparse
import time

parser = argparse.ArgumentParser(description='Process some integers. ')
parser.add_argument('--waveform',     type=str, default='mixing',    help='select waveform')
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--n_trials', type=int, default=20, help='number of trials')
parser.add_argument('--N', type=int, default=10, help='number of subject')
parser.add_argument('--K', type=int, default=10, help='number of components')
parser.add_argument('--D', type=int, default=100, help='data dimension')
parser.add_argument('--T', type=int, default=100, help='number of time points')
parser.add_argument('--density', type=float, default=0.4, help='matrix sparsity')
parser.add_argument('--n_block', type=int,   default=4, help='number of blocks')
parser.add_argument('--kernel_length', type=int, default=5, help='length scale of the Matern five-half kernel')
parser.add_argument('--filename', type=str, default='test', help='output filename')
args = parser.parse_args()

N = args.N
K = args.K
T = args.T
D = args.D
density = args.density
n_block = args.n_block
np.random.seed(200)
M = synth_data.sparse_ortho_component(D,D,density,n_block)
select_id = np.random.choice(D, K)
v = M[:,select_id].T
C = np.einsum('ki,kj->kij', v, v)

np.random.seed(1246)
alphas = synth_data.generate_latent_splines(T, K, 1).T

np.set_printoptions(precision=2)


nonzero = np.sum(v!=0,axis=1)
print(nonzero)
print('maximum number of non-zero component:',np.max(nonzero))
amp=np.ceil(1.2*np.max(alphas))+2.



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
#    print(sum_logpdf)
    return sum_logpdf

def compute_rank_regress(data, V, A):
    pass
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
def compute_sdp_sparse(true_covar,test_covar, v,lam=0.01):
    v_norm = np.sqrt(np.sum(v**2,axis=1))
    return 0.5*np.sum((true_covar-test_covar)**2)+lam*np.sum(v_norm)
def compute_sdp_sparse_threshold(true_covar,test_covar, v,lam=0.1):
    v_0 = np.where(np.abs(v) >= 1e-5)[0].size
    return 0.5*np.sum((true_covar-test_covar)**2)+lam*v_0
#tuning parameter


length_scale = np.array([5,10,100])
a_coef = np.array([ 0.0001, 0.01, 0.1]) #[100]
b_coef = np.array([ 0.01, 0.1])
lr_rate = np.array([1e-2, 1e-1,5])



np.random.seed(8)
inv_noise_level = 1e-1
sd = synth_data.synthesize_data_noiselevel(alphas, C, N, BETA_inv = inv_noise_level)
# partition k_fold cross validation data

k_fold = 5
shuffle_idx = np.random.permutation(N)
#randomly shuflle data
sd = sd[shuffle_idx]
d_pfold = int(N/k_fold)
 

max_iter = 2000



K_set = np.array([5,10,20])



oos_mse_m= np.zeros((len(lr_rate),len(b_coef),len(a_coef),len(length_scale),len(K_set)))
aic_v_n = np.zeros(oos_mse_m.shape)
bic_v_n = np.zeros(oos_mse_m.shape)

eva_metric = np.zeros((len(lr_rate),len(b_coef),len(a_coef),len(length_scale),len(K_set),2))
for lr in range(len(lr_rate)):
    for s in range(len(b_coef)):
        for g in range(len(a_coef)):
            for ls in range(len(length_scale)):
                for kk in range(len(K_set)):
                    kernel = Matern52Kernel(T, 2, length_scale[ls])
                    kernel = linalg.solve(kernel, np.eye(T))
                    acc_oos_mse = 0.
                    acc_aic_v = 0.
                    acc_bic_v = 0.
                    temp_metric = np.zeros(2)
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
                    st = time.time()
                    for k in range(k_fold):
                        val_index = np.arange(k*d_pfold,min((k+1)*d_pfold, N), 1)
                        train_index = np.hstack((np.arange(0,k*d_pfold,1),np.arange(min((k+1)*d_pfold, N),N,1)))
                        train_data = sd[train_index]
                        val_data = sd[val_index]
                    
                        tkst_test = Model6_large(K_set[kk], T, D, lambda1 = a_coef[g], lambda2= b_coef[s], la_rate = lr_rate[lr],
                            ld_rate=lr_rate[lr], kernel = kernel, group=None, tol=1e-3,
                            max_iter = max_iter, a_method = 'temporal_kernel', d_method = 'sparse')
                        tkst_test.fit(train_data, true_a=feed_a, true_d=feed_v)

                        tkst_re_dict, tkst_re_alphas = tkst_test.best_dictionary, tkst_test.best_alphas
                        
                        est_C  = np.einsum('ki,kj->kij', tkst_re_dict,  tkst_re_dict)
                        est_covar  = np.einsum('ti,ijk->tjk',tkst_re_alphas ,  est_C)
                        
                        metric = np.array([avg_LERM_cov((alphas).T, v, est_covar ),avg_LERM_cov((alphas).T, v, est_covar )])
                                                       # large_scale_signed_permutation((alphas).T, v, tkst_re_alphas, tkst_re_dict)])
                        
                        temp_metric += metric
                        S_n = np.einsum('ntj,ntk->tjk',val_data,val_data)/val_data.shape[0]
                        oos_mse = compute_sdp_sparse_threshold(est_covar, S_n, tkst_re_dict,lam=.1)
                        acc_oos_mse += oos_mse
                        est_covar += inv_noise_level*np.eye(D) 

                        is_aic_v = aic_v(train_data, np.zeros((T,D)), est_covar,tkst_re_dict, tkst_re_alphas)
                        is_bic_v = bic_v(train_data, np.zeros((T,D)), est_covar,tkst_re_dict, tkst_re_alphas)

                        acc_aic_v += is_aic_v
                        acc_bic_v += is_bic_v   
                        print('fold {}|result: b_coef {}, a_coef {},ls {},lr {} K {}'.format(k,b_coef[s], a_coef[g],length_scale[ls], lr_rate[lr],K_set[kk]), metric) 
                    et = time.time()
                    eva_metric[lr, s, g,ls,kk,:] = temp_metric/k_fold

                    oos_mse_m[lr,s,g,ls,kk] = acc_oos_mse
                    aic_v_n[lr,s,g,ls,kk] = acc_aic_v/k_fold
                    bic_v_n[lr,s,g,ls,kk] = acc_bic_v/k_fold
                    print('Accumulated OOS MSE', acc_oos_mse)
                    print('Execution time', et-st)
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

print('minimum oos bic_v',np.min(bic_v_n))
loc = np.where(bic_v_n==np.min(bic_v_n))
print('parameters',loc)
print('minimum oos aic_v',np.min(aic_v_n))
loc = np.where(aic_v_n==np.min(aic_v_n))
print('parameters',loc)


np.savez('../../parameters/reg_{}_sparse_ms2.npz'.format(args.filename),oos_mse=oos_mse_m, metric=eva_metric, aic_v=aic_v_n,bic_v=bic_v_n)