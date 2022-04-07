import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
import snscov.synth_data as synth_data
from  snscov.model_large import snscov_model
from snscov.model_reg_large import Model6_large
from snscov.kernel import Matern52Kernel
from snscov.evaluate import *
from snscov.methods import *

import time

import argparse

parser = argparse.ArgumentParser(description='Process some integers. ')
parser.add_argument('--num_subjects', type=int, default=10,         help='number of test subjects')
parser.add_argument('--num_trial',    type=int, default=10,        help='number of trials')
parser.add_argument('--waveform',     type=str, default='weight',    help='select waveform')
parser.add_argument('--title',        type=str, default='compare_large_scale', help='experiment name')
parser.add_argument('--noise_level',  type=int, default=2,       help='noise level'  )
parser.add_argument('--K', type=int, default=10, help='number of components')
parser.add_argument('--D', type=int, default=100, help='data dimension')
parser.add_argument('--T', type=int, default=100, help='number of time points')
parser.add_argument('--density', type=float, default=0.4, help='matrix sparsity')
parser.add_argument('--n_block', type=int,   default=4, help='number of blocks')
parser.add_argument('--kernel_length', type=int, default=5, help='length scale of the Matern five-half kernel')


args = parser.parse_args()
np.set_printoptions(precision=2)

N = args.num_subjects
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
#print(C.shape)
#for i in range(C.shape[0]):
#    plt.subplot(6,2,i+1)
#    plt.imshow(C[i], cmap='jet')
#plt.show()
np.random.seed(1246)
alphas = synth_data.generate_latent_splines(100, K, 1).T
#plt.plot(alphas.T)
#plt.show()
#print(np.max(alphas))

sd = synth_data.synthesize_data_noiselevel(alphas, C, N, BETA_inv = 1/args.noise_level)


kernel = Matern52Kernel(T, 2, args.kernel_length)
np.set_printoptions(precision=2)

kernel = linalg.solve(kernel, np.eye(T))

nonzero = np.sum(v!=0,axis=1)
print('maximum number of non-zero component:',np.max(nonzero))
a_coef = np.array([0.001]) #[100]
b_coef = np.array([30])
lr_rate = np.array([1e-2])
amp=np.ceil(1.2*np.max(alphas))+1.
print('max amplitude', amp)

W = [10,20,40]
test_T = T - max(W) + 1
states = [4,8,12,16]

sine_metric = np.zeros((len(W)+7,args.num_trial))
time_counter = np.zeros(sine_metric.shape)
for round in range (args.num_trial):
    sd = synth_data.synthesize_data_noiselevel(alphas, C, N, 1./args.noise_level)
 
    """
    for idx, w in enumerate(W):
        st = time.time()
        cov = sliding_window_covaraince(sd, w)
        end = time.time()
        time_counter[idx, round] = end -st
        sine_metric[idx, round] = avg_LERM_cov(alphas[:, 0:test_T].T, v, cov[0:test_T,:,:])
        print("Round:", round, " SW w_length:", w, "%4f"%avg_LERM_cov(alphas[:, 0:test_T].T, v, cov[0:test_T,:,:]), "ext time:%4f"%(end-st))
    """
    
    #run M1
    for idx, w in enumerate(W):
        st = time.time()
        cov = SWPCA(sd, K, w)
        end = time.time()
        time_counter[idx, round] = end - st
        sine_metric[idx, round] = avg_LERM_cov(alphas[:, 0:test_T].T, v, cov[0:test_T,:,:])
        print("Round:", round,  " SWPCA w_length:", w, "%4f"%avg_LERM_cov(alphas[:, 0:test_T].T, v, cov[0:test_T,:,:]), "ext time:%4f"%(end-st))

    #run M2    
    # denotes the number of states
    for idx, s in enumerate([10]):
        st = time.time()
        cov, states_cov = HMM(sd, s, method="em", n_iters=300)
        end = time.time()
        time_counter[idx+len(W), round] = end - st
        sine_metric[idx+len(W), round] = avg_LERM_cov(alphas[:, 0:test_T].T, v, cov[0:test_T,:,:]) 
        print("Round:", round, " HMM n_states:", s, "%4f"%avg_LERM_cov(alphas[:, 0:test_T].T, v, cov[0:test_T,:,:]), "ext time:%4f"%(end-st))
    
    #run M3
    for idx, s in enumerate([30]):
        st = time.time()
        cov, states_cov = ARHMM(sd, s, method="em", n_iters=300)
        end = time.time()
        time_counter[idx+len(W)+1, round] = end - st
        sine_metric[idx+len(W)+1, round] = avg_LERM_cov(alphas[:, 0:test_T].T, v, cov[0:test_T,:,:])
        print("Round:", round, " ARHMM n_states:", s, "%4f"%avg_LERM_cov(alphas[:, 0:test_T].T, v, cov[0:test_T,:,:]), "ext time:%4f"%(end-st))            
   
    #run M4
    st = time.time()
    cov = sparse_dict_learning(sd, K, alpha=1.0)
    end = time.time()
    time_counter[len(W)+2, round] = end-st
    sine_metric[len(W)+2, round] = avg_LERM_cov(alphas[:, 0:test_T].T , v, cov[0:test_T,:,:])

    print("Round:", round, " Sparse dictionary learning:", "%4f"%avg_LERM_cov(alphas[:, 0:test_T].T , v, cov[0:test_T,:,:]), "ext time:%4f"%(end-st)) 
    
    #run MS
    st = time.time()
    cov, _, _ = spectral_initialization(sd, K)
    end = time.time()
    time_counter[len(W)+3, round] = end - st
    sine_metric[len(W)+3, round] = avg_LERM_cov(alphas[:, 0:test_T].T, v, cov[0:test_T,:,:])  
    print("Spectral Initialization",  "%4f"%avg_LERM_cov(alphas[:, 0:test_T].T, v, cov[0:test_T,:,:]), "ext time:%4f"%(end-st))

    #run M**

    tkst_test = snscov_model(K, T, D, amp=amp, la_rate = 1e-5, ld_rate= 1e-5/T, kernel = kernel, smooth=a_coef[0], k_sparse=b_coef[0],
                                            max_iter = 300, a_method = 'temporal_kernel', d_method = 'sparse')
    st = time.time()
    tkst_test.fit(sd, true_a=alphas, true_d=v, evaluate=True)
    end = time.time()
    tkst_re_dict, tkst_re_alphas = (tkst_test.best_dictionary, tkst_test.best_alphas)
    est_C = np.einsum('ki,kj->kij', tkst_re_dict, tkst_re_dict)
    test_covar = np.einsum('tk,kij->tij', tkst_re_alphas, est_C)
    time_counter[len(W)+4, round] = end -st
    sine_metric[len(W)+4, round] = avg_LERM_cov(alphas[:, 0:test_T].T , v, test_covar[0:test_T,:,:])
    
    print("Round:", round, " Proposed Method:", "%4f"%avg_LERM_cov(alphas[:, 0:test_T].T , v, test_covar[0:test_T,:,:]), "ext time:%4f"%(end-st)) 

    #run MR
    
    tkst_test = snscov_model(K, T, D, amp=amp, la_rate = 1e-5, ld_rate= 1e-5/T, kernel = kernel, smooth=a_coef[0], k_sparse=b_coef[0],ini_method="random",
                                            max_iter = 300, a_method = 'temporal_kernel', d_method = 'sparse')
    st = time.time()
    tkst_test.fit(sd, true_a=alphas, true_d=v, evaluate=True)
    end = time.time()
    tkst_re_dict, tkst_re_alphas = (tkst_test.best_dictionary, tkst_test.best_alphas)
    est_C = np.einsum('ki,kj->kij', tkst_re_dict, tkst_re_dict)
    test_covar = np.einsum('tk,kij->tij', tkst_re_alphas, est_C)
    time_counter[len(W)+5, round] = end -st
    sine_metric[len(W)+5, round] = avg_LERM_cov(alphas[:, 0:test_T].T , v, test_covar[0:test_T,:,:])
    
    print("Round:", round, " Random initialization:", "%4f"%avg_LERM_cov(alphas[:, 0:test_T].T , v, test_covar[0:test_T,:,:]), "ext time:%4f"%(end-st)) 
    
    #run M6

    tkst_test = Model6_large(K, T, D, lambda1 = 0.0001, lambda2= 0.1, la_rate = 1e-4,
                        ld_rate=1e-4, kernel = kernel, group=None,
                        max_iter = 300, a_method = 'temporal_kernel', d_method = 'sparse')
    
    st = time.time()
    tkst_test.fit(sd, true_a=alphas, true_d=v, evaluate=True)
    end = time.time()
    tkst_re_dict, tkst_re_alphas = (tkst_test.dict, tkst_test.alphas)
    est_C = np.einsum('ki,kj->kij', tkst_re_dict, tkst_re_dict)
    test_covar = np.einsum('tk,kij->tij', tkst_re_alphas, est_C)
    time_counter[-1, round] = end -st
    sine_metric[-1, round] =avg_LERM_cov(alphas[:, 0:test_T].T , v, test_covar[0:test_T,:,:])
    
    print("Round:", round, " regularized approach:", "%4f"%avg_LERM_cov(alphas[:, 0:test_T].T , v, test_covar[0:test_T,:,:]), "ext time:%4f"%(end-st)) 



    np.savez('../../results/ortho_version{}_{}_subjects_{}_lv_{}.npz'.format(args.title,args.waveform, N,args.noise_level), metric=sine_metric, ext_time=time_counter)

