import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import time
from sklearn.preprocessing import normalize
import sys
sys.path.append('../../src')
plt.rcParams['text.latex.preamble']=[r"\udepackage{asmath}"]
from snscov.model import snscov_model
from snscov.model_reg import Model6
from snscov.kernel import Matern52Kernel
from snscov.evaluate import *
from snscov.methods import *
import snscov.synth_data as synth_data
import argparse

parser = argparse.ArgumentParser(description='Process some integers. ')
parser.add_argument('--num_subjects', type=int, default=10,         help='number of test subjects')
parser.add_argument('--num_trial',    type=int, default=20,        help='number of trials')
parser.add_argument('--waveform',     type=str, default='mixing',    help='select waveform')
parser.add_argument('--title',        type=str, default='compare_lv10_take2', help='experiment name')
parser.add_argument('--noise_level',  type=int, default=2,       help='noise level'  )
args = parser.parse_args()
np.set_printoptions(precision=2)

N = args.num_subjects
K = 4
T = 50
D = 16

v = np.array([[1, -1, 1, -1,  0, 0, 0, 0,  0,  0,  0,  0, 0, 0, 0, 0],
              [0,  0, 0,  0, -1, 1, 0, 1,  0,  0,  0,  0, 0, 0, 0, 0],
              [0,  0, 0,  0,  0, 0, 0, 0,  0,  0,  0,  0, 1, 1, -1, 1],
              [0,  0, 0,  0,  0, 0, 0, 0, -1,  1,  1,  0, 0, 0, 0,  0]], dtype=np.float32)
v /= np.sqrt(np.sum(v ** 2, axis=1))[:,np.newaxis]

C = np.einsum('ki,kj->kij', v, v)
W = [5,10,20]
test_T = T - max(W) + 1
states = [4,8,12,16]


#test sine waves
if args.waveform == 'sine':
    alphas=synth_data.synthesize_sines(K,T)
    length_scale = 200
elif args.waveform == 'square':
    alphas=synth_data.synthesize_switches(K,T)
    length_scale = 2
elif args.waveform == 'mixing':
    alphas=synth_data.synthesize_weights(K,T)
    length_scale = 200


kernel = Matern52Kernel(T, 2, length_scale) 
kernel = linalg.solve(kernel, np.eye(T))
sine_metric = np.zeros((len(W)+7,args.num_trial))
time_counter = np.zeros(sine_metric.shape)
for round in range (args.num_trial):
    sd = synth_data.synthesize_data_noiselevel(alphas, C, N, 1./args.noise_level)

    #sliding window method (deprecated)
    #for idx, w in enumerate(W):
    #    st = time.time()
    #    cov = sliding_window_covaraince(sd, w)
    #    end = time.time()
    #    time_counter[idx, round] = end -st
    #    sine_metric[idx, round] = avg_cov(alphas[:, 0:test_T].T, v, cov[0:test_T,:,:])
    #    print("Round:", round, " SW w_length:", w, "%4f"%avg_cov(alphas[:, 0:test_T].T, v, cov[0:test_T,:,:]), "ext time:%4f"%(end-st))
    
    #run M1

    for idx, w in enumerate(W):
        st = time.time()
        cov = SWPCA(sd, K, w)
        end = time.time()
        time_counter[idx, round] = end - st
        sine_metric[idx, round] = avg_cov(alphas[:, 0:test_T].T, v, cov[0:test_T,:,:])
        print("Round:", round,  " SWPCA w_length:", w, "%4f"%avg_cov(alphas[:, 0:test_T].T, v, cov[0:test_T,:,:]), "ext time:%4f"%(end-st))


    #run M2    
    # denotes the number of states
    for idx, s in enumerate([12]):
        st = time.time()
        cov, states_cov = HMM(sd, s, method="em", n_iters=300)
        end = time.time()
        time_counter[idx+len(W), round] = end - st
        sine_metric[idx+len(W), round] = avg_cov(alphas[:, 0:test_T].T, v, cov[0:test_T,:,:]) 
        print("Round:", round, " HMM n_states:", s, "%4f"%avg_cov(alphas[:, 0:test_T].T, v, cov[0:test_T,:,:]), "ext time:%4f"%(end-st))

    #run M3
    for idx, s in enumerate([12]):
        st = time.time()
        cov, states_cov = ARHMM(sd, s, method="em", n_iters=300)
        end = time.time()
        time_counter[idx+len(W)+1, round] = end - st
        sine_metric[idx+len(W)+1, round] = avg_cov(alphas[:, 0:test_T].T, v, cov[0:test_T,:,:])
        print("Round:", round, " ARHMM n_states:", s, "%4f"%avg_cov(alphas[:, 0:test_T].T, v, cov[0:test_T,:,:]), "ext time:%4f"%(end-st))            

   #run M4
    st = time.time()
    cov = sparse_dict_learning(sd, K, alpha=1.0)
    end = time.time()
    time_counter[len(W)+2, round] = end-st
    sine_metric[len(W)+2, round] = avg_cov(alphas[:, 0:test_T].T , v, cov[0:test_T,:,:])

    print("Round:", round, " Sparse dictionary learning:", "%4f"%avg_cov(alphas[:, 0:test_T].T , v, cov[0:test_T,:,:]), "ext time:%4f"%(end-st)) 


    #run MS  
    st = time.time()
    cov, _, _ = spectral_initialization(sd, K)
    end = time.time()
    time_counter[len(W)+3, round] = end - st
   
    sine_metric[len(W)+3, round] = avg_cov(alphas[:, 0:test_T].T, v, cov[0:test_T,:,:])  
    print("Spectral Initialization",  "%4f"%avg_cov(alphas[:, 0:test_T].T, v, cov[0:test_T,:,:]), "ext time:%4f"%(end-st))


    #run M**      
    tkst_test = snscov_model(K, T, D, la_rate = 1e-6, ld_rate= 1e-6/T, kernel = kernel, smooth=0.1, k_sparse=4,
                                            max_iter = 50, a_method = 'temporal_kernel', d_method = 'sparse')
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
    tkst_test = snscov_model(K, T, D, la_rate = 1e-1, ld_rate= 1e-1, kernel = kernel, smooth=0.1, k_sparse=4,ini_method="random",
                                            max_iter = 300, a_method = 'temporal_kernel', d_method = 'sparse')
    st = time.time()
    tkst_test.fit(sd, true_a=alphas, true_d=v, evaluate=True)
    end = time.time()
    tkst_re_dict, tkst_re_alphas = (tkst_test.best_dictionary, tkst_test.best_alphas)
    est_C = np.einsum('ki,kj->kij', tkst_re_dict, tkst_re_dict)
    test_covar = np.einsum('tk,kij->tij', tkst_re_alphas, est_C)
    time_counter[len(W)+5, round] = end -st
    sine_metric[len(W)+5, round] =avg_LERM_cov(alphas[:, 0:test_T].T , v, test_covar[0:test_T,:,:])
    
    print("Round:", round, " Random initialization:", "%4f"%avg_LERM_cov(alphas[:, 0:test_T].T , v, test_covar[0:test_T,:,:]), "ext time:%4f"%(end-st)) 

    #run M6  
    tkst_test = Model6(K, T, D, lambda1 = 0.0001, lambda2= 0.1, la_rate = 1e-4,
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



    np.savez('../../results/compare_{}_{}_subjects_{}_lv_{}_run1.npz'.format(args.title,args.waveform, N,args.noise_level), metric=sine_metric, ext_time=time_counter)
 



