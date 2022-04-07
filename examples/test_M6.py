
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
import snscov.synth_data as synth_data
from  snscov.model_reg import Model6
from snscov.kernel import Matern52Kernel
from snscov.evaluate import *
import argparse

parser = argparse.ArgumentParser(description='Process some integers. ')
parser.add_argument('--waveform',     type=str, default='mixing',    help='select waveform')

args = parser.parse_args()
N = 2000
K = 4
T = 50
D = 20

#sybthesize spatial components
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
    amp=2*np.max(alphas)
elif args.waveform == 'square':
    alphas=synth_data.synthesize_switches2(K,T)
    amp=1.2*np.max(alphas)
elif args.waveform == 'mixing':
    alphas=synth_data.synthesize_weights(K,T)

    amp=2*np.max(alphas)

kernel = Matern52Kernel(T, 2, 200)
np.set_printoptions(precision=2)

kernel = linalg.solve(kernel, np.eye(T))


np.random.seed(8)
sd = synth_data.synthesize_data_noiselevel(alphas, C, N, BETA_inv = .0)
a_coef = np.array([ 0.0001, 0.01, 0.1]) #[100]
b_coef = np.array([ 0.01, 0.1])
lr_rate = np.array([1e-3, 1e-4,1e-5])

max_iter = 2000
tkst_metric = np.zeros((3, 2, a_coef.size, b_coef.size, lr_rate.size))
distance_metric = np.zeros((3 , a_coef.size, b_coef.size, lr_rate.size, max_iter))

num_run = 1
for i in range (num_run):
    subjects = ['mom_subject1','mom_subject5', 'mom_subject15']
    number = [1,5,15]
    for s in range(3):
        for b in range(len(b_coef)):
            for a in range(len(a_coef)):
                for l in range(len(lr_rate)):
                    subject = subjects[s]
                    tkst_test = Model6(K, T, D, lambda1 = a_coef[a], lambda2= b_coef[b], la_rate = lr_rate[l],
                                            ld_rate=lr_rate[l], kernel = kernel, group=None,
                                            max_iter = max_iter, a_method = 'temporal_kernel', d_method = 'sparse')
                    tkst_test.fit(sd[0:number[s],:,:], true_a=alphas, true_d=v, evaluate=True)

                    tkst_re_dict, tkst_re_alphas = tkst_test.dict, tkst_test.alphas

                    test_covariance = np.zeros((T, D, D))
                    for t in range (T):
                        test_covariance[t,:,:] = tkst_re_dict.T @ np.diag(tkst_re_alphas[t,:]) @ tkst_re_dict
                    
                    est_C  = np.einsum('ki,kj->kij', tkst_re_dict,  tkst_re_dict)
                    est_covar  = np.einsum('ti,ijk->tjk',tkst_re_alphas ,  est_C)
                    tkst_metric[s,:, a, b, l] = np.array([avg_LERM_cov((alphas).T, v, est_covar ),
                                                    dual_permutation((alphas).T, v, tkst_re_alphas, tkst_re_dict)])
                    print('run:', s, 
                        '[tkst reg]', 
                        'a coef:',  a_coef[a],
                        'b coef:',  b_coef[b], 
                        'lr rate:', lr_rate[l], 
                        '\n',
                        'avg lerm:', tkst_metric[s,0,a,b,l],
                        'dist:', tkst_metric[s,1,a,b,l]) 

                    #print(dual_permutation_separate((alphas).T, v, tkst_re_alphas, tkst_re_dict))
                    title='tkst_'+subject
                    if i == 0:
   
                        otho_len = np.size(tkst_test.dual_ortho)
                        matrix = np.ones(max_iter) * -1
                        matrix[0:otho_len]=tkst_test.dual_ortho
                        distance_metric[s,a,b,l,:]=matrix
                        

    np.save('../output/reg_{}_tkst_metric.npy'.format('parameter'), tkst_metric)
    np.save('../output/reg_{}_distance_metric.npy'.format('parameter'), distance_metric)