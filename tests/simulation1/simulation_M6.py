##########################
#filename:simulation_M6.py
#description:
#   a. this program implements M6 (regularization approach)
#  

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
rho = 2.5
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





kernel = Matern52Kernel(T, 2, 5)
#kernel = Matern52Kernel(T, 2,40)
np.set_printoptions(precision=5)

kernel = linalg.solve(kernel, np.eye(T))




a_coef = np.array([0.0001]) #[100]
b_coef = np.array([ 0.1])
lr_rate = np.array([1e-4])
max_iter = 2000
tkst_metric = np.zeros((10, 2, a_coef.size, b_coef.size, lr_rate.size))

num_run = 20
distance_metric = np.zeros((6,num_run))
for i in range (num_run):

    np.random.seed(i)
    sd = synth_data.synthesize_data_noiselevel(alphas, C, N, BETA_inv = .0)
    subjects = []
    number = [ 1, 5,15,200, 1000,10000]
    for q in number:
        title = "subject{}".format(q)
        subjects.append(title)
    print(subjects)
    for s in range(len(number)):
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
                    
                    print(avg_LERM_cov((alphas).T, v, est_covar ))
                    tkst_metric[s,:, a, b, l] = np.array([avg_LERM_cov((alphas).T, v, est_covar ),
                                                    dual_permutation((alphas).T, v, tkst_re_alphas, tkst_re_dict)])
                    
                    print('result: smoothness {}, sparse {},lr {}'.format(a_coef[a], b_coef[b], lr_rate[l]), tkst_metric[s,:,a,b,l]) 
                    print(dual_permutation_separate((alphas).T, v, tkst_re_alphas, tkst_re_dict))
                    title=args.waveform+'_'+subject
                    distance_metric[s,i], _, _ = dual_permutation_separate((alphas).T, v, tkst_re_alphas, tkst_re_dict)


np.save('../../results/reg_20_replica_{}_distance_metric.npy'.format(args.waveform, distance_metric), distance_metric)

def remove_outlier(arr, a=2):
    mean = np.mean(arr)
    std = np.std(arr)
    keep = np.intersect1d(np.where(arr <= mean +a*std)[0], np.where(arr>= mean -a*std)[0])
    keep = np.array(keep)

    out = [arr[i] for i in keep]
    return out


print(np.mean(distance_metric, axis=1))
print(np.std(distance_metric, axis=1))

for i in range(distance_metric.shape[0]):
    out = remove_outlier(distance_metric[i,:], a=1)
    print(np.mean(out),np.std(out))
    