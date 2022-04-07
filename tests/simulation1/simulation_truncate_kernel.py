import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
import snscov.synth_data as synth_data
from  snscov.model_truncate import snscov_model
from snscov.kernel import Matern52Kernel
from snscov.evaluate import *
from mpl_toolkits.mplot3d import Axes3D


import argparse

parser = argparse.ArgumentParser(description='Process some integers. ')
parser.add_argument('--waveform',     type=str, default='mixing',    help='select waveform')

args = parser.parse_args()
N = 200
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
    amp=2.5*np.max(alphas)
elif args.waveform == 'square':
    alphas=synth_data.synthesize_switches2(K,T)
    amp=1.2*np.max(alphas)
elif args.waveform == 'mixing':
    alphas=synth_data.synthesize_weights(K,T)

    amp=2*np.max(alphas)



np.random.seed(8)
sd = synth_data.synthesize_data_noiselevel(alphas, C, N, BETA_inv = .0)

kernel = Matern52Kernel(T, 2, 500)
#linalg.eigh(kernel)
#kernel = Matern52Kernel(T, 2,40)
np.set_printoptions(precision=2)




a_coef = np.array([1]) #[100]
kernel2 = linalg.solve(kernel, np.eye(T))
w,eigh = linalg.eigh(kernel)
#print(w)
idx = np.where(w>=.001)
#print(idx)
new_w = 1./w[idx]
new_eigh = np.squeeze(eigh[:,idx])
#print(np.diag(new_w).shape)
#print(new_eigh.shape)
kernel = new_eigh @ np.diag(new_w) @ new_eigh.T

#print(linalg.norm(kernel-kernel2))
b_coef = np.array([ 7])
lr_rate = np.array([1e-4])
max_iter = 200
tkst_metric = np.zeros((10, 2, a_coef.size, b_coef.size, lr_rate.size))
distance_metric = np.zeros((10 , a_coef.size, b_coef.size, lr_rate.size, max_iter+1))
for i in range (1):


    subjects = []
    number = [ 1, 5,15,200]
    for q in number:
        title = "subject{}".format(q)
        subjects.append(title)
    print(subjects)
    for s in range(len(number)):
        for b in range(len(b_coef)):
            for a in range(len(a_coef)):
                for l in range(len(lr_rate)):
                    subject = subjects[s]
                    tkst_test = snscov_model(K, T, D, la_rate = lr_rate[l], amp=amp,tol=0.001,
                                            ld_rate=lr_rate[l]/T, w=new_w, v=new_eigh, smooth=a_coef[a],k_sparse=b_coef[b],
                                            max_iter = max_iter, a_method = 'temporal_kernel', d_method = 'sparse')
                    tkst_test.fit(sd[0:number[s],:,:], true_a=alphas, true_d=v, evaluate=True)

                    tkst_re_dict, tkst_re_alphas = tkst_test.best_dictionary, tkst_test.best_alphas

                    test_covariance = np.zeros((T, D, D))
                    for t in range (T):
                        test_covariance[t,:,:] = tkst_re_dict.T @ np.diag(tkst_re_alphas[t,:]) @ tkst_re_dict
                    
                    est_C  = np.einsum('ki,kj->kij', tkst_re_dict,  tkst_re_dict)
                    est_covar  = np.einsum('ti,ijk->tjk',tkst_re_alphas ,  est_C)
                    tkst_metric[s,:, a, b, l] = np.array([avg_LERM_cov((alphas).T, v, est_covar ),
                                                    dual_permutation((alphas).T, v, tkst_re_alphas, tkst_re_dict)])
                    
                    print('result: smoothness {}, sparse {},lr {}'.format(a_coef[a], b_coef[b], lr_rate[l]), tkst_metric[s,:,a,b,l]) 
                    print('dual permutation',dual_permutation_separate((alphas).T, v, tkst_re_alphas, tkst_re_dict))

                    title=args.waveform+'_'+subject
                    if i == 0:
 
                        otho_len = np.size(tkst_test.dual_ortho)
                        matrix = np.ones(max_iter+1) * -1
                        matrix[0:otho_len]=tkst_test.dual_ortho
                        distance_metric[s,a,b,l,:]=matrix

    subject_id = 'simulation_{}'.format(args.waveform)                    
   
    np.save('../../results/{}_truncate_metric.npy'.format(subject_id), tkst_metric)
    np.save('../../results/{}_truncate_distance_metric.npy'.format(subject_id), distance_metric)
     
    