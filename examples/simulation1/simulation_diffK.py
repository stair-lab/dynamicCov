##########################
#filename:simulation_diffK.py
#description:
#   a. this program tests the model with varying K. 
#   b. this program generates Figure 8 -- 11 in the manuscript
#   c. load snscov_model from snscov.model_large when K is greater than 8. 
#   this model uses birkhoff decomposition to relax the combinatorial problem of finding best permutation matrix

import matplotlib.pyplot as plt 

import numpy as np
from scipy import linalg
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
import snscov.synth_data as synth_data
from  snscov.model import snscov_model

#load the following model when K>8 and replace snscov_model with snscov_model_large below
from snscov.model_large import snscov_model as snscov_model_large
from snscov.kernel import Matern52Kernel
from snscov.evaluate import *
import argparse
from matplotlib.ticker import FormatStrFormatter

from matplotlib.gridspec import GridSpec
#produce plot for different K
from plot_format import *
plt.rcParams.update({'font.size': 22})


parser = argparse.ArgumentParser(description='Process some integers. ')
parser.add_argument('--waveform',     type=str, default='mixing',    help='select waveform')
parser.add_argument('--K',     type=int, default=2,    help='select K')

args = parser.parse_args()
N = 2000
K = args.K
T = 50
D = 20
fullcov = np.zeros((K, D, D), dtype=np.float32)
v = np.zeros((K,20))

if K == 2:
    v[0,0:5] = np.array([1, -1, 1, -1, -1])
    v[0,10:15] = np.array([1, 1, 1, 1, 1])
    v[1,5:10] = np.array([1, -1, -1, 1, -1])
    v[1,15:20] = np.array([-1, 1, -1, 1, -1])
    b_coef = np.array([ 10])
elif K == 4:
    v[0,0:5] = np.array([1, -1, 1, -1, -1])
    v[0,9:10] = np.array([1])

    v[1,5:9] = np.array([1,-1, -1, 1])

    v[2,10:13] = np.array([1, 1, -1])

    v[3,13:16] = np.array([1, -1,  1])
    v[3,17:20] = np.array([-1, -1, -1])
    b_coef = np.array([ 7])
elif K == 6:
    v[0,0:4] = np.array([1, -1, 1, -1])
    v[1,4:5] = np.array([-1])
    v[1,9:10] = np.array([1])

    v[2,5:9] = np.array([1,-1, -1, 1])

    v[3,10:13] = np.array([1, 1, -1])

    v[4,13:16] = np.array([1, -1,  1])
    v[5,17:20] = np.array([-1, -1, -1])  
    b_coef = np.array([ 4])  
elif K == 8:
    v[0,0:2] = np.array([1, -1])
    v[1,2:4] = np.array([1, -1])
    v[2,4:5] = np.array([-1])
    v[2,9:10] = np.array([1])

    v[3,5:7] = np.array([ -1, 1])
    v[4,7:9] = np.array([1,-1])
    v[5,10:13] = np.array([1, 1, -1])

    v[6,13:16] = np.array([1, -1,  1])
    v[7,17:20] = np.array([-1, -1, -1])  
    b_coef = np.array([ 5])  
elif K == 10:
    v[0,0:2] = np.array([1, -1])
    v[1,2:4] = np.array([1, -1])
    v[2,4:5] = np.array([-1])
    v[2,9:10] = np.array([1])

    v[3,5:7] = np.array([ -1, 1])
    v[4,7:9] = np.array([1,-1])
    v[5,10:12] = np.array([1, 1])
    v[6,12:14] = np.array([-1, 1])
    v[7,14:16] = np.array([-1,  1])
    v[8,16:18] = np.array([1,-1])
    v[9,18:20] = np.array([-1, -1])  
    b_coef = np.array([ 2])  


v /= np.sqrt(np.sum(v ** 2, axis=1))[:,np.newaxis]
C = np.einsum('ki,kj->kij', v, v)


alphas=synth_data.synthesize_weights(K,T)
amp=2.*np.max(alphas)


kernel = Matern52Kernel(T, 2,200)
np.set_printoptions(precision=2)

kernel = linalg.solve(kernel, np.eye(T))

a_coef = np.array([0.001]) #[100]
 
lr_rate = np.array([1e-5])
max_iter = 100
tkst_metric = np.zeros((10, 2, a_coef.size, b_coef.size, lr_rate.size))
distance_metric = np.zeros((10 , a_coef.size, b_coef.size, lr_rate.size, max_iter+1))
#distance_metric = np.zeros((6,20))

dict_batch = []
alphas_batch = []
rmatirx_batch = []

for i in range (1):
    np.random.seed(8)
    sd = synth_data.synthesize_data_noiselevel(alphas, C, N, BETA_inv = .0)
    subjects = []
    number = [20,200,2000]
    for q in number:
        title = "subject{}".format(q)
        subjects.append(title)
    print(subjects)
    for s in range(len(number)):
        for b in range(len(b_coef)):
            for a in range(len(a_coef)):
                for l in range(len(lr_rate)):
                    subject = subjects[s]
                    #use snscov_model_large when K==10
                    tkst_test = snscov_model(K, T, D, la_rate = lr_rate[l], amp=amp,tol=1e-3,
                                            ld_rate=lr_rate[l]/T, kernel = kernel, smooth=a_coef[a],k_sparse=b_coef[b],
                                            max_iter = max_iter, a_method = 'temporal_kernel', d_method = 'sparse')

                    tkst_test.fit(sd[0:number[s],:,:], true_a=alphas, true_d=v, evaluate=True)

                    tkst_re_dict, tkst_re_alphas = tkst_test.best_dictionary, tkst_test.best_alphas

                    dis,r_matrix = dual_permutation_matrix(alphas.T, v, tkst_re_alphas, tkst_re_dict)
                    #uncomment the following line when using sns_movel_large
                    #dis,r_matrix = large_scale_signed_permutation_matrix(alphas.T, v, tkst_re_alphas, tkst_re_dict)
                    dict_batch.append(tkst_re_dict)
                    alphas_batch.append(tkst_re_alphas)
                    rmatirx_batch.append(r_matrix)
                   
if args.K == 2:
    plot_func = plot2figK2
elif args.K == 6:
    plot_func = plot2figK6
elif args.K == 8:
    plot_func = plot2figK8
elif args.K == 10:
    plot_func = plot2figK10


plot_func(alphas, 
        np.einsum('ki,kj->kij', v, v),
        (alphas_batch[0] @ rmatirx_batch[0]).T,
        np.einsum('ki,kj->kij',  rmatirx_batch[0].T @ dict_batch[0], rmatirx_batch[0].T @ dict_batch[0]),
        (alphas_batch[1] @ rmatirx_batch[1]).T,
        np.einsum('ki,kj->kij',  rmatirx_batch[1].T @ dict_batch[1], rmatirx_batch[1].T @ dict_batch[1]),        
        (alphas_batch[2] @ rmatirx_batch[2]).T,
        np.einsum('ki,kj->kij',  rmatirx_batch[2].T @ dict_batch[2], rmatirx_batch[2].T @ dict_batch[2]),        
        '../../figures/combine_K{}.pdf'.format(args.K)
    )