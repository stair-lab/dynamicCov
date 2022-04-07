##########################
#filename:simulation_kerneltype.py
#description:
#   a. this program tests different type of kernel functions


import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
import snscov.synth_data as synth_data
from  snscov.model import snscov_model
from snscov.kernel import *
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
    amp=4
elif args.waveform == 'square':
    alphas=synth_data.synthesize_switches2(K,T)
    amp=1.5
elif args.waveform == 'mixing':
    alphas=synth_data.synthesize_weights(K,T)
    amp=4


kernel_type= 'Matern52'
np.random.seed(8)
sd = synth_data.synthesize_data_noiselevel(alphas, C, N, BETA_inv = .0)
if kernel_type == 'Matern52':
    kernel = Matern52Kernel(T, 2, 20)
elif kernel_type == 'RBF':
    kernel = RBFKernel(T, length_scale=20.0)
elif kernel_type == 'RationalQuadratic':
    kernel = RationalQuadraticKernel(T, length_scale=20.0)
elif 'ExpSineSquared':
    kernel = ExpSineSquaredKernel(T, length_scale=20.0)

#kernel = Matern52Kernel(T, 2,40)
np.set_printoptions(precision=2)

kernel += 1e-12*np.eye(T)
kernel = linalg.solve(kernel, np.eye(T))

synth_data.plot_components(np.einsum('ki,kj->kij', v, v), 
                                    fname='../../figures/a_components.png', 
                                    title_append="Ground Truth Components")

synth_data.plot_weights((alphas) ,
                        fname='../../figures/a_weights.png', 
                        title_append="Ground Truth Weights")


a_coef = np.array([0.01]) #[100]
b_coef = np.array([ 7])
lr_rate = np.array([1e-3])
max_iter = 200
tkst_metric = np.zeros((9, 2, a_coef.size, b_coef.size, lr_rate.size))
distance_metric = np.zeros((9 , a_coef.size, b_coef.size, lr_rate.size, max_iter+1))
for i in range (1):


    subjects = []
    number = [ 1,2,3,4, 5, 10,15, 20,200]
    for q in number:
        title = "kernel_{}_subject{}".format(kernel_type, q)
        subjects.append(title)
    print(subjects)
    for s in range(len(number)):

        for b in range(len(b_coef)):
            for a in range(len(a_coef)):
                for l in range(len(lr_rate)):
                    subject = subjects[s]
                    tkst_test = snscov_model(K, T, D, la_rate = lr_rate[l], amp=amp,tol=0.001,
                                            ld_rate=lr_rate[l]/T, kernel = kernel, smooth=a_coef[a],k_sparse=b_coef[b],
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
                    print(dual_permutation_separate((alphas).T, v, tkst_re_alphas, tkst_re_dict))
                    title=args.waveform+'_'+subject
                    if i == 0:
                        _,r_matrix = dual_permutation_matrix(alphas.T, v, tkst_re_alphas, tkst_re_dict)
                        synth_data.plot_components(np.einsum('ki,kj->kij',  r_matrix.T @ tkst_re_dict, r_matrix.T @ tkst_re_dict ), 
                                                            fname='../../figures/random_{}_components_{}{}{}.png'.format(title, a,b,l), 
                                                            title_append="Components Recovery (subjects={})".format(number[s]))
                        
                        synth_data.plot_weights((tkst_re_alphas @ r_matrix ).T ,
                                                fname='../../figures/random_{}_weights_{}{}{}.png'.format(title, a,b, l), 
                                                title_append="Weights Recovery (subjects={})".format(number[s]))
                        
                        np.save('../../results/model_{}_{}{}{}.npy'.format(title, a,b,l), tkst_re_alphas @ r_matrix,  r_matrix.T @ tkst_re_dict)
                                                     
                        
                        t = np.arange(tkst_test.n_iter_)
                        plt.figure()

                        plt.title(r"$dist^2$($\bf{{Z}}$,$\bf{{Z}}^\star$) $\lambda_1$={}, $\lambda_2$={}, lr={}".format(a_coef[a], b_coef[b], lr_rate[l]))
                        plt.plot(t, tkst_test.dual_ortho, label='distance', color='teal')
                        plt.legend(loc="best")
                        plt.xlabel('iteration')
                        otho_len = np.size(tkst_test.dual_ortho)
                        matrix = np.ones(max_iter+1) * -1
                        matrix[0:otho_len]=tkst_test.dual_ortho
                        distance_metric[s,a,b,l,:]=matrix
                        plt.savefig('../../figures/{}_dist_{}{}{}.png'.format(title, a,b, l))
                        plt.close()
    plt.figure(figsize=(4,4))           
    for i in range(9):
        index = np.where(distance_metric[i].squeeze()>0.)[0][:-1]
        plt.plot(distance_metric[i].squeeze()[index], label='N='+str(number[i]),marker='*')
    plt.legend(loc="upper right")
    plt.xlabel('iteration')
    plt.ylim(0,100)
    plt.ylabel(r"$dist^2$($\bf{{Z}}$,$\bf{{Z}}^\star$)")
    plt.grid()
    plt.tight_layout()
    plt.savefig('../../figures/random_linear_convergence_{}.png'.format(args.waveform),bbox_inches = 'tight', pad_inches = 0)
    subject_id = 'simulation'                    
   
    np.save('../../results/{}_metric.npy'.format(subject_id), tkst_metric)
    np.save('../../results/{}_distance_metric.npy'.format(subject_id), distance_metric)
    
    