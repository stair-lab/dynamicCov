import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
import snscov.synth_data as synth_data
from  snscov.model_large import snscov_model
from snscov.kernel import Matern52Kernel
from snscov.evaluate import *
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
import time
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--n_trials', type=int, default=11, help='number of trials')
parser.add_argument('--N', type=int, default=50, help='number of subject')
parser.add_argument('--K', type=int, default=10, help='number of components')
parser.add_argument('--D', type=int, default=100, help='data dimension')
parser.add_argument('--T', type=int, default=100, help='number of time points')
parser.add_argument('--density', type=float, default=0.3, help='matrix sparsity')
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




kernel = Matern52Kernel(T, 2, args.kernel_length)
np.set_printoptions(precision=2)

kernel = linalg.solve(kernel, np.eye(T))

nonzero = np.sum(v!=0,axis=1)

print('maximum number of non-zero component:',np.max(nonzero))
import math
s1 = math.ceil(0.5*np.max(nonzero))
s2 = math.ceil(1*np.max(nonzero))
s3 = math.ceil(1.5*np.max(nonzero))
s4 = math.ceil(2*np.max(nonzero))
a_coef = np.array([0.1]) #[100]
import math
b_coef = np.array([s4])
lr_rate = np.array([1e-3])
amp=np.ceil(1.2*np.max(alphas))+1.
print('max amplitude', amp)
max_iter=100
number = [args.N]

meta_tkst_metric = []
meta_distance_metric = []
for i in range (args.n_trials):
    sd = synth_data.synthesize_data_noiselevel(alphas, C, N, BETA_inv = 0.)
    tkst_metric = np.zeros((len(number), 4, a_coef.size, b_coef.size, lr_rate.size))
    distance_metric = np.zeros((len(number) , a_coef.size, b_coef.size, lr_rate.size, max_iter+1))

    subjects = []
    
    for q in number:
        title = "projAD_mom_subject{}".format(q)
        subjects.append(title)
    print(subjects)
    for s in range(len(number)):
        S_N = np.einsum('nji,njk->ik', sd[0:number[s]], sd[0:number[s]])/ number[s]
        w, eigv = linalg.eigh(S_N)
        

        w = w[::-1] #shape(D) v
        eigv = eigv[:,::-1] #shape(D, D)v
        S_T = np.einsum('ntk,ntj->tkj', sd[0:number[s]], sd[0:number[s]]) / number[s] #shape(T,D,D) v
        S_Tv = np.einsum('tkj,ji->tki', S_T, eigv) #shape(T,D,D) v
        vtS_Tv = np.einsum('ki,tkj->tij', eigv, S_Tv) # shape(T,D,D) v
        A = np.einsum('tii->ti', vtS_Tv) #v
            

        #synth_data.plot_weights(A.T[0:K,::], '../figures/{}_initial_weights.png'.format(subjects[s]),title_append="Initial Weights")
        C = np.einsum('ki,kj->kij', eigv.T, eigv.T)
        #synth_data.plot_components(C[0:K,:,:], '../figures/{}_initial_componts.png'.format(subjects[s]),title_append="Initial Components")
        
        for b in range(len(b_coef)):
            for a in range(len(a_coef)):
                for l in range(len(lr_rate)):
                    subject = subjects[s]
                    tkst_test =  snscov_model(K, T, D, la_rate = lr_rate[l], amp=amp,tol=10,
                                            ld_rate=lr_rate[l]/T, kernel = kernel, smooth=a_coef[a],k_sparse=b_coef[b],
                                            max_iter = max_iter, a_method = 'temporal_kernel', d_method = 'sparse')
                    tkst_test.fit(sd[0:number[s],:,:], true_a=alphas, true_d=v, evaluate=True)

                    tkst_re_dict, tkst_re_alphas = tkst_test.best_dictionary, tkst_test.best_alphas
 
                    tkst_metric[s,:, a, b, l] = np.array([avg_LERM((alphas).T, v, tkst_re_alphas , tkst_re_dict),
                                                    large_scale_signed_permutation((alphas).T, v, tkst_re_alphas, tkst_re_dict),
                                                    ortogonal_procrustes((alphas).T, v, tkst_re_alphas, tkst_re_dict),
                                                    dual_orthogonal_procrustes((alphas).T, v, tkst_re_alphas, tkst_re_dict)])

                    
                    print('[tkst]', a_coef[a], b_coef[b], lr_rate[l], tkst_metric[s,:,a,b,l]) 

                    title='tkst_'+subject
                    if i == 1:
                        """
                        synth_data.plot_components(np.einsum('ki,kj->kij',  tkst_re_dict, tkst_re_dict ), 
                                                            fname='../figures/{}_components_{}{}{}.png'.format(title, a,b,l), 
                                                            title_append="Components Recovery (subjects={})".format(number[s]))
                        
                        synth_data.plot_weights((tkst_re_alphas ).T ,
                                                fname='../figures/{}_weights_nosquare_{}{}{}.png'.format(title, a,b, l), 
                                                title_append="Weights Recovery (subjects={})".format(number[s]))
                        np.save('model_{}_{}{}{}.npy'.format(title, a,b,l), tkst_re_alphas,   tkst_re_dict)
                        
                        #synth_data.plot_metric(tkst_test.error_, tkst_test.residuals, tkst_test.alphas_error, tkst_test.dic_error, 
                        #                        fname='../figures/{}_error_{}{}.png'.format(title, a, l))                                
                        
                        t = np.arange(tkst_test.n_iter_)
                        plt.figure()

                        plt.title(r"$dist^2$($\bf{{Z}}$,$\bf{{Z}}^\star$) $\lambda_1$={}, $\lambda_2$={}, lr={}".format(a_coef[a], b_coef[b], lr_rate[l]))
                        plt.plot(t, tkst_test.dual_ortho, label='distance', color='teal')
                        plt.legend(loc="best")
                        plt.xlabel('iteration')
                        #plt.show()
                        """
                        otho_len = np.size(tkst_test.dual_ortho)
                        matrix = np.ones(max_iter+1) * -1
                        matrix[0:otho_len]=tkst_test.dual_ortho
                        distance_metric[s,a,b,l,:]=matrix
                        #plt.savefig('../figures/{}_dist_{}{}{}.png'.format(title, a,b, l))
                        #plt.close()
    meta_distance_metric.append(distance_metric)
    meta_tkst_metric.append(tkst_metric)                
meta_distance_metric = np.array(meta_distance_metric)
meta_tkst_metric = np.array(meta_tkst_metric)
np.save('../../results/{}.npy'.format(args.filename),{'metric':meta_tkst_metric, 'distance': meta_distance_metric})
    
                        

