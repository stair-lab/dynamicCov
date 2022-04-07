import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from sklearn.preprocessing import normalize
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
from snscov.model_task import snscov_model
from snscov.kernel import Matern52Kernel
from snscov.evaluate import *
import snscov.synth_data as synth_data
import sys
import scipy.io

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--num_s',    type=int, default=20,       help='bumber of subject')
parser.add_argument('--start_id', type=int, default=0,       help='start subject idx, less than number of total test subjects - number of subjects')
parser.add_argument('--max_iter', type=int, default=400,     help='maximum iteration')
parser.add_argument('--title',    type=str, default='tfmri_sub20_15', help='experiment name')
parser.add_argument('--length_scale', type=int, default=5,   help='length_scale')
args = parser.parse_args()
#set print precision
np.set_printoptions(precision=2)

#load task activattion
motor = scipy.io.loadmat('../data/HCP_motor/reg_motor_conv.mat')
motor_conv = motor["reg_motor_conv"]
label= ["Right Hand Tapping", "Left Foot Tapping", "Tongue Wagging", "Right Foot Tapping", "Left Hand Tapping"]
print(motor_conv.shape)
for i in range(5):
    plt.plot(motor_conv[:,i], label=label[i])
plt.xlabel('time (s)')
plt.title('Task Acitvation')
plt.legend(loc='lower right')
plt.savefig('../figures/motor.png')
plt.close()


#load data
data_matrix=[]
for i in range(20):

    mat = scipy.io.loadmat('../data/HCP_motor/sub{}.mat'.format(i+1))
    data = mat["data4"]
    data_matrix.append(data)


train_data = np.array(data_matrix).transpose((0,2,1))
#train_data = train_data[:,10::,]
print(train_data.shape)

N = train_data.shape[0]
T = train_data.shape[1]
D = train_data.shape[2]
K = motor_conv.shape[1]+10


kernel = Matern52Kernel(T, 2, args.length_scale)
kernel = linalg.solve(kernel, np.eye(T))

smooth_coef = np.array([1.0,])
sparse_coef = np.array([54])#np.array([np.int(D*(i+1)*0.1) for i in range(0,10,3)])

lr_rate = np.array([ 1.0,])
max_iter = args.max_iter
#metric = np.zeros((4, 2, smooth_coef.size, sparse_coef.size, lr_rate.size))
#distance_metric = np.zeros((5 , a_coef.size, b_coef.size, lr_rate.size, max_iter+1))

test_idx = [i for i in range(args.start_id, args.start_id+args.num_s, 1)]
S_N = np.einsum('nji,njk->ik', train_data[test_idx], train_data[test_idx]) / (len(test_idx))
w, eigv = linalg.eigh(S_N)


w = w[::-1] #shape(D) v
eigv = eigv[:,::-1] #shape(D, D)v
S_T = np.einsum('ntk,ntj->tkj', train_data[test_idx], train_data[test_idx]) / len(test_idx) #shape(T,D,D) v
S_Tv = np.einsum('tkj,ji->tki', S_T, eigv) #shape(T,D,D) v
vtS_Tv = np.einsum('ki,tkj->tij', eigv, S_Tv) # shape(T,D,D) v
A = np.einsum('tii->ti', vtS_Tv) #v
    

synth_data.plot_weights(A.T[0:K,::], '../figures/{}_initial_weights.png'.format(args.title),title_append="Initial Weights")
C = np.einsum('ki,kj->kij', eigv.T, eigv.T)
synth_data.plot_components(C[0:K,:,:], '../figures/{}_initial_componts.png'.format(args.title),title_append="Initial Components")

for b in range(len(sparse_coef)):
    for a in range(len(smooth_coef)):
        for l in range(len(lr_rate)):

            tkst_test = snscov_model(K, T, D, la_rate = lr_rate[l],
                                    ld_rate= lr_rate[l]/T, kernel = kernel, smooth=smooth_coef[a],k_sparse=sparse_coef[b],
                                    max_iter = max_iter, a_method = 'temporal_kernel', d_method = 'sparse')
            tkst_test.fit(train_data[test_idx,:,:], evaluate=True)

            tkst_re_dict, tkst_re_alphas = (tkst_test.best_dictionary, tkst_test.best_alphas)

            print('taskfmri', smooth_coef[a], sparse_coef[b], lr_rate[l], tkst_test.best_error) 
    
            
            synth_data.plot_components(np.einsum('ki,kj->kij', tkst_re_dict, tkst_re_dict), 
                                                fname='../figures/{}_best_components_{}{}{}.png'.format(args.title, a,b,l), 
                                                title_append="Components \n $\lambda_1$={}, $\lambda_2$={}, lr={}".format(smooth_coef[a],sparse_coef[b], lr_rate[l]))
            
            synth_data.plot_weights_with_box((tkst_re_alphas ).T , motor_conv,
                                    fname='../figures/{}_best_weights_{}{}{}.png'.format(args.title, a,b, l), 
                                    title_append="Weights\n $\lambda_1$={}, $\lambda_2$={}, lr={}".format(smooth_coef[a], sparse_coef[b], lr_rate[l]))

            synth_data.plot_components(np.einsum('ki,kj->kij', tkst_test.dict, tkst_test.dict), 
                                                fname='../figures/{}_last_components_{}{}{}.png'.format(args.title, a,b,l), 
                                                title_append="Components \n $\lambda_1$={}, $\lambda_2$={}, lr={}".format(smooth_coef[a],sparse_coef[b], lr_rate[l]))
            synth_data.plot_weights_with_box((tkst_test.alphas).T[0:5,::] , motor_conv,
                                    fname='../figures/{}_last_weights_{}{}{}1.png'.format(args.title, a,b, l), 
                                    title_append="Weights\n $\lambda_1$={}, $\lambda_2$={}, lr={}".format(smooth_coef[a], sparse_coef[b], lr_rate[l])) 
            synth_data.plot_weights_with_box((tkst_test.alphas).T[5:10,::] , motor_conv,
                                    fname='../figures/{}_last_weights_{}{}{}2.png'.format(args.title, a,b, l), 
                                    title_append="Weights\n $\lambda_1$={}, $\lambda_2$={}, lr={}".format(smooth_coef[a], sparse_coef[b], lr_rate[l]))                         
            synth_data.plot_weights_with_box((tkst_test.alphas).T[10:15,::] , motor_conv,
                                    fname='../figures/{}_last_weights_{}{}{}3.png'.format(args.title, a,b, l), 
                                    title_append="Weights\n $\lambda_1$={}, $\lambda_2$={}, lr={}".format(smooth_coef[a], sparse_coef[b], lr_rate[l]))                         
            
            for i in range(4,K,1):
                synth_data.plot_weights_with_box((tkst_test.alphas).T[i,::][np.newaxis,:] , motor_conv,
                                    fname='../figures/{}_last_weights_ind{}{}{}{}.png'.format(args.title, a,b, l,i), 
                                    title_append="Weights\n $\lambda_1$={}, $\lambda_2$={}, lr={}".format(smooth_coef[a], sparse_coef[b], lr_rate[l])) 
   

            np.savez('../results/{}_{}{}{}.npz'.format(args.title, a, b, l), tkst_test.alphas, tkst_test.dict)

