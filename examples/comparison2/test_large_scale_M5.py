
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import time
import numpy as np
import snscov.synth_data as synth_data
from snscov.evaluate import *
from snscov.model_bayesian import BSL_ModelBuilder
import argparse
import tensorflow as tf
from tensorflow.python.client import device_lib

config = tf.ConfigProto()
config.gpu_options.allow_growth = True




tf.executing_eagerly()

import argparse

parser = argparse.ArgumentParser(description='Process some integers. ')
parser.add_argument('--num_subjects', type=int, default=10,         help='number of test subjects')
parser.add_argument('--num_trial',    type=int, default=10,        help='number of trials')
parser.add_argument('--title',        type=str, default='compare_large_scale', help='experiment name')
parser.add_argument('--noise_level',  type=int, default=2,       help='noise level'  )
parser.add_argument('--K', type=int, default=10, help='number of components')
parser.add_argument('--D', type=int, default=100, help='data dimension')
parser.add_argument('--T', type=int, default=100, help='number of time points')
parser.add_argument('--density', type=float, default=0.4, help='matrix sparsity')
parser.add_argument('--n_block', type=int,   default=4, help='number of blocks')
parser.add_argument('--kernel_length', type=int, default=5, help='length scale of the Matern five-half kernel')
parser.add_argument('--id',    type=int, default=0,        help='id')

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

np.random.seed(1246)
alphas = synth_data.generate_latent_splines(100, K, 1).T




for trial in range(args.num_trial):
    sd = synth_data.synthesize_data_noiselevel(alphas, C, N, BETA_inv = 1/args.noise_level)

    mb = BSL_ModelBuilder(sd, [N,T,D], K , np.eye(D), beta=args.noise_level, length_scale=args.kernel_length)
    train = mb.train()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        elbos = []
        start = time.time()
        for i in range(20000):
            elbos.append(sess.run(mb.elbo))

            sess.run(train)
            #if i>10 and np.abs(elbos[-1]-elbos[-2])<0.0001:
            #    break
            if i%1000==0:
                print(i/1000, elbos[-1])
        end = time.time()
        execution_time = end - start
        elbos.append(sess.run(mb.elbo))
        
        print(elbos[-1])
        print('excution_time:', execution_time)
        #plt.plot(elbos)
        #plt.savefig('../figures/elbo_{}_lv_{}.png'.format( args.num_subjects, args.noise_level))
        #plt.close()
        v_mean = sess.run(mb.qv_mean)
        v_var = sess.run(mb.qv_var)
        a_var = sess.run(mb.qa_var)
        a_mean = sess.run(mb.qa_mean)
        
    temp_covar = np.zeros(20)
    
    for j in range(20):
        v_sample = np.zeros(v_mean.shape)
        for iidx, m_v in enumerate(v_mean):
            v_sample[iidx] = np.random.multivariate_normal(m_v,np.diag(v_var[iidx]),1)    
        a_sample = np.zeros(a_mean.shape)
        for iidx, m_a in enumerate(a_mean):    
            a_sample[iidx] = np.random.multivariate_normal(m_a, np.diag(a_var[iidx]),1)
            

        est_C = np.einsum('ki,kj->kij', v_sample , v_sample)
        test_covar = np.einsum('tk,kij->tij', a_sample.T, est_C)
    
        temp_covar[j] = avg_LERM_cov(alphas.T , v, test_covar)
            

    print('avg LERM:', np.mean(temp_covar))

    np.savez('../../results/bayesian_high_{}_lv_{}_{}.npz'.format( args.num_subjects, args.noise_level, trial), a_mean=a_mean, a_var=a_var, v_mean=v_mean, v_var=v_var, ext_time=execution_time, avg_logerm=np.mean(temp_covar))
    print('model saved')






