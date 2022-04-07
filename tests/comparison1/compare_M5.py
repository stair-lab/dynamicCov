
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




parser = argparse.ArgumentParser(description='Process some integers. ')
parser.add_argument('--num_subjects', type=int, default=10,       help='number of test subjects')
parser.add_argument('--waveform',     type=str, default='square', help='select waveform')
parser.add_argument('--noise_level',  type=int, default=100,       help='noise level'  )
parser.add_argument('--id',    type=int, default=0,        help='id')
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

if args.waveform == 'sine':
    alphas=synth_data.synthesize_sines(K,T)
elif args.waveform == 'square':
    alphas=synth_data.synthesize_switches(K,T)
elif args.waveform == 'mixing':
    alphas=synth_data.synthesize_weights(K,T)

sd = synth_data.synthesize_data_noiselevel(alphas, C, N, 1./args.noise_level)


mb = BSL_ModelBuilder(sd, [N,T,D], K , np.eye(D), beta=args.noise_level)
train = mb.train()

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    elbos = []
    start = time.time()
    for i in range(10000):
        elbos.append(sess.run(mb.elbo))

        sess.run(train)
        #if i>10 and np.abs(elbos[-1]-elbos[-2])<0.0001:
        #    break
        if i%1000==0:
            print(i/1000, elbos[-1])
    end = time.time()
    execution_time = end - start
    elbos.append(sess.run(mb.elbo))
    
    #print(elbos[-1])
    #print('excution_time:', execution_time)
    #plt.plot(elbos)
    #plt.savefig('../figures/elbo_{}_lv_{}.png'.format(args.waveform, args.num_subjects, args.noise_level))
    #plt.close()
    vv = sess.run(mb.qv_mean)
    v_var = sess.run(mb.qv_var)
    a_var = sess.run(mb.qa_var)
    a = sess.run(mb.qa_mean)
    
    #synth_data.plot_components(np.einsum('ki,kj->kij',vv,vv), fname='../figures/bayesian_{}_{}_lv_{}_component.png'.format(args.waveform, args.num_subjects, args.noise_level), 
    #                            title_append="Learned Components")
    #synth_data. plot_weights(a,fname='../figures/bayesian_{}_{}_lv_{}_weight.png'.format(args.waveform, args.num_subjects, args.noise_level), 
    #                        title_append="Learned Weights")
   

    
    #m = sess.run(mb.m)
    #length_scale = sess.run(mb.length_scale)
    #amplitude=sess.run(mb.amplitude)

    np.savez('../../results/bayesian_{}_{}_lv_{}_{}.npz'.format(args.waveform, args.num_subjects, args.noise_level,args.id),
                                                          a_mean=a, a_var=a_var, v_mean=vv, v_var=v_var, ext_time=execution_time)

   





