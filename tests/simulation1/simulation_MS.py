##########################
#filename:simulation_MS.py
#description:
#   a. this program implements MS (spectral initialization only)
#   b. the result can be found in Table 4

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
import snscov.synth_data as synth_data
from  snscov.model import snscov_model
from snscov.kernel import Matern52Kernel
from snscov.evaluate import *
from mpl_toolkits.mplot3d import Axes3D
from snscov.methods import *

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
    amp=2.5*np.max(alphas)
elif args.waveform == 'square':
    alphas=synth_data.synthesize_switches2(K,T)
    amp=1.2*np.max(alphas)
elif args.waveform == 'mixing':
    alphas=synth_data.synthesize_weights(K,T)

    amp=2*np.max(alphas)



#np.random.seed(5)


#kernel = Matern52Kernel(T, 2, 5000)
kernel = Matern52Kernel(T, 2,20)
np.set_printoptions(precision=2)

kernel = linalg.solve(kernel, np.eye(T))




a_coef = np.array([0.001]) #[100]
b_coef = np.array([ 7])
lr_rate = np.array([1e-1])
max_iter = 500
tkst_metric = np.zeros((10, 2, a_coef.size, b_coef.size, lr_rate.size))
#distance_metric = np.zeros((10 , a_coef.size, b_coef.size, lr_rate.size, max_iter+1))
distance_metric = np.zeros((6,20))


for i in range (20):
    np.random.seed(i)
    sd = synth_data.synthesize_data_noiselevel(alphas, C, N, BETA_inv = .0)

    subjects = []
    number = [ 1, 5,15,200,1000]
    for q in number:
        title = "subject{}".format(q)
        subjects.append(title)
    print(subjects)
    for s in range(len(number)):

        subject = subjects[s]
            
        cov, est_dict, est_alphas = spectral_initialization(sd[0:number[s],:,:], K)
    

        
        print('dual permutation',dual_permutation_separate((alphas).T, v, est_alphas, est_dict.T))
        print('orthogonal', ortogonal_procrustes((alphas).T, v, est_alphas, est_dict.T))
        title=args.waveform+'_'+subject

        distance_metric[s,i],_,_=dual_permutation_separate((alphas).T, v, est_alphas, est_dict.T)

np.save('../../results/spectral_20_replica_{}_distance_metric.npy'.format(args.waveform), distance_metric)

print(np.mean(distance_metric, axis=1))
print(np.std(distance_metric, axis=1))
def remove_outlier(arr, a=2):
    mean = np.mean(arr)
    std = np.std(arr)
    keep = np.intersect1d(np.where(arr <= mean +a*std)[0], np.where(arr>= mean -a*std)[0])
    keep = np.array(keep)

    out = [arr[i] for i in keep]
    return out

for i in range(distance_metric.shape[0]):
    out = remove_outlier(distance_metric[i,:], a=1)
    print(np.mean(out),np.std(out))
    