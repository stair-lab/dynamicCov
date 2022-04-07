import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
import snscov.synth_data as synth_data
from  snscov.model import snscov_model
from snscov.model_large import snscov_model as snscov_model_large
from snscov.kernel import Matern52Kernel
from snscov.evaluate import *
from matplotlib.gridspec import GridSpec
plt.rcParams.update({'font.size': 22})




def plot(alphas, C, title_append, fname, legend=[True, True]):
    K, T = alphas.shape
    fig = plt.figure(figsize=(10, 5))
    width_ratio = [((K+1)//2)*1.5]
    width_ratio += [1 for i in range(int((K+1)/2))]
    gs = GridSpec(nrows=2, ncols=1+np.int(np.ceil(K/2)), width_ratios=width_ratio, hspace=0.4)
    ax0 =  fig.add_subplot(gs[:, 0])
    marker_set=['p','v','s','*','D','X','H','^','1','2','3','4','8','+','d'] 
    color_set = ['blue','red','green','black', 'yellow', 'purple', 'hotpink', 'sienna', 'orange', 'grey']
    for k in range(K):
        ax0.plot(alphas[k, :],marker=marker_set[k], color=color_set[k],label="k={}".format(str(k)))
    #ax0.legend(loc='upper right')
    ax0.set_xlabel('t')
    ax0.set_ylabel('amplitude')
    handles, labels = ax0.get_legend_handles_labels()
    #ax0.title(wtitle_append, fontsize=14)
    

    K, D, _D = C.shape
    vmin = np.min(C)
    vmax = np.max(C)

    for k in range(K):
        if k == 0:
            ax = fig.add_subplot(gs[0, 1])
        else:
            ax = fig.add_subplot(gs[k%2, (k//2)+1])
        pos = ax.imshow(C[k], cmap='jet')
        ax.set_axis_off()
        #ax.set_title("k={}".format(str(k)))
        ax.text(5, 25,"k={}".format(str(k)))
        pos.set_clim(vmin, vmax)
    
    fig.subplots_adjust(right=0.85, bottom=0.25, wspace=0.25, hspace=0.4)
    if legend[0]:
        cbar_ax = fig.add_axes([0.86, 0.25, 0.03, 0.6])
        fig.colorbar(pos, cax=cbar_ax)
    fig.suptitle(title_append)

    fig.legend(handles, labels, ncol=4,  bbox_to_anchor=[1., 0.1])
    if fname is not None:
        plt.savefig(fname,  bbox_inches='tight')

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
v = np.zeros((K,20))

v[0,0:5] = np.array([1, -1, 1, -1, -1])
v[0,9:10] = np.array([1])

v[1,5:9] = np.array([1,-1, -1, 1])

v[2,10:13] = np.array([1, 1, -1])

v[3,13:16] = np.array([1, -1,  1])
v[3,17:20] = np.array([-1, -1, -1])
b_coef = np.array([ 7])
  

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


kernel = Matern52Kernel(T, 2, 400)
#kernel = Matern52Kernel(T, 2,40)
np.set_printoptions(precision=2)

kernel = linalg.solve(kernel, np.eye(T))



plot(alphas, 
                np.einsum('ki,kj->kij', v, v), 
                title_append="True components of {} waveform".format(args.waveform), 
                fname='../../figures/true_weights_{}.pdf'.format(args.waveform),
                legend=[True, True])
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
    number = [1,5,15,200]
    for q in number:
        title = "subject{}".format(q)
        subjects.append(title)
    print(subjects)
    for s in range(len(number)):
        for b in range(len(b_coef)):
            for a in range(len(a_coef)):
                for l in range(len(lr_rate)):
                    subject = subjects[s]

                    tkst_test = snscov_model(K, T, D, la_rate = lr_rate[l], amp=amp,tol=1e-3,
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
                                                    large_scale_signed_permutation((alphas).T, v, tkst_re_alphas, tkst_re_dict)])
                    
                    print('result: smoothness {}, sparse {},lr {}'.format(a_coef[a], b_coef[b], lr_rate[l]), tkst_metric[s,:,a,b,l]) 

                    title=args.waveform+'_'+subject


                    dis,r_matrix = dual_permutation_matrix(alphas.T, v, tkst_re_alphas, tkst_re_dict)
                    
                    #print(dis)
                    if number[s] == 15:
                        plot((tkst_re_alphas @ r_matrix ).T, 
                            np.einsum('ki,kj->kij',  r_matrix.T @ tkst_re_dict, r_matrix.T @ tkst_re_dict ), 
                            title_append="Estimated components of {} waveform  (subjects={})".format(args.waveform, number[s]), 
                            fname='../../figures/estimated_components_{}.pdf'.format(args.waveform),
                            legend=[True, True])     





#


