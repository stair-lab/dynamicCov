##########################
#filename:simulation_linearplot.py
#description:
#   a. this program produces the linear convergence plot of Figure 2 in the manuscript
#   b. set init_method='random' in snscov_model allows random initialization. The result is shown in Figure 3 in the manuscript


import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
import snscov.synth_data as synth_data
from  snscov.model import snscov_model
from snscov.model_large import snscov_model as snscov_model_large
from snscov.kernel import Matern52Kernel
from snscov.evaluate import *
from mpl_toolkits.mplot3d import Axes3D

from matplotlib.gridspec import GridSpec
plt.rcParams.update({'font.size': 20})


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




alphas=synth_data.synthesize_weights(K,T)

amp=2*np.max(alphas)
#np.random.seed(5)


kernel = Matern52Kernel(T, 2, 200)
#kernel = Matern52Kernel(T, 2,40)
np.set_printoptions(precision=2)

kernel = linalg.solve(kernel, np.eye(T))

a_coef = np.array([0.001]) #[100]
 
lr_rate = np.array([1e-4])
max_iter = 200
tkst_metric = np.zeros((10, 2, a_coef.size, b_coef.size, lr_rate.size))
distance_metric = np.zeros((10 , a_coef.size, b_coef.size, lr_rate.size, max_iter+1))
#distance_metric = np.zeros((6,20))

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
    if legend[1]:
        fig.legend(handles, labels, ncol=4,  bbox_to_anchor=[0., -0.01], loc='lower left')
    if fname is not None:
        plt.savefig(fname,  bbox_inches='tight')
for i in range (1):
    np.random.seed(8)
    sd = synth_data.synthesize_data_noiselevel(alphas, C, N, BETA_inv = .0)

    subjects = []
    number = [ 1, 5, 15, 200]
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
                                                    dual_permutation((alphas).T, v, tkst_re_alphas, tkst_re_dict)])
                    
                    print('result: smoothness {}, sparse {},lr {}'.format(a_coef[a], b_coef[b], lr_rate[l]), tkst_metric[s,:,a,b,l]) 
                    #print('dual permutation',dual_permutation_separate((alphas).T, v, tkst_re_alphas, tkst_re_dict))
                    #print('orthogonal', ortogonal_procrustes((alphas).T, v, tkst_re_alphas, tkst_re_dict))
                    title=args.waveform+'_'+subject

                    distance_metric[s,i],_,_=dual_permutation_separate((alphas).T, v, tkst_re_alphas, tkst_re_dict)
                    #if i == 0:
                    dis,r_matrix = dual_permutation_matrix(alphas.T, v, tkst_re_alphas, tkst_re_dict)

                    #print(dis)
                    
                    otho_len = np.size(tkst_test.dual_ortho)
                    matrix = np.ones(max_iter+1) * -1
                    matrix[0:otho_len]=tkst_test.dual_ortho
                    distance_metric[s,a,b,l,:]=matrix


                
    
    
    fig = plt.figure(figsize=(16,6))
    ax = plt.subplot(121)
    marker_set=['p','v','s','*','+','H','D','1','2','3','4','8','','d']  
    color_set = ['blue','red','green','black', 'darkviolet']         
    for i in range(4):
        index = np.where(distance_metric[i].squeeze()>0.)[0][:-1][::2]
        ax.plot(index,distance_metric[i].squeeze()[index], color=color_set[i],label='N='+str(number[i]),marker=marker_set[i],markersize=8,  linewidth=3.)
    ax.legend(bbox_to_anchor=(2, -0.2),ncol=4, prop={'size': 22})
    ax.set_xlabel('iteration', fontsize=22)
    #plt.yticks([ 100, 120, 140, 160, 180])
    ax.set_yticks([0, 20, 40, 60, 80,100])
    ax.set_ylim(0,100)
    ax.set_xlim(0,30)
    ax.set_ylabel(r"$dist^2$(${{Z}}$,${{Z}}^\star$)", fontsize=22)
    ax.set_title("Convergence plot for mixing waveform", fontsize=22, y=1.05)
    ax.grid()


alphas=synth_data.synthesize_sines(K,T)
amp=2.5*np.max(alphas)


kernel = Matern52Kernel(T, 2, 200)
#kernel = Matern52Kernel(T, 2,40)
np.set_printoptions(precision=2)

kernel = linalg.solve(kernel, np.eye(T))

a_coef = np.array([0.001]) #[100]
 
lr_rate = np.array([1e-4])
max_iter = 100
tkst_metric = np.zeros((10, 2, a_coef.size, b_coef.size, lr_rate.size))
distance_metric = np.zeros((10 , a_coef.size, b_coef.size, lr_rate.size, max_iter+1))
#distance_metric = np.zeros((6,20))


for i in range (1):
    np.random.seed(8)
    sd = synth_data.synthesize_data_noiselevel(alphas, C, N, BETA_inv = .0)

    subjects = []
    number = [ 1, 5, 15, 200]
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
                                                    dual_permutation((alphas).T, v, tkst_re_alphas, tkst_re_dict)])
                    
                    print('result: smoothness {}, sparse {},lr {}'.format(a_coef[a], b_coef[b], lr_rate[l]), tkst_metric[s,:,a,b,l]) 

                    title=args.waveform+'_'+subject

                    distance_metric[s,i],_,_=dual_permutation_separate((alphas).T, v, tkst_re_alphas, tkst_re_dict)
                    #if i == 0:
                    dis,r_matrix = dual_permutation_matrix(alphas.T, v, tkst_re_alphas, tkst_re_dict)

                    
                    otho_len = np.size(tkst_test.dual_ortho)
                    matrix = np.ones(max_iter+1) * -1
                    matrix[0:otho_len]=tkst_test.dual_ortho
                    distance_metric[s,a,b,l,:]=matrix

    ax = plt.subplot(122)
        
    for i in range(4):
        index = np.where(distance_metric[i].squeeze()>0.)[0][:-1][::2]
        ax.plot(index,distance_metric[i].squeeze()[index], color=color_set[i],label='N='+str(number[i]),marker=marker_set[i],markersize=8,  linewidth=3.)

    ax.set_xlabel('iteration', fontsize=22)
    #plt.yticks([ 100, 120, 140, 160, 180])
    ax.set_yticks([0, 20,40,60, 80, 100, 120, 140, 160, 180])
    ax.set_ylim(0,180)
    ax.set_xlim(0,30)
    #ax.set_ylabel(r"$dist^2$(${{Z}}$,${{Z}}^\star$)", fontsize=22)
    ax.grid()

    ax.set_title("Convergence plot for sine waveform", fontsize=22, y=1.05)
    plt.savefig('../../figures/linear_convergence.png',bbox_inches = 'tight')
    plt.savefig('../../figures/linear_convergence.pdf',bbox_inches = 'tight')

                
    