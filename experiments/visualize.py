import numpy as np
import nilearn
import nilearn.plotting as plotting
import nibabel as nb
import scipy.io
import matplotlib.pyplot as plt
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
import snscov.synth_data as synth_data
from scipy.stats.stats import pearsonr   

img = nb.load("../data/Parcels_Combo.nii")
coords = plotting.find_parcellation_cut_coords(img)      

motor = scipy.io.loadmat('../data/HCP_motor/reg_motor_conv.mat')
motor_conv = motor["reg_motor_conv"]
motor_conv = motor_conv
label= ["Right Hand Tapping", "Left Foot Tapping", "Tongue Wagging", "Right Foot Tapping", "Left Hand Tapping"]


test_data = np.load("../results/tfmri_sub20_15_000.npz")

alphas= test_data['arr_0']
print(alphas.shape)
dictionary = test_data['arr_1']
smooth_coef=np.array([10])
sparse_coef=np.array([54])
lr_rate=np.array([1.0])
title="tfmri_5_020"
a=b=l=0

synth_data.plot_weights_with_box((alphas).T[[4,8,9],::] , motor_conv[:,[1,2,3]],
                    fname='../figures/2{}_last_weights_0.png'.format(title), 
                    title_append="Task Activation")

pear_all = []
for i in range(5):
    ref = np.array(np.where(motor_conv[:,i] !=0 )[0])
    reference = np.zeros(284)
    reference[ref] = 1

    pear_m = np.zeros(alphas.shape[1])
    for seq in range(0,10,1):
        pear_m[seq] = pearsonr(alphas[:,seq], reference)[0]
    pear_all.append(pear_m)
    index = np.argsort((pear_m))[::-1]
    print(index)

    synth_data.plot_weights_with_box((alphas).T[index[0],:][np.newaxis,:], motor_conv[:,i][:, np.newaxis],
                    fname='../figures/2{}_task_{}_best_component_{}{}.png'.format(title, i, index[0], index[1]), 
                    title_append="Task Activation: {}".format(label[i])) 
pear_all = (np.array(pear_all))

"""
best_id = np.array([np.argsort(pear_all[i])[-1] for i in range (5)])
for i in range(5):
    index = np.argsort(pear_all[i])[::-1]
    uni_id, counts = np.unique(best_id, return_counts=True)
    start_id = best_id[i]
    count = 0
    while (counts[np.where(uni_id==start_id)[0]]>=1):
        
        compare_id = np.where(best_id == start_id)[0]
        if(pear_all[i, start_id] == np.max(pear_all[compare_id, start_id])):
            break
        else:
            count +=1
            start_id = index[count]


    print('task:', i, 'component:', start_id)
    synth_data.plot_weights_with_box((alphas).T[start_id,::][np.newaxis, :] , motor_conv[:,i][:, np.newaxis],
                        fname='../figures/2{}_task_{}_best_component_{}.png'.format(title, i, start_id), 
                        title_append="Task Activation: {}".format(label[i]))     
"""
synth_data.plot_weights((alphas).T[0:5,::] , 
                        fname='../figures/2{}_last_weights_{}{}{}1.png'.format(title, a,b, l), 
                        title_append="Weights\n $\lambda_1$={}, $\lambda_2$={}, lr={}".format(smooth_coef[a], sparse_coef[b], lr_rate[l])) 
synth_data.plot_weights((alphas).T[4:10,::] , 
                        fname='../figures/2{}_last_weights_{}{}{}2.png'.format(title, a,b, l), 
                        title_append="Weights\n $\lambda_1$={}, $\lambda_2$={}, lr={}".format(smooth_coef[a], sparse_coef[b], lr_rate[l]))                         
synth_data.plot_weights((alphas).T[10:15,::] , 
                        fname='../figures/2{}_last_weights_{}{}{}3.png'.format(title, a,b, l), 
                        title_append="Weights\n $\lambda_1$={}, $\lambda_2$={}, lr={}".format(smooth_coef[a], sparse_coef[b], lr_rate[l])) 
synth_data.plot_weights((alphas).T ,
                        fname='../figures/2{}_last_weights_{}{}{}4.png'.format(title, a,b, l), 
                        title_append="Weights\n $\lambda_1$={}, $\lambda_2$={}, lr={}".format(smooth_coef[a], sparse_coef[b], lr_rate[l])) 

for i in range(0,15,1):
    synth_data.plot_weights_with_box((alphas).T[i,::][np.newaxis,:] , motor_conv,
                        fname='../figures/2{}_last_weights_imdifivual{}{}{}{}.png'.format(title, a,b, l,i), 
                        title_append="Weights\n $\lambda_1$={}, $\lambda_2$={}, lr={}".format(smooth_coef[a], sparse_coef[b], lr_rate[l])) 


component = np.einsum('xi,xj->xij', dictionary, dictionary)
#synth_data.plot_components()
print(component.shape)
q = np.unique(np.array(np.where(np.abs(dictionary)>=1e-5)))
#plot individual

for k in range(component.shape[0]):
    idx = np.where(np.abs(dictionary[k,:])>=1e-5)[0]
    new_coords = coords[idx,:]
    x,y = np.meshgrid(idx,idx)
    new_component = component[k,x,y] 
    plotting.plot_connectome(new_component , new_coords,node_size=6,
                        edge_threshold="99%", title="component {}".format(k))
    plt.savefig('../figures/component_{}.png'.format(k))
    plt.close()



#plot all
component = np.einsum('xi,xj->xij', dictionary, dictionary)
for i in range(motor_conv.shape[1]):
    active = np.where(motor_conv[:,i] != 0)[0]
    index = np.argsort(pear_all[i])[::-1]

    sep = [j for j in range(active.size-1) if active[j]+1 != active[j+1]]
    print(index[0:3])
    a_1 = alphas[active[0:sep[0]+1],:][:,index[0:3]]
    a_2 = alphas[active[sep[0]+1::],:][:, index[0:3]]
    b = np.vstack([a_1,a_2])
    for idx, a in enumerate([b]):

        q = np.unique(np.array(np.where(np.abs(dictionary[index[0:3],:])>=1e-5)))
        new_coords = coords[q,:]
        x,y = np.meshgrid(q,q)
        #normalize amplitude
        b = b /np.max(b,axis=0)[np.newaxis,:]
        new_component = np.squeeze(component[:,x,y][index[0:3]])
        
        #cov_1 = new_component
        cov_1 = np.einsum('ti,ijk->jk', [[1],[1],[1]],new_component)
        
        plotting.plot_connectome(cov_1 , new_coords,node_size=6, 
                            edge_threshold="97%",title="Task:{}".format(label[i]))
        plt.savefig('../figures/{}_.png'.format(label[i]))
        plt.close()
    



