
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
from matplotlib.gridspec import GridSpec
plt.rcParams.update({'font.size': 22})
from matplotlib.ticker import FormatStrFormatter

def plot2figK10(alphas, C, estalphas, estC, alphas1, C1, estalphas1, estC1, fname):
    title_loc = 1.8
    x_loc = 5
    y_loc = 25
    K, T = alphas.shape
    #fig = plt.figure(figsize=(20, 10))
    width_ratio = [5*1.5]
    width_ratio += [2.25 for i in range(5)]
    #width_ratio += width_ratio 
    gs_kw = dict(width_ratios=width_ratio, height_ratios=[3,3,1,3,3,1,3,3,1,3,3])
    
    fig, axd = plt.subplot_mosaic([[ 'l1',  'l2',  'l4', 'l26', 'l28', 'l50'],
                                   [ 'l1',  'l3',  'l5', 'l27', 'l29', 'l51'],
                                    ['w',    'w',  'w',  'w',   'w',    'w'],
                                    [ 'l6',  'l7', 'l9', 'l31', 'l33', 'l55'],
                                    [ 'l6',  'l8', 'l10','l32', 'l34', 'l56'],
                                    ['w1',   'w1', 'w1', 'w1',  'w1',  'w1'],
                                    ['l11', 'l12', 'l14','l36', 'l38', 'l60'],
                                    ['l11', 'l13', 'l15','l37', 'l39', 'l61'],
                                    ['w2',   'w2', 'w2', 'w2',  'w2',  'w2'],
                                    ['l16', 'l17', 'l19','l41', 'l43', 'l65'],
                                    ['l16', 'l18', 'l20','l42', 'l44', 'l66']],
                              gridspec_kw=gs_kw, figsize=(15, 20)
                              )
    #plot the true mixing waveform

    axd['w'].remove()
    axd['w1'].remove()
    axd['w2'].remove()
    marker_set=['p','v','s','*','D','X','H','^','1','2','3','4','8','+','d'] 
    color_set = ['blue','red','green','black', 'yellow', 'purple', 'hotpink', 'sienna', 'orange', 'grey']
    for k in range(K):
        axd['l1'].plot(alphas[k, :],marker=marker_set[k], color=color_set[k],label="k={}".format(str(k)))
    #ax0.set_xlabel('t')
    axd['l1'].set_ylabel('amplitude')
    axd['l1'].text(0.5,title_loc,'True components')
    axd['l1'].yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    handles, labels = axd['l1'].get_legend_handles_labels()

    K, D, _D = C.shape
    vmin = np.min(C)
    vmax = np.max(C)

    for k in range(K):
        id = "l"+str(k+2)
        if k >= 4:
            id = "l"+str(k+2+20)  
        if k >= 8:
            id = "l"+str(k+2+20+20)   
        pos = axd[id].imshow(C[k], cmap='jet')
        axd[id].set_axis_off()
        #ax.set_title("k={}".format(str(k)))
        axd[id].text(x_loc, y_loc,"k={}".format(str(k)))
        pos.set_clim(vmin, vmax)
    start = (K//2)+1 


    #plot the true sine waveform
    for k in range(K):
        axd['l11'].plot(alphas1[k, :],marker=marker_set[k], color=color_set[k],label=str(k))
    #axd['l11'].set_xlabel('t')
    axd['l11'].set_ylabel('amplitude')
    axd['l11'].get_shared_y_axes().join(axd['l1'], axd['l11'])
    axd['l11'].yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    axd['l11'].text(0.5,title_loc,'Estimated components (subjects=200)')
    

    K, D, _D = C.shape
    vmin = np.min(C1)
    vmax = np.max(C1)

    for k in range(K):
        id = "l"+str(k+12) 
        if k >= 4:
            id = "l"+str(k+12+20) 
        if k >= 8:
            id = "l"+str(k+12+20+20) 
        pos = axd[id].imshow(C1[k], cmap='jet')
        axd[id].set_axis_off()
        #ax.set_title("k={}".format(str(k)))
        axd[id].text(x_loc, y_loc,"k={}".format(str(k)))
        pos.set_clim(vmin, vmax)
    start = (K//2)+1 
    
    
    #estimated mixing graph

 
    for k in range(K):
        axd['l6'].plot(estalphas[k, :],marker=marker_set[k], color=color_set[k],label="k={}".format(str(k)))
    #ax2.set_xlabel('t')
    axd['l6'].text(0.5,title_loc,'Estimated components (subjects=20)')
    axd['l6'].get_shared_y_axes().join(axd['l1'], axd['l6'])
    axd['l6'].set_ylabel('amplitude')
    axd['l6'].yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))

    #axd['l6'].set_yticklabels([])

    for k in range(K):
        id = "l"+str(k+7) 
        if k >= 4:
            id = "l"+str(k+7+20) 
        if k >= 8:
            id = "l"+str(k+7+20+20)             
        pos = axd[id].imshow(estC[k], cmap='jet')
        axd[id].set_axis_off()
        #ax.set_title("k={}".format(str(k)))
        axd[id].text(x_loc, y_loc,"k={}".format(str(k)))
        pos.set_clim(vmin, vmax)

    #estimated sine graph

 
    for k in range(K):
        axd['l16'].plot(estalphas1[k, :],marker=marker_set[k], color=color_set[k],label="k={}".format(str(k)))
    axd['l16'].set_xlabel('t')
    axd['l16'].text(0.5,title_loc,'Estimated components (subjects=2000)')
    axd['l16'].get_shared_y_axes().join(axd['l1'], axd['l16'])
    axd['l16'].set_ylabel('amplitude')
    
    axd['l16'].yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    #axd['l11'].get_shared_y_axes().join(axd['l1'], axd['l16'])
    #axd['l16'].set_yticklabels([])

    for k in range(K):
        id = "l"+str(k+17)
        if k >=4:
            id = "l"+str(k+17+20)
        if k >=8:
            id = "l"+str(k+17+20+20)
        pos = axd[id].imshow(estC1[k], cmap='jet')
        axd[id].set_axis_off()
        #ax.set_title("k={}".format(str(k)))
        axd[id].text(x_loc, y_loc,"k={}".format(str(k)))
        pos.set_clim(vmin, vmax)
    axd['l16'].legend(handles, labels, ncol=5, loc='upper center', bbox_to_anchor=(1.5, -0.3))
    fig.subplots_adjust(right=0.85, bottom=0.2, wspace=0.25, hspace=0.5)

    cbar_ax = fig.add_axes([0.88, 0.4, 0.02, 0.2])
    fig.colorbar(pos, cax=cbar_ax)
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight')

def plot2figK8(alphas, C, estalphas, estC, alphas1, C1, estalphas1, estC1, fname):
    title_loc = 1.8
    x_loc = 5
    y_loc = 25
    K, T = alphas.shape
    #fig = plt.figure(figsize=(20, 10))
    width_ratio = [4*1.5]
    width_ratio += [2.25 for i in range(4)]
    #width_ratio += width_ratio 
    gs_kw = dict(width_ratios=width_ratio, height_ratios=[3,3,1,3,3,1,3,3,1,3,3])
    
    fig, axd = plt.subplot_mosaic([[ 'l1',  'l2',  'l4', 'l26', 'l28'],
                                   [ 'l1',  'l3',  'l5', 'l27', 'l29'],
                                    ['w',    'w',  'w',  'w',   'w'],
                                    [ 'l6',  'l7', 'l9', 'l31', 'l33'],
                                    [ 'l6',  'l8', 'l10','l32', 'l34'],
                                    ['w1',   'w1', 'w1', 'w1',  'w1'],
                                    ['l11', 'l12', 'l14','l36', 'l38'],
                                    ['l11', 'l13', 'l15','l37', 'l39'],
                                    ['w2',   'w2', 'w2', 'w2',  'w2'],
                                    ['l16', 'l17', 'l19','l41', 'l43'],
                                    ['l16', 'l18', 'l20','l42', 'l44']],
                                #    ['l11', 'l13', 'l16', 'l18']],
                              gridspec_kw=gs_kw, figsize=(15, 20)
                              )
    #plot the true mixing waveform

    axd['w'].remove()
    axd['w1'].remove()
    axd['w2'].remove()
    marker_set=['p','v','s','*','D','X','H','^','1','2','3','4','8','+','d'] 
    color_set = ['blue','red','green','black', 'yellow', 'purple', 'hotpink', 'sienna', 'orange', 'grey']
    for k in range(K):
        axd['l1'].plot(alphas[k, :],marker=marker_set[k], color=color_set[k],label="k={}".format(str(k)))
    #ax0.set_xlabel('t')
    axd['l1'].set_ylabel('amplitude')
    axd['l1'].text(0.5,title_loc,'True components')
    axd['l1'].yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    handles, labels = axd['l1'].get_legend_handles_labels()

    K, D, _D = C.shape
    vmin = np.min(C)
    vmax = np.max(C)

    for k in range(K):
        id = "l"+str(k+2)
        if k >= 4:
            id = "l"+str(k+2+20)   
        pos = axd[id].imshow(C[k], cmap='jet')
        axd[id].set_axis_off()
        #ax.set_title("k={}".format(str(k)))
        axd[id].text(x_loc, y_loc,"k={}".format(str(k)))
        pos.set_clim(vmin, vmax)
    start = (K//2)+1 


    #plot the true sine waveform
    for k in range(K):
        axd['l11'].plot(alphas1[k, :],marker=marker_set[k], color=color_set[k],label=str(k))
    #axd['l11'].set_xlabel('t')
    axd['l11'].set_ylabel('amplitude')
    axd['l11'].get_shared_y_axes().join(axd['l1'], axd['l11'])
    axd['l11'].yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    axd['l11'].text(0.5,title_loc,'Estimated components (subjects=200)')
    

    K, D, _D = C.shape
    vmin = np.min(C1)
    vmax = np.max(C1)

    for k in range(K):
        id = "l"+str(k+12) 
        if k >= 4:
            id = "l"+str(k+12+20)   
        pos = axd[id].imshow(C1[k], cmap='jet')
        axd[id].set_axis_off()
        #ax.set_title("k={}".format(str(k)))
        axd[id].text(x_loc, y_loc,"k={}".format(str(k)))
        pos.set_clim(vmin, vmax)
    start = (K//2)+1 
    
    
    #estimated mixing graph

 
    for k in range(K):
        axd['l6'].plot(estalphas[k, :],marker=marker_set[k], color=color_set[k],label="k={}".format(str(k)))
    #ax2.set_xlabel('t')
    axd['l6'].text(0.5,title_loc,'Estimated components (subjects=20)')
    axd['l6'].get_shared_y_axes().join(axd['l1'], axd['l6'])
    axd['l6'].set_ylabel('amplitude')
    axd['l6'].yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))

    #axd['l6'].set_yticklabels([])

    for k in range(K):
        id = "l"+str(k+7) 
        if k >= 4:
            id = "l"+str(k+7+20) 
        pos = axd[id].imshow(estC[k], cmap='jet')
        axd[id].set_axis_off()
        #ax.set_title("k={}".format(str(k)))
        axd[id].text(x_loc, y_loc,"k={}".format(str(k)))
        pos.set_clim(vmin, vmax)

    #estimated sine graph

 
    for k in range(K):
        axd['l16'].plot(estalphas1[k, :],marker=marker_set[k], color=color_set[k],label="k={}".format(str(k)))
    axd['l16'].set_xlabel('t')
    axd['l16'].text(0.5,title_loc,'Estimated components (subjects=2000)')
    axd['l16'].get_shared_y_axes().join(axd['l1'], axd['l16'])
    axd['l16'].set_ylabel('amplitude')
    
    axd['l16'].yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    #axd['l11'].get_shared_y_axes().join(axd['l1'], axd['l16'])
    #axd['l16'].set_yticklabels([])

    for k in range(K):
        id = "l"+str(k+17)
        if k >=4:
            id = "l"+str(k+17+20)
        pos = axd[id].imshow(estC1[k], cmap='jet')
        axd[id].set_axis_off()
        #ax.set_title("k={}".format(str(k)))
        axd[id].text(x_loc, y_loc,"k={}".format(str(k)))
        pos.set_clim(vmin, vmax)
    axd['l16'].legend(handles, labels, ncol=4, loc='upper center', bbox_to_anchor=(1.5, -0.3))
    fig.subplots_adjust(right=0.85, bottom=0.2, wspace=0.25, hspace=0.5)

    cbar_ax = fig.add_axes([0.88, 0.4, 0.02, 0.2])
    fig.colorbar(pos, cax=cbar_ax)
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight')
def plot2figK6(alphas, C, estalphas, estC, alphas1, C1, estalphas1, estC1, fname):
    title_loc = 1.7
    x_loc = 3
    y_loc = 27
    K, T = alphas.shape
    #fig = plt.figure(figsize=(20, 10))
    width_ratio = [3*1.5]
    width_ratio += [1 for i in range(3)]
    #width_ratio += width_ratio 
    gs_kw = dict(width_ratios=width_ratio, height_ratios=[3,3,1,3,3,1,3,3,1,3,3])
    
    fig, axd = plt.subplot_mosaic([[ 'l1',  'l2',  'l4', 'l26'],
                                   [ 'l1',  'l3',  'l5', 'l27'],
                                    ['w',    'w',  'w',  'w'],
                                    [ 'l6',  'l7', 'l9', 'l31'],
                                    [ 'l6',  'l8', 'l10','l32'],
                                    ['w1',   'w1', 'w1', 'w1'],
                                    ['l11', 'l12', 'l14','l36'],
                                    ['l11', 'l13', 'l15','l37'],
                                    ['w2',   'w2', 'w2', 'w2'],
                                    ['l16', 'l17', 'l19','l41'],
                                    ['l16', 'l18', 'l20','l42']],
                                #    ['l11', 'l13', 'l16', 'l18']],
                              gridspec_kw=gs_kw, figsize=(10, 20)
                              )
    #plot the true mixing waveform

    axd['w'].remove()
    axd['w1'].remove()
    axd['w2'].remove()
    marker_set=['p','v','s','*','D','X','H','^','1','2','3','4','8','+','d'] 
    color_set = ['blue','red','green','black', 'yellow', 'purple', 'hotpink', 'sienna', 'orange', 'grey']
    for k in range(K):
        axd['l1'].plot(alphas[k, :],marker=marker_set[k], color=color_set[k],label="k={}".format(str(k)))
    #ax0.set_xlabel('t')
    axd['l1'].set_ylabel('amplitude')
    axd['l1'].text(0.5,title_loc,'True components')
    axd['l1'].yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    handles, labels = axd['l1'].get_legend_handles_labels()

    K, D, _D = C.shape
    vmin = np.min(C)
    vmax = np.max(C)

    for k in range(K):
        id = "l"+str(k+2)
        if k >= 4:
            id = "l"+str(k+2+20)   
        pos = axd[id].imshow(C[k], cmap='jet')
        axd[id].set_axis_off()
        #ax.set_title("k={}".format(str(k)))
        axd[id].text(x_loc, y_loc,"k={}".format(str(k)))
        pos.set_clim(vmin, vmax)
    start = (K//2)+1 


    #plot the true sine waveform
    for k in range(K):
        axd['l11'].plot(alphas1[k, :],marker=marker_set[k], color=color_set[k],label=str(k))
    #axd['l11'].set_xlabel('t')
    axd['l11'].set_ylabel('amplitude')
    axd['l11'].get_shared_y_axes().join(axd['l1'], axd['l11'])
    axd['l11'].yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    axd['l11'].text(0.5,title_loc,'Estimated components (subjects=200)')
    

    K, D, _D = C.shape
    vmin = np.min(C1)
    vmax = np.max(C1)

    for k in range(K):
        id = "l"+str(k+12) 
        if k >= 4:
            id = "l"+str(k+12+20)   
        pos = axd[id].imshow(C1[k], cmap='jet')
        axd[id].set_axis_off()
        #ax.set_title("k={}".format(str(k)))
        axd[id].text(x_loc, y_loc,"k={}".format(str(k)))
        pos.set_clim(vmin, vmax)
    start = (K//2)+1 
    
    
    #estimated mixing graph

 
    for k in range(K):
        axd['l6'].plot(estalphas[k, :],marker=marker_set[k], color=color_set[k],label="k={}".format(str(k)))
    #ax2.set_xlabel('t')
    axd['l6'].text(0.5,title_loc,'Estimated components (subjects=20)')
    axd['l6'].get_shared_y_axes().join(axd['l1'], axd['l6'])
    axd['l6'].set_ylabel('amplitude')
    axd['l6'].yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))

    #axd['l6'].set_yticklabels([])

    for k in range(K):
        id = "l"+str(k+7) 
        if k >= 4:
            id = "l"+str(k+7+20) 
        pos = axd[id].imshow(estC[k], cmap='jet')
        axd[id].set_axis_off()
        #ax.set_title("k={}".format(str(k)))
        axd[id].text(x_loc, y_loc,"k={}".format(str(k)))
        pos.set_clim(vmin, vmax)

    #estimated sine graph

 
    for k in range(K):
        axd['l16'].plot(estalphas1[k, :],marker=marker_set[k], color=color_set[k],label="k={}".format(str(k)))
    axd['l16'].set_xlabel('t')
    axd['l16'].text(0.5,title_loc,'Estimated components (subjects=2000)')
    axd['l16'].get_shared_y_axes().join(axd['l1'], axd['l16'])
    axd['l16'].set_ylabel('amplitude')
    
    axd['l16'].yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    #axd['l11'].get_shared_y_axes().join(axd['l1'], axd['l16'])
    #axd['l16'].set_yticklabels([])

    for k in range(K):
        id = "l"+str(k+17)
        if k >=4:
            id = "l"+str(k+17+20)
        pos = axd[id].imshow(estC1[k], cmap='jet')
        axd[id].set_axis_off()
        #ax.set_title("k={}".format(str(k)))
        axd[id].text(x_loc, y_loc,"k={}".format(str(k)))
        pos.set_clim(vmin, vmax)
    axd['l16'].legend(handles, labels, ncol=3, loc='upper center', bbox_to_anchor=(1, -0.3))
    fig.subplots_adjust(right=0.85, bottom=0.2, wspace=0.25, hspace=0.2)

    cbar_ax = fig.add_axes([0.88, 0.4, 0.02, 0.2])
    fig.colorbar(pos, cax=cbar_ax)
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight')
def plot2figK2(alphas, C, estalphas, estC, alphas1, C1, estalphas1, estC1, fname):
    K, T = alphas.shape
    #fig = plt.figure(figsize=(20, 10))
    width_ratio = [(2)*1.5]
    width_ratio += [1 for i in range(2)]
    #width_ratio += width_ratio 
    gs_kw = dict(width_ratios=width_ratio, height_ratios=[3,1,3,1,3,1,3])
    
    fig, axd = plt.subplot_mosaic([[ 'l1',  'l2', 'l3'],
                                    ['w',   'w', 'w'],
                                    [ 'l6',  'l7', 'l8'],
                                    ['w1',   'w1', 'w1'],
                                    ['l11', 'l12', 'l13'],
                                    ['w2',   'w2', 'w2'],
                                    ['l16', 'l17', 'l18']],
                                #    ['l11', 'l13', 'l16', 'l18']],
                              gridspec_kw=gs_kw, figsize=(10, 20)
                              )
    #plot the true mixing waveform

    axd['w'].remove()
    axd['w1'].remove()
    axd['w2'].remove()
    marker_set=['p','v','s','*','D','X','H','^','1','2','3','4','8','+','d'] 
    color_set = ['blue','red','green','black', 'yellow', 'purple', 'hotpink', 'sienna', 'orange', 'grey']
    for k in range(K):
        axd['l1'].plot(alphas[k, :],marker=marker_set[k], color=color_set[k],label="k={}".format(str(k)))
    #ax0.set_xlabel('t')
    axd['l1'].set_ylabel('amplitude')
    axd['l1'].text(0.5,1.1,'True components')
    axd['l1'].yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    handles, labels = axd['l1'].get_legend_handles_labels()

    K, D, _D = C.shape
    vmin = np.min(C)
    vmax = np.max(C)

    for k in range(K):
        id = "l"+str(k+2)
        pos = axd[id].imshow(C[k], cmap='jet')
        axd[id].set_axis_off()
        #ax.set_title("k={}".format(str(k)))
        axd[id].text(5, 25,"k={}".format(str(k)))
        pos.set_clim(vmin, vmax)
    start = (K//2)+1 


    #plot the true sine waveform
    for k in range(K):
        axd['l11'].plot(alphas1[k, :],marker=marker_set[k], color=color_set[k],label=str(k))
    #axd['l11'].set_xlabel('t')
    axd['l11'].set_ylabel('amplitude')
    axd['l11'].get_shared_y_axes().join(axd['l1'], axd['l11'])
    axd['l11'].yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    axd['l11'].text(0.5,1.1,'Estimated components (subjects=200)')
    

    K, D, _D = C.shape
    vmin = np.min(C1)
    vmax = np.max(C1)

    for k in range(K):
        id = "l"+str(k+12) 
        pos = axd[id].imshow(C1[k], cmap='jet')
        axd[id].set_axis_off()
        #ax.set_title("k={}".format(str(k)))
        axd[id].text(5, 25,"k={}".format(str(k)))
        pos.set_clim(vmin, vmax)
    start = (K//2)+1 
    
    
    #estimated mixing graph

 
    for k in range(K):
        axd['l6'].plot(estalphas[k, :],marker=marker_set[k], color=color_set[k],label="k={}".format(str(k)))
    #ax2.set_xlabel('t')
    axd['l6'].text(0.5,1.1,'Estimated components (subjects=20)')
    axd['l6'].get_shared_y_axes().join(axd['l1'], axd['l6'])
    axd['l6'].set_ylabel('amplitude')
    axd['l6'].yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))

    #axd['l6'].set_yticklabels([])

    for k in range(K):
        id = "l"+str(k+7) 
        pos = axd[id].imshow(estC[k], cmap='jet')
        axd[id].set_axis_off()
        #ax.set_title("k={}".format(str(k)))
        axd[id].text(5, 25,"k={}".format(str(k)))
        pos.set_clim(vmin, vmax)

    #estimated sine graph

 
    for k in range(K):
        axd['l16'].plot(estalphas1[k, :],marker=marker_set[k], color=color_set[k],label="k={}".format(str(k)))
    axd['l16'].set_xlabel('t')
    axd['l16'].text(0.5,1.1,'Estimated components (subjects=2000)')
    axd['l16'].get_shared_y_axes().join(axd['l1'], axd['l16'])
    axd['l16'].set_ylabel('amplitude')
    
    axd['l16'].yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    #axd['l11'].get_shared_y_axes().join(axd['l1'], axd['l16'])
    #axd['l16'].set_yticklabels([])

    for k in range(K):
        id = "l"+str(k+17)
        pos = axd[id].imshow(estC1[k], cmap='jet')
        axd[id].set_axis_off()
        #ax.set_title("k={}".format(str(k)))
        axd[id].text(5, 25,"k={}".format(str(k)))
        pos.set_clim(vmin, vmax)
    axd['l16'].legend(handles, labels, ncol=4, loc='upper center', bbox_to_anchor=(1, -0.3))
    fig.subplots_adjust(right=0.85, bottom=0.2, wspace=0.25, hspace=0.2)

    cbar_ax = fig.add_axes([0.88, 0.4, 0.02, 0.2])
    fig.colorbar(pos, cax=cbar_ax)
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight')



def plot2fig(alphas, C, estalphas, estC, alphas1, C1, estalphas1, estC1, fname):
    K, T = alphas.shape
    #fig = plt.figure(figsize=(20, 10))
    width_ratio = [((K+1)//2)*1.5]
    width_ratio += [1 for i in range(int((K+1)/2))]
    width_ratio += width_ratio 
    gs_kw = dict(width_ratios=width_ratio, height_ratios=[3,3,1,3,3])
    
    fig, axd = plt.subplot_mosaic([[ 'l1',  'l2',  'l4',  'l6',  'l7', 'l9'],
                                    ['l1',  'l3',  'l5',  'l6',  'l8', 'l10'],
                                    ['w',   'w',   'w',   'w',    'w',  'w'],
                                    ['l11', 'l12', 'l14', 'l16', 'l17', 'l19'],
                                    ['l11', 'l13', 'l15', 'l16', 'l18', 'l20']],
                              gridspec_kw=gs_kw, figsize=(20, 10)
                              )
    #plot the true mixing waveform

    axd['w'].remove()
    marker_set=['p','v','s','*','D','X','H','^','1','2','3','4','8','+','d'] 
    color_set = ['blue','red','green','black', 'yellow', 'purple', 'hotpink', 'sienna', 'orange', 'grey']
    for k in range(K):
        axd['l1'].plot(alphas[k, :],marker=marker_set[k], color=color_set[k],label="k={}".format(str(k)))
    #ax0.set_xlabel('t')
    axd['l1'].set_ylabel('amplitude')
    axd['l1'].text(0.5,1.5,'True components of mixing waveform')
    axd['l1'].yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    handles, labels = axd['l1'].get_legend_handles_labels()

    K, D, _D = C.shape
    vmin = np.min(C)
    vmax = np.max(C)

    for k in range(K):
        id = "l"+str(k+2)
        pos = axd[id].imshow(C[k], cmap='jet')
        axd[id].set_axis_off()
        #ax.set_title("k={}".format(str(k)))
        axd[id].text(5, 25,"k={}".format(str(k)))
        pos.set_clim(vmin, vmax)
    start = (K//2)+1 


    #plot the true sine waveform
    for k in range(K):
        axd['l11'].plot(alphas1[k, :],marker=marker_set[k], color=color_set[k],label=str(k))
    axd['l11'].set_xlabel('t')
    axd['l11'].set_ylabel('amplitude')
    axd['l11'].yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    axd['l11'].text(0.5,2.75,'True components of sine waveform')
    

    K, D, _D = C.shape
    vmin = np.min(C1)
    vmax = np.max(C1)

    for k in range(K):
        id = "l"+str(k+12) 
        pos = axd[id].imshow(C1[k], cmap='jet')
        axd[id].set_axis_off()
        #ax.set_title("k={}".format(str(k)))
        axd[id].text(5, 25,"k={}".format(str(k)))
        pos.set_clim(vmin, vmax)
    start = (K//2)+1 
    
    
    #estimated mixing graph

 
    for k in range(K):
        axd['l6'].plot(estalphas[k, :],marker=marker_set[k], color=color_set[k],label="k={}".format(str(k)))
    #ax2.set_xlabel('t')
    axd['l6'].text(0.5,1.5,'Estimated components (subjects=15)')

    axd['l1'].get_shared_y_axes().join(axd['l6'], axd['l1'])
    axd['l6'].set_yticklabels([])

    for k in range(K):
        id = "l"+str(k+7) 
        pos = axd[id].imshow(estC[k], cmap='jet')
        axd[id].set_axis_off()
        #ax.set_title("k={}".format(str(k)))
        axd[id].text(5, 25,"k={}".format(str(k)))
        pos.set_clim(vmin, vmax)

    #estimated sine graph

 
    for k in range(K):
        axd['l16'].plot(estalphas1[k, :],marker=marker_set[k], color=color_set[k],label="k={}".format(str(k)))
    axd['l16'].set_xlabel('t')
    axd['l16'].text(0.5,2.75,'Estimated components (subjects=15)')

    axd['l11'].get_shared_y_axes().join(axd['l11'], axd['l16'])
    axd['l16'].set_yticklabels([])

    for k in range(K):
        id = "l"+str(k+17)
        pos = axd[id].imshow(estC1[k], cmap='jet')
        axd[id].set_axis_off()
        #ax.set_title("k={}".format(str(k)))
        axd[id].text(5, 25,"k={}".format(str(k)))
        pos.set_clim(vmin, vmax)
    axd['l11'].legend(handles, labels, ncol=4, loc='upper center', bbox_to_anchor=(2, -0.25))
    fig.subplots_adjust(right=0.85, bottom=0.2, wspace=0.25)

    cbar_ax = fig.add_axes([0.86, 0.25, 0.02, 0.6])
    fig.colorbar(pos, cax=cbar_ax)
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight')

def plot2fig_ver0(alphas, C, estalphas, estC, alphas1, C1, estalphas1, estC1, fname):
    K, T = alphas.shape
    fig = plt.figure(figsize=(20, 10))
    width_ratio = [((K+1)//2)*1.5]
    width_ratio += [1 for i in range(int((K+1)/2))]
    width_ratio += width_ratio 
    gs = GridSpec(nrows=4, ncols=2*(1+np.int(np.ceil(K/2))), width_ratios=width_ratio, hspace=.2)
    gs_hspace = GridSpec(nrows=4, ncols=2*(1+np.int(np.ceil(K/2))), width_ratios=width_ratio, hspace=.2)
    #plot the true mixing waveform
    ax0 =  fig.add_subplot(gs[0:2, 0])
    marker_set=['p','v','s','*','D','X','H','^','1','2','3','4','8','+','d'] 
    color_set = ['blue','red','green','black', 'yellow', 'purple', 'hotpink', 'sienna', 'orange', 'grey']
    for k in range(K):
        ax0.plot(alphas[k, :],marker=marker_set[k], color=color_set[k],label="k={}".format(str(k)))
    #ax0.set_xlabel('t')
    ax0.set_ylabel('amplitude')
    ax0.text(0.5,2.75,'True components of mixing waveform')
    handles, labels = ax0.get_legend_handles_labels()

    K, D, _D = C.shape
    vmin = np.min(C)
    vmax = np.max(C)

    for k in range(K):
        if k == 0:
            ax = fig.add_subplot(gs[0, 1])
        else:
            if k%2 == 0:
                ax = fig.add_subplot(gs[k%2, (k//2)+1])
            else:
                ax = fig.add_subplot(gs[k%2, (k//2)+1]) 
        pos = ax.imshow(C[k], cmap='jet')
        ax.set_axis_off()
        #ax.set_title("k={}".format(str(k)))
        ax.text(5, 25,"k={}".format(str(k)))
        pos.set_clim(vmin, vmax)
    start = (K//2)+1 


    ax1 =  fig.add_subplot(gs_hspace[2:4, 0])

    #plot the true sine waveform
    for k in range(K):
        ax1.plot(alphas1[k, :],marker=marker_set[k], color=color_set[k],label=str(k))
    ax1.set_xlabel('t')
    ax1.set_ylabel('amplitude')
    ax1.text(0.5,2.75,'True components of sine waveform')
    

    K, D, _D = C.shape
    vmin = np.min(C1)
    vmax = np.max(C1)

    for k in range(K):
        if k == 0:
            ax = fig.add_subplot(gs_hspace[2, 1])
        else:
            if k%2 == 0:
                ax = fig.add_subplot(gs_hspace[k%2+2, (k//2)+1])
            else:
                ax = fig.add_subplot(gs_hspace[k%2+2, (k//2)+1])               
        pos = ax.imshow(C1[k], cmap='jet')
        ax.set_axis_off()
        #ax.set_title("k={}".format(str(k)))
        ax.text(5, 25,"k={}".format(str(k)))
        pos.set_clim(vmin, vmax)
    start = (K//2)+1 
    
    
    #estimated mixing graph
    ax2 =  fig.add_subplot(gs[0:2, start])
 
    for k in range(K):
        ax2.plot(estalphas[k, :],marker=marker_set[k], color=color_set[k],label="k={}".format(str(k)))
    #ax2.set_xlabel('t')
    ax2.text(0.5,2.75,'Estimated components (subjects=15)')

    ax0.get_shared_y_axes().join(ax0, ax2)
    ax2.set_yticklabels([])

    for k in range(K):
        if k == 0:
            ax = fig.add_subplot(gs[0, start+1])
        else:
            if k%2 == 0:
                ax = fig.add_subplot(gs[k%2, (k//2)+1+start])
            else:
                ax = fig.add_subplot(gs[k%2, (k//2)+1+start]) 
        pos = ax.imshow(estC[k], cmap='jet')
        ax.set_axis_off()
        #ax.set_title("k={}".format(str(k)))
        ax.text(5, 25,"k={}".format(str(k)))
        pos.set_clim(vmin, vmax)

    #estimated sine graph
    ax3 =  fig.add_subplot(gs_hspace[2:4, start])
 
    for k in range(K):
        ax3.plot(estalphas1[k, :],marker=marker_set[k], color=color_set[k],label="k={}".format(str(k)))
    ax3.set_xlabel('t')
    ax3.text(0.5,2.75,'Estimated components (subjects=15)')

    ax1.get_shared_y_axes().join(ax1, ax3)
    ax3.set_yticklabels([])

    for k in range(K):
        if k == 0:
            ax = fig.add_subplot(gs_hspace[2, start+1])
        else:
            if k%2 == 0:
                ax = fig.add_subplot(gs_hspace[k%2+2, (k//2)+1+start],)
            else:
                ax = fig.add_subplot(gs_hspace[k%2+2, (k//2)+1+start]) 
        pos = ax.imshow(estC1[k], cmap='jet')
        ax.set_axis_off()
        #ax.set_title("k={}".format(str(k)))
        ax.text(5, 25,"k={}".format(str(k)))
        pos.set_clim(vmin, vmax)
    ax1.legend(handles, labels, ncol=4, loc='upper center', bbox_to_anchor=(2, -0.25))
    fig.subplots_adjust(right=0.85, bottom=0.2, wspace=0.2, hspace=0.2)
    

    cbar_ax = fig.add_axes([0.86, 0.25, 0.02, 0.6])
    fig.colorbar(pos, cax=cbar_ax)
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname)
