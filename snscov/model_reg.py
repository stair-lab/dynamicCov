import numpy as np
from tqdm import tqdm
from scipy import linalg
from snscov.evaluate import dual_permutation


#############################################
#                                           #
# M6 pregularization approach               #
#                                           #
#############################################
#used for running small-scale data (small K, recommended for K<=8)

def construct_gamma(eta, group, D):
    """
    Construct gammma matrix
    
    Parameters:
    -----------
    eta: array-like, shape (K, |G|)
    group: group structure, list
    D: data dimension, scalar

    Returns:
    -----------
    gamma: array-like, shape (K, D)
    """
    #construct gamma
            
    inv_eta = 1 / (eta + 1e-10)
    gamma = np.zeros((eta.shape[0], D))

    for i, g in enumerate(group):
        gamma[:, np.array(g)] += np.tile(inv_eta[:, i],(len(g),1)).T 

    return gamma

def _update_eta(dictionary, group):
    """
    temporal eta update

    Parameters        if(np.sum(ak_s_wk) > 1e-10):
            dictionary[:, k] /= np.sum(ak_s_wk)
        atom_norm = nrm2(dictionary[:,k])

        dictionary[:, k] /= np.max((atom_norm, 1))
    ----------
    dictionary: array-like, shape(D, K)
    group: group structure, list

    Returns
    ---------
    eta: update eta, array-like, shape (K, |G|)
    """
    group_card = len(group)
    new_eta = np.zeros((dictionary.shape[1], group_card))
    dict_square = dictionary ** 2
    count = 0
    for g in group:
        eta_g = np.sum(dict_square[np.array(g),:], axis=0)
        new_eta[:, count] = np.sqrt(eta_g)
        count +=1
    return new_eta


def check_dimension(X, dictionary, alphas):
    # check dimension
    if dictionary.shape[1] != X.shape[2]:
        raise ValueError("Dictionary and X have different numbers of features:"
                        "dictionary.shape: {} X.shape{}".format(
                            dictionary.shape, X.shape))
    if alphas.shape[0] != X.shape[1]:
        raise ValueError("Alphas and X have different numbers of timepoints:"
                        "Alphas.shape: {} X.shape{}".format(
                            alphas.shape, X.shape))
 

def check_group(group, D):     
    group_card = len(group)

    for i in range(D):
        count = [ 1 for g in group if i in g]
        if len(count) == 0:
            group.append([i])
    return group



class Model6:
    def __init__(self, K, T, D, lambda1 = 1., lambda2 = 1., kernel = None, group = None, max_iter = 1000, tol=1e-3,
                 a_method = 'temporal_kernel', d_method = 'sparse', verbose=False, la_rate = 1e-3, ld_rate=0.):
        self.K = K
        self.D = D
        self.T = T
        self.dict = None      #shape (K, D)
        self.alphas = None    #shape (T, K)
        self.kernel = kernel
        self.la_rate = la_rate
        self.ld_rate = ld_rate
        self.tol = tol
        if group == None:
            #lasso
            self.eta = np.zeros((self.K, self.D))
            self.group = [[i] for i in range(self.D)]
        else:
            group_card = len(group)
            self.eta = np.zeros((self.K, group_card))
            
            self.group = check_group(group, self.D)
            print('check group cardinality', len(self.group))

        
        self.max_iter = max_iter
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.a_method = a_method #[temporal_kernel', 'no_reg']
        self.d_method = d_method #['sparse', 'sparse']

        self.verbose = verbose
        if self.verbose ==True:
            print('Initialize Dictionary Learning')
            print('------------------------------')
            print('Temporal penalty function: %s'%(self.a_method))
            print('Structural penalty function: %s'%(self.d_method))
    def compute_sample_covariance(self, X):
        """
        compute sample covariance at every time point

        Parameters
        ----------
        X: array-like, shape (N, T, D)

        return
        ----------
        """
        self.sample_c = np.einsum('nti,ntj->tij',X,X) / (X.shape[0])


    def update_alphas(self, X, dictionary, alphas):
        """
        alphas update

        Parameters
        ----------
        X: data, array-like, shape (N, T, D)
        W: weights, arry-like, shape (N, T, K)
        dictionary: array-like, shape(K, D)
        alphas: array-like, shape(T, K)

        Returns
        ---------
        alphas: update alphas, array-like, shape(T, K),
        """
        # check dimension

        check_dimension(X, dictionary, alphas)
        
        diag_a = np.zeros((self.T, self.K, self.K))
        s0,s1,s2 = diag_a.shape
        diag_a.reshape(s0,-1)[:,::s2+1] = alphas #shape T,K,K

        est_cov1 = np.einsum('ij,tik->tjk', dictionary, diag_a) #shape T D K
        est_cov2 = np.einsum('tij,jk->tik', est_cov1, dictionary) #shape T D D
        diff_cov = est_cov2 - self.sample_c #shape T,D,D

        step_1 = np.einsum('tij,kj->tik', diff_cov, dictionary) #shape T D K
        step_2 = np.einsum('ki,tik->tk', dictionary, step_1) #shape T K
        delta_A = 2 * (alphas * step_2)/self.T
        #print('delta_A', np.max(delta_A))
        #print('regularizer', np.max(self.lambda1 * np.einsum('ij,jk->ik', self.kernel, alphas)))
        if self.a_method == 'temporal_kernel':
            #print('max a update', np.max((self.la_rate * (delta_A + self.lambda1 * np.einsum('ij,jk->ik', self.kernel, alphas)))))
            new_alphas = alphas - self.la_rate * (delta_A + 2*self.lambda1 * np.einsum('ij,jk->ik', self.kernel, alphas))
        else:
            new_alphas = alphas - self.la_rate * delta_A
        #compute projected gradient descent
        new_alphas = np.where(new_alphas <0. ,0., new_alphas)
        amp = 10.
        new_alphas = np.where(new_alphas > amp , amp, new_alphas)
        return new_alphas

    def update_dictionary(self, X, dictionary, alphas):
        """
        dictionary update

        Parameters
        ----------
        X: array-like, shape (N, T, D)
        W: weights, arry-like, shape (N, T, K)
        dictionary: array-like, shape(K, D)
        alphas: array-like, shape(T, K)


        Returns
        ---------
        dictionary: update dictionary, array-like, shape(K, D)
        residuals: scalar
        """
        # check dimension
        check_dimension(X, dictionary, alphas)

        
        diag_a = np.zeros((self.T, self.K, self.K))
        s0,s1,s2 = diag_a.shape
        diag_a.reshape(s0,-1)[:,::s2+1] = alphas #shape T,K,K
        est_cov1 = np.einsum('ij,tik->tjk', dictionary, diag_a) #shape T D K
        est_cov2 = np.einsum('tij,jk->tik', est_cov1, dictionary) #shape T D D
        
        diff_cov = np.einsum('tij,tjk->ik',(est_cov2 - self.sample_c),est_cov1)/self.T #shape D K
        delta_D = 2 * diff_cov /self.T


        #x_ave = X.mean(axis=0)
        if self.d_method == "sparse":
            gamma = construct_gamma(self.eta, self.group, self.D)
            new_dictionary = dictionary - self.ld_rate * (delta_D.T + 2*self.lambda2*(gamma*dictionary)) 
            self.eta = _update_eta(new_dictionary.T, self.group)
                    
        else: 
            new_dictionary = dictionary - self.ld_rate * delta_D.T
        residuals = np.sum((est_cov2 - self.sample_c)**2) / (2*self.T)
        #compute projected gradient descent

        nrm = np.sqrt(np.sum(new_dictionary**2, axis=1)) #shape K
        for k in range(self.K):
            new_dictionary[k,:] /= np.max((nrm[k], 1))

        return new_dictionary, residuals



        
    def fit(self, X, tol=0., true_a=None, true_d=None, evaluate=False):
        """
        fit the model from X

        Parameters
        ----------
        X: array-like, shape (N, T, D)

        Latent Variable
        ----------
        W: weights, array-like, shape (N, T , K)
        dictionary: array-like, shape (K, D)
        alphas: array-like, shape (T, K)

        Returns
        ---------
        self: object
            Returns the object itself

        """
        """
        #chcek array, warm restart
        x_ave = X.mean(axis=0)
        if self.dict is not None and self.alphas is not None:
            alphas = self.alphas
            dictionary = self.dict
        else:
            #initialization by taking the mean
            alphas, S, dictionary = linalg.svd(x_ave, full_matrices=False)
            dictionary = S[:, np.newaxis] * dictionary

        if len(dictionary) >= self.K:
            dictionary = dictionary[:self.K, :]
            alphas = alphas[:, :self.K]
        else:
            ##padding with zeros
            pad_c = self.K-len(dictionary)
            dictionary = np.r_[dictionary, np.zeros((pad_c, self.D))]
            alphas = np.c_[alphas, np.zeros(len(alphas), pad_c)]
        # Fortran-order dict, as we are going to access its row vectors
        """
        #compute the sample covariance
        if self.dict is not None and self.alphas is not None:
            alphas = self.alphas
            dictionary = self.dict
        else:
            #initialization by taking the mean
            S_N = np.einsum('nji,njk->ik', X, X) / (X.shape[0]) #shape(D,D) v
            w, eigv = linalg.eigh(S_N) #v
            w = w[::-1] #shape(D) v
            eigv = eigv[:,::-1] #shape(D, D)v
            S_T = np.einsum('ntk,ntj->tkj', X, X) / X.shape[0] #shape(T,D,D) v
            S_Tv = np.einsum('tkj,ji->tki', S_T, eigv) #shape(T,D,D) v
            vtS_Tv = np.einsum('ki,tkj->tij', eigv, S_Tv) # shape(T,D,D) v
            A = np.einsum('tii->ti', vtS_Tv) #v

            
            
        
        
        if self.D >= self.K:
            dictionary = eigv[:, :self.K].T
            alphas = A[:, :self.K]
        else:
            ##padding with zeros
            pad_c = self.K-len(dictionary)
            dictionary = np.r_[eigv.T, np.zeros((pad_c, self.D))]
            alphas = np.c_[A, np.zeros(len(alphas), pad_c)]


        dictionary = np.array(dictionary, order='F')
        
        self.eta = _update_eta(dictionary.T, self.group)
          

        ii = -1
        self.error_ = []
        self.residuals = []
        self.alphas_error = []
        self.dic_error  =[]
        self.dual_ortho = []
        self.compute_sample_covariance(X)
        self.max_gamma = []
        self.min_gamma = []
        self.best_alphas = None
        self.best_dictionary = None
        self.min_dual = 1e10
        for ii in tqdm(range(self.max_iter)):
            

            #update alphasX: mean of the data, array-like, shape (D, T)
            alphas = self.update_alphas(X, dictionary, alphas)
            
            #update dictionary
            dictionary, residuals = self.update_dictionary(X, dictionary, alphas)

            #cost function
            if self.a_method == 'no_reg':
                reg_a = 0.
            else:
                reg_a = self.lambda1 * np.trace(alphas.T @ self.kernel @alphas)
                
            if self.d_method == 'sparse':
                gamma = construct_gamma(self.eta, self.group, self.D) #KxD
                reg_d = self.lambda2 * np.sum(gamma * (dictionary ** 2))
                            
            else:
                reg_d = 0.

            if evaluate == True:
                self.dual_ortho.append(dual_permutation(true_a.T, true_d, alphas, dictionary))
                self.max_gamma.append(np.max(construct_gamma(self.eta, self.group, self.D)))
                self.min_gamma.append(np.min(construct_gamma(self.eta, self.group, self.D)))
                #print(self.dual_ortho[ii])
                if self.dual_ortho[ii] - self.dual_ortho[ii-1] > self.tol: 
                    break
            else:

                if ii>0:
                    if (np.abs(self.error_[ii+1] - self.error_[ii])<self.tol): 
                        break
                    else:
                        self.best_alphas = alphas
                        self.best_dictionary = dictionary
                else:
                    self.best_alphas = alphas
                    self.best_dictionary = dictionary                   

            self.alphas = alphas
            self.dict = dictionary                    

            cost = residuals + reg_a + reg_d
            self.error_.append(cost)
            self.residuals.append(residuals)
            self.alphas_error.append(reg_a)
            self.dic_error.append(reg_d)



        self.n_iter_ = ii+1
        return self

    def transform(self,X):
        """
        Encode the data as a sparse combination of the dictionary atoms.

        Parameters
        ----------
        self
        X: input data, array-like, shape (N,T,D)
        
        Returns
        ---------
        w: encode coefficients, array-like, shape (N, T, K)
        """
        code = self.update_weights( X, self.dict, self.alphas)
        nrm2, = linalg.get_blas_funcs(('nrm2',), (X,))
        R = X - ((code * self.alphas) @ self.dict)
        Residual = 0.5 * (nrm2(R) ** 2.0 ) / X.shape[0]
        
        return code, Residual



##test
if __name__ == "__main__":
    import synth_data as synth_data
    import matplotlib.pyplot as plt
    from kernel import Matern52Kernel
    import matplotlib.pyplot as plt
    from evaluate import residual, avg_LERM
    from sklearn.decomposition import DictionaryLearning
    from sklearn.preprocessing import normalize
    N = 10
    K = 3
    T = 50
    D = 10
    rho = 2.5
    fullcov = np.zeros((K, D, D), dtype=np.float32)

    v = np.array([[1, -1, 1, -1, -1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, -1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]], dtype=np.float32)

    alphas = synth_data.synthesize_weights(K, T)
    C = np.einsum('ki,kj->kij', v, v)



    sd = synth_data.synthesize_data(alphas, C, N)
    
    #synth_data.plot_weights(alphas, None, title_append="Synthetic Components")
    #synth_data.plot_components(C, None,title_append="Synthetic Weights")
    #construct structural_prior
    struc_v = np.array([
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]], dtype=np.float32)
    #struc = np.einsum('ki,kj->ij', struc_v, struc_v)
    struc = struc_v.T @ struc_v
    struc[3,0] = 1
    struc[0,3] = 1

    sc = np.linalg.matrix_power(struc, 4) # raise the square matrix to the power 4
    sc[struc == 0] = sc[struc==0] / np.max(sc[struc==0])
    sc[struc != 0] = struc[struc != 0]


    v = np.array([[1, -1, 1, -1, -1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, -1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]], dtype=np.float32)

    alphas = synth_data.synthesize_weights(K, T)
    C = np.einsum('ki,kj->kij', v, v)
    sd = synth_data.synthesize_data(alphas, C, N)
    
    N_test = 5
    test_sd = synth_data.synthesize_data(alphas, C, N_test) 
    eigs, _ = np.linalg.eigh(sc)
    mineig = np.min(eigs)
    if mineig <= 0.0:
        sc = sc - mineig*np.eye(D)
    
    #plt.imshow(sc, cmap='jet')
    #plt.title('Structural Prior Matrix')
    #plt.show()
    kernel = Matern52Kernel(T, 2.)
    kernel = linalg.solve(kernel,np.eye(T))
    np.set_printoptions(precision=2)

    sc = linalg.solve(sc, np.eye(D))


    

    if(True):
        a_coef = [0, 1e-10, 1e-5, 1e-3, 1e-2, 1e-1,]
        b_coef = [0, 1e-10, 1e-5, 1e-3, 1e-2, 1e-1,]
        avg_lerm = np.zeros((10, 2, 2, 1))
        train_residual = np.zeros((10, 2, 2, 1))
        test_residual = np.zeros((10, 2, 2, 1))
        labels = []
        a_method = ['temporal_kernel','no_reg']# 'temporal_kernel']
        d_method = ['sparse', 'sparse']#, 'sparse']#, 'sparse']#, 'sparse']
        w_method = ['ridge']#, 'no_reg', 'lasso_cd']#, 'ridge', 'no_reg']
        
        for i in range (1):
            sd = synth_data.synthesize_data(alphas, C, N)
            
            N_test = 5
            test_sd = synth_data.synthesize_data(alphas, C, N_test) 
            samples = sd.mean(axis=0)

            
            for a in range(len(a_method)):
                for d in range(len(d_method)):
                    for w in range(len(w_method)):

                        print('Test a_method: ', a_method[a], ' d_method: ', d_method[d], 'w_method:', w_method[w])
                        print('-------------------------------------------------------')
                        st_test = stDictionary(5, T, D, lambda1 = 1., lambda2= 1., lambda3 = 1.,
                                                kernel = kernel, struc_prior = sc,
                                                max_iter = 1000, a_method = a_method[a], d_method = d_method[d], w_method = w_method[w])
                        st_test.fit(sd)
                        print('[train group]', ' n_iter: ', st_test.n_iter_, ' training error: ', st_test.error_[-1])               
                        print('[train group]', ' avg_LERM: ', avg_LERM((alphas).T, v, st_test.alphas ** 2, st_test.dict) ,' data residual: ', st_test.transform(sd)[1])
                        print('[test  group]', ' data residual: ',st_test.transform(test_sd)[1])
                        print('[dictionary]', st_test.dict)
                        synth_data.plot_components(np.einsum('ki,kj->kij', st_test.dict, st_test.dict), fname='drd_components_best.png', title_append="Learned Components")
                        synth_data.plot_weights((st_test.alphas ** 2).T ,fname='drd_weights_best.png', title_append="Learned Weights")
                        avg_lerm[i, a, d, w] = avg_LERM((alphas).T, v, st_test.alphas ** 2, st_test.dict) 
                        train_residual[i, a, d, w] = st_test.transform(sd)[1]
                        test_residual[i, a, d, w] = st_test.transform(test_sd)[1]
                        
                        if i == 0:
                            labels.append("%s + %s + %s"%(a_method[a], d_method[d], w_method[w])) 
                        
                       
                            plt.figure()
                          
                            plt.plot(st_test.error_, label="%s + %s"%(a_method[a], d_method[d]))
                            plt.plot(st_test.train_residual, label='training residual', marker='x')
                            plt.plot(st_test.alphas_error,   label='alphas penalty', marker='D')
                            plt.plot(st_test.dic_error,      label='dictionary penalty', marker='o')
                            del st_test            

                            plt.title("Error")
                            plt.xlabel("n iterations")
                            plt.legend()
                            plt.savefig("../figures/%s_%s_%s.png"%(a_method[a], d_method[d], w_method[w]))
                            plt.close()
        mean_avg_lerm = np.mean(avg_lerm, axis=0).flatten()
        std_avg_lerm  = np.std(avg_lerm, axis=0).flatten()

        mean_train_residual = np.mean(train_residual, axis=0).flatten()
        std_train_residual  = np.std(train_residual, axis=0).flatten()

        mean_test_residual  = np.mean(test_residual, axis=0).flatten()
        std_test_residual   = np.std(test_residual, axis=0).flatten()
        
        plt.figure()
        plt.title('Average LERM')
        plt.errorbar(np.array(labels), mean_avg_lerm, std_avg_lerm, linestyle='None', marker='^')
        #plt.show()

        plt.figure()
        plt.title('Training Residual')
        plt.errorbar(np.array(labels), mean_train_residual, std_train_residual, linestyle='None', marker='^')
        #plt.show()

        plt.figure()
        plt.title('Testing Residual')
        plt.errorbar(np.array(labels), mean_test_residual, std_test_residual, linestyle='None', marker='^')
       # plt.show()
        
        
    