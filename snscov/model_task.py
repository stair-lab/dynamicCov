import numpy as np
from tqdm import tqdm
from scipy import linalg


#############################################
#                                           #
# M** proposed model                        #
#                                           #
#############################################
#used for running real data (taskfMRI)


def project_to_kernel(x, k_matrix, gamma):
    amp = 60.
    ub = np.ones(k_matrix.shape[1]) * amp
    w,v = np.linalg.eigh(k_matrix)
    u = np.einsum('ij,j->i', v.T, x)
    ub = np.einsum('ij,j->i', v.T, ub)
    

    if (np.sum( x.T @ k_matrix @ x) <= gamma):

        return x

    else:
        poly_order = np.int(4)
        poly2 = [np.sum((u**2) * w) - gamma]
        for i in range(1,poly_order):
            s = np.sum((u**2) * (w**(i+1)))
            new_s = (-1)**i * s * (i+1)
            
            poly2.append(new_s)
        poly2 = np.array(poly2)[::-1]
        roots = (np.roots(poly2))
        
        
        
        if roots.size != 0:
            l = np.abs(np.max(roots))
            A_inv = (l*w + 1)
            y = np.real(np.einsum('ij,j->i', v, u/A_inv))

            
        return y

        

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
 





class snscov_model:
    def __init__(self, K, T, D, kernel = None, max_iter = 1000, ini_method = 'spectral',
                 a_method = 'temporal_kernel', d_method = 'sparse', k_sparse=9, smooth=0.5, verbose=False, la_rate = 1e-3, ld_rate=1e-3):
        self.K = K
        self.D = D
        self.T = T
        self.dict = None      #shape (K, D)
        self.alphas = None    #shape (T, K)
        self.kernel = kernel
        self.la_rate = la_rate
        self.ld_rate = ld_rate
        self.k_sparse = k_sparse
        self.smooth = smooth
        self.ini_method = ini_method

        
        self.max_iter = max_iter


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
        self.sample_c = np.einsum('nti,ntj->tij',X,X) / (X.shape[0]-1)


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
        delta_A = (alphas * step_2)/self.T

        if self.a_method == 'temporal_kernel':

            new_alphas = alphas - self.la_rate * delta_A
            amp = 60.
            #print('lower',new_alphas.all()>=0.)
            new_alphas = np.where(new_alphas <0. ,0., new_alphas)
            #print('upper',new_alphas.all()<=amp)
            new_alphas = np.where(new_alphas > amp , amp, new_alphas)             
            for col in range(new_alphas.shape[1]):
                new_alphas[:,col]=project_to_kernel(new_alphas[:,col], self.kernel, self.smooth)
            while(new_alphas.any()<0 or new_alphas.any()>amp):
                print('iteration')
                new_alphas = np.where(new_alphas <0. ,0., new_alphas)
                #print('upper',new_alphas.all()<=amp)
                new_alphas = np.where(new_alphas > amp , amp, new_alphas)             
                for col in range(new_alphas.shape[1]):
                    new_alphas[:,col]=project_to_kernel(new_alphas[:,col], self.kernel, self.smooth)               

           
        else:
            new_alphas = alphas - self.la_rate * delta_A
            #compute projected gradient descent
            new_alphas = np.where(new_alphas <0. ,0., new_alphas)
            amp = 60.
            new_alphas = np.where(new_alphas > amp , amp, new_alphas)
        return new_alphas
    def compute_residual(self, X, dictionary, alphas):
        diag_a = np.zeros((self.T, self.K, self.K))
        s0,s1,s2 = diag_a.shape
        diag_a.reshape(s0,-1)[:,::s2+1] = alphas #shape T,K,K
        est_cov1 = np.einsum('ij,tik->tjk', dictionary, diag_a) #shape T D K
        est_cov2 = np.einsum('tij,jk->tik', est_cov1, dictionary) #shape T D D

        residuals = np.sum((est_cov2 - self.sample_c)**2) / (2*self.T)
        return residuals
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
            new_dictionary = dictionary - self.ld_rate * delta_D.T
        else: 
            new_dictionary = dictionary - self.ld_rate * (delta_D.T) 
            index = np.argsort(np.abs(new_dictionary), axis=1)
            s_k = self.D-self.k_sparse
            index_y = index[:,0:s_k]
            index_x = np.repeat(np.arange(self.K),s_k).reshape((self.K, s_k))
            new_dictionary[index_x,index_y]=1e-15

        residuals = np.sum((est_cov2 - self.sample_c)**2) / (2*self.T)
        #compute projected gradient descent

        nrm = np.sqrt(np.sum(new_dictionary**2, axis=1)) #shape K
        
        for k in range(self.K):
            #new_dictionary[k,:] /= np.max((nrm[k], 1))
            new_dictionary[k,:] /= (nrm[k])

        return new_dictionary, residuals



        
    def fit(self, X, tol=0.0001, true_a=None, true_d=None, evaluate=False):
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



        
        #compute the sample covariance
        if self.dict is not None and self.alphas is not None:
            alphas = self.alphas
            dictionary = self.dict
        else:
            if self.ini_method == 'spectral':
                #initialization by taking the mean
                S_N = np.einsum('nji,njk->ik', X, X) / (X.shape[0]*self.T) #shape(D,D) v
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
            else:
                dictionary = np.random.normal(0., 1., (self.K, self.D))
                d_norm = np.sum(dictionary**2, axis=1)
                dictionary /= d_norm[:, np.new_axis]
                alphas = np.abs(np.random.normal(0.,1.), (self.T, self.K))


        dictionary = np.array(dictionary, order='F')
        

          

        ii = -1
        self.error_ = []
        self.compute_sample_covariance(X)
        self.error_.append(self.compute_residual( X, dictionary, alphas))
        
        
        self.best_error = np.inf
        self.best_alphas = []
        self.best_dictionary = []
        
        for ii in tqdm(range(self.max_iter)):
            

            #update alphasX: mean of the data, array-like, shape (D, T)
            alphas = self.update_alphas(X, dictionary, alphas)
            
            #update dictionary
            dictionary, residuals = self.update_dictionary(X, dictionary, alphas)


            self.error_.append(residuals)
            self.alphas = alphas
            self.dict = dictionary
            if evaluate == True:
                
                if self.error_[ii+1] < self.best_error :
                    self.best_error  = self.error_[ii+1]
                    self.best_alphas = alphas
                    self.best_dictionary = dictionary
                #print(self.dual_ortho[ii])
                #if (self.error_[ii+1] - self.error_[ii])>tol: 
                #    break
                    






        self.n_iter_ = ii+1+1
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
    pass
    