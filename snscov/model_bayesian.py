import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import positive_semidefinite_kernels as psd
from tensorflow_probability import edward2 as ed
from tensorflow.python.keras.optimizers import Adam, SGD

###############################
#                             #
# M5 Bayesian Learning method #
#                             #
###############################

class BSL_ModelBuilder:
    def __init__(self, data, shape, n_components, fullcov, beta=2., dict_components=None, 
                 rho=2.5, length_scale=10.0, amplitude=2.0):
        self.data = data
        self.beta = tf.Variable(beta, dtype=tf.float32)
        self.N, self.T, self.D = shape
        self.K = n_components
        self.fullcov = fullcov
        self.dict_components = dict_components
        self.rho = rho
        self.ls = length_scale
        self.amp = amplitude 

    def train(self):
        def model(n_components, num_data_points, n_time, data_dim, beta=2.):
            ### Sample Matern Gaussian Process for weights
            # construct the Gram matrix
            self.length_scale = tf.Variable(self.ls*np.ones(n_components, dtype=np.float32), constraint=lambda x: tf.nn.relu(x) + 1e-8)
            self.amplitude = tf.Variable(self.amp*np.ones(n_components, dtype=np.float32), constraint=lambda x: tf.nn.relu(x) + 1e-8)
            self.m = tf.Variable(np.zeros(n_components, dtype=np.float32),name='m')
            self.d = tf.Variable(np.zeros(n_components, dtype=np.float32), constraint=lambda x: tf.nn.relu(x) + 1e-8)
            c_mean = tf.transpose(tf.reshape(tf.tile(self.m, [n_time]),(n_time,n_components)))
            kernel = psd.MaternFiveHalves(amplitude=self.amplitude, length_scale=self.length_scale)
            times = tf.constant(np.arange(n_time).reshape(n_time,1), dtype=tf.float32)
            matrix = kernel.matrix(times, times)+self.d[:,tf.newaxis,tf.newaxis]
            # draw the weights from the Gaussian process
            a = ed.MultivariateNormalFullCovariance(loc=c_mean,
                    covariance_matrix=matrix, name='a')
        
            ### Sample components the dependent relevance determination prior
            # draw the correlated sparsity
            z = ed.MultivariateNormalFullCovariance(
                    loc=-self.rho * tf.ones([data_dim]),
                    covariance_matrix=tf.constant(self.fullcov, dtype=tf.float32), name='z')
            # draw vs using correlated sparsity
            v = ed.MultivariateNormalDiag(
                    loc=tf.zeros([n_components,data_dim]),
                    scale_diag=tf.math.softplus(z[tf.newaxis,:]), name='v')
            ### draw the data from the model
            betaInv = tf.reciprocal(self.beta)
            # weight the components appropriately
            sumDictionary = tf.einsum('kt,ki,kj->tij', tf.nn.relu(a), v, v)
            noiseMatrix = betaInv*tf.diag(tf.ones(data_dim))
            X = ed.MultivariateNormalFullCovariance(
                    loc=tf.zeros([num_data_points, n_time, data_dim]),
                    covariance_matrix=(noiseMatrix[tf.newaxis,:,:] + sumDictionary),
                    name='X')
            return X, (v, a, z)
        
        log_joint = ed.make_log_joint_fn(model)
        
        def likelihood(x_train, v, a, z):
            
            return log_joint(data_dim=self.D, num_data_points=self.N, n_components=self.K, n_time=self.T,
                X=x_train, v=v, a=a, z=z)
        def variational_model(qv_mean, qv_var, qz_mean, qz_var, qa_mean, qa_var):
            qv = ed.MultivariateNormalDiag(loc=qv_mean, scale_diag=qv_var, name='qv')
            qa = ed.MultivariateNormalDiag(loc=qa_mean, scale_diag=qa_var, name='qa')
            qz = ed.MultivariateNormalDiag(loc=qz_mean, scale_diag=qz_var, name='qz')
            return qv, qz, qa

        log_q = ed.make_log_joint_fn(variational_model)

        self.qv_mean = tf.Variable(np.zeros((self.K,self.D), dtype=np.float32))
        self.qv_var = tf.Variable(np.ones((self.K,self.D), dtype=np.float32), constraint=lambda x: tf.nn.relu(x) + 1e-8)
        self.qa_mean = tf.Variable(np.ones((self.K, self.T), dtype=np.float32) +
                np.random.uniform(low=-0.5, high=0.5, size=(self.K, self.T)), constraint=lambda x: tf.nn.relu(x) + 1e-8,dtype=tf.float32)
        self.qa_var = tf.Variable(np.ones((self.K, self.T), dtype=np.float32), constraint=lambda x: tf.nn.relu(x) + 1e-8)
        self.qz_mean = tf.Variable(-np.ones(self.D, dtype=np.float32))
        self.qz_var = tf.Variable(np.ones(self.D, dtype=np.float32), constraint=lambda x: tf.nn.relu(x) + 1e-8)

        def entropy(qv, qa, qz):
            return -log_q(
                    qv_mean=self.qv_mean, qv_var=self.qv_var,
                    qa_mean=self.qa_mean, qa_var=self.qa_var,
                    qz_mean=self.qz_mean, qz_var=self.qz_var,
                    qv=qv, qa = qa, qz=qz)

        qv, qz, qa = variational_model(self.qv_mean, self.qv_var, self.qz_mean, self.qz_var, self.qa_mean, self.qa_var)
        self.elbo = entropy(qv, qa, qz) + likelihood(self.data, qv, qa, qz)

        optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
        train = optimizer.minimize(-self.elbo)
        return train
