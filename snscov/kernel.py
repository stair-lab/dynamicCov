import numpy as np

from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, ExpSineSquared
def Matern52Kernel(dim, amplitude=1.0, length_scale=5.0):
    x = np.arange(dim) + 1
    X,Y = np.meshgrid(x, x)
    z = np.sqrt(5*(X-Y)**2/length_scale)
    kernel_z = (1 + z + z**2/3) * np.exp(-z)
    kernel_z = amplitude * kernel_z
    return kernel_z 
def GaussianKernel(dim, amplitude=1.0, length_scale=3.0):
    x = np.arange(dim) + 1
    X,Y = np.meshgrid(x, x)
    z = (X-Y) **2 /(2*length_scale)
    kernel_z = amplitude * np.exp(-z)
    return kernel_z

def RBFKernel(dim, length_scale=5.0):
    kernel = RBF(length_scale=length_scale)
    x = np.arange(dim)+1
    x = x[:, np.newaxis]
    return kernel(x)

def RationalQuadraticKernel(dim, length_scale=5.0):
    kernel = RationalQuadratic(length_scale=length_scale)
    x = np.arange(dim)+1
    x = x[:, np.newaxis]
    return kernel(x)

def ExpSineSquaredKernel(dim, length_scale=5.0):
    kernel = ExpSineSquared(length_scale=length_scale)
    x = np.arange(dim)+1
    x = x[:, np.newaxis]
    return kernel(x)