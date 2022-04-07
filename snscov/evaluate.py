import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import cvxpy as cp
from birkhoff import birkhoff_von_neumann_decomposition

import itertools
import math


def LERM(true_cov, est_cov):
    """
    compute average log Riemannian meric

    Parameters
    -----------
    true_cov: ground truth covariance matrix, shape (D, D)
    est_cov:  estimated covariance matrix, shape (D, D)

    Returns
    -------
    LERM: scalar
    """

    w_true, v_true = linalg.eigh(true_cov)
    w_est,  v_est  = linalg.eigh(est_cov)

    w_true = np.where(w_true <= 1e-10, 1e-10, w_true)
    w_est  = np.where(w_est  <= 1e-10, 1e-10, w_est)

    log_true = v_true @ np.diag(np.log10(w_true)) @ v_true.T
    log_est  = v_est  @ np.diag(np.log10(w_est))  @ v_est.T
    lerm = np.sqrt(np.sum((log_true - log_est) ** 2))
    return lerm

def avg_LERM(true_a, true_d, est_a, est_d):
    """
    compute average log Riemannian meric

    Parameters
    -----------
    true_a: ground truth temporal coefficients, shape (T, K) 
    true_d: ground truth dictionary matrix, shape (K, D)
    est_a:  estimated temporal coefficients, shape (T, K)
    est_d:  estimated dictionary matrix, shape (K, D)

    Returns
    -------
    avg_lerm: \frac{1}{T}\sum_{t=1}^T LERM(true_cov_t, est_cov_t), scalar
    """

    true_C = np.einsum('ki,kj->kij', true_d, true_d)
    est_C  = np.einsum('ki,kj->kij', est_d,  est_d)

    true_a = np.where(true_a<=1e-5, 1e-5, true_a)
    est_a  = np.where(est_a<=1e-5,  1e-5, est_a)

    true_covar = np.einsum('ti,ijk->tjk', np.log(true_a), true_C)
    est_covar  = np.einsum('ti,ijk->tjk', np.log(est_a),  est_C)    


    avg_lerm = np.sqrt(np.sum((true_covar-est_covar) ** 2)) / true_covar.shape[0]
    return avg_lerm
    
def avg_LERM_cov(true_a, true_d, cov):
    """
    compute average log Riemannian meric

    Parameters
    -----------
    true_a: ground truth temporal coefficients, shape (T, K) 
    true_d: ground truth dictionary matrix, shape (K, D)


    Returns
    -------
    avg_lerm: \frac{1}{T}\sum_{t=1}^T LERM(true_cov_t, est_cov_t), scalar
    """

    true_covar  = np.zeros(cov.shape)
    for t in range(cov.shape[0]):
        idx = np.where(true_a[t,:]>1e-5)[0]
      
        log_a = np.log(true_a[t,idx])
        true_covar[t,:,:] =  true_d[idx,:].T @ np.diag(log_a) @ true_d[idx,:]

    est_covar  = np.zeros(cov.shape)
    for t in range(cov.shape[0]):
        w, eigh = linalg.eigh(cov[t,:,:])
        idx = np.where(w>1e-5)[0]
        log_w = np.log(w[idx])
        est_covar[t,:,:] =  eigh[:,idx] @ np.diag(log_w) @ eigh[:,idx].T


    avg_lerm = np.sqrt(np.sum((true_covar-est_covar) ** 2)) / est_covar.shape[0]
    return avg_lerm

def avg_cov(true_a, true_d, cov):
    """
    compute average log Riemannian meric

    Parameters
    -----------
    true_a: ground truth temporal coefficients, shape (T, K) 
    true_d: ground truth dictionary matrix, shape (K, D)


    Returns
    -------
    avg_lerm: \frac{1}{T}\sum_{t=1}^T LERM(true_cov_t, est_cov_t), scalar
    """

    true_covar  = np.zeros(cov.shape)
    for t in range(cov.shape[0]):

        true_covar[t,:,:] =  true_d.T @ np.diag(true_a[t]) @ true_d

    est_covar  = np.zeros(cov.shape)
    for t in range(cov.shape[0]):
        est_covar[t,:,:] =  cov[t,:,:]


    avg_lerm = np.sum(np.sqrt(np.sum((true_covar-est_covar) ** 2, axis=(1,2)))) / est_covar.shape[0]
    return avg_lerm


def ortogonal_procrustes(true_a, true_d, est_a, est_d):
    """
    Parameters
    -----------
    true_a: ground truth temporal coefficients, shape (T, K) 
    true_d: ground truth dictionary matrix, shape (K, D)
    est_a:  estimated temporal coefficients, shape (T, K)
    est_d:  estimated dictionary matrix, shape (K, D)

    Returns   
    -----------
    ortho_procrustes:

    """
    m = np.einsum('ij,kj->ik', est_d, true_d)
    u, s, vh = linalg.svd(m, full_matrices=True)
    r_matrix = np.einsum('ij,jk->ik', u, vh)

    d_dist = np.sum((true_d - np.einsum('ij,jk->ik', r_matrix.T, est_d)) **2)

    a_dist = 0 
    for t in range(est_a.shape[0]):
        a_dist += np.sum((np.diag(true_a[t])-r_matrix.T@np.diag(est_a[t])@ r_matrix) **2)
    print ('d_dist',d_dist,'a_dist',a_dist)

    return a_dist+d_dist

def dual_orthogonal_procrustes(true_a, true_d, est_a, est_d):
    """
    Parameters
    -----------
    true_a: ground truth temporal coefficients, shape (T, K) 
    true_d: ground truth dictionary matrix, shape (K, D)
    est_a:  estimated temporal coefficients, shape (T, K)
    est_d:  estimated dictionary matrix, shape (K, D)

    Returns
    -------
    dual op: sum(Sigma) , scalar 
    """
    total_a = 0
    total_d = 0

    m = np.einsum('ij,kj->ik', est_d, true_d)+np.einsum('ji,jk->ik',est_a, true_a)
    u, s, vh = linalg.svd(m, full_matrices=True)
    r_matrix = np.einsum('ij,jk->ik', u, vh)


    a_dist = np.sum((true_a - np.einsum('ij,jk->ik', est_a, r_matrix)) **2)
    d_dist = np.sum((true_d.T - np.einsum('ij,jk->ik', est_d.T, r_matrix.T)) **2)

    print(a_dist,d_dist, np.sum(true_d.T@true_d-est_d.T@est_d)**2)
    return a_dist + d_dist

def I(n):
    A = []
    for i in range(n):
        A.append([1. if j == i else 0 for j in range(n)])
    return A

def dual_permutation(true_a, true_d, est_a, est_d):
    """
    Parameters
    -----------
    true_a: ground truth temporal coefficients, shape (T, K) 
    true_d: ground truth dictionary matrix, shape (K, D)
    est_a:  estimated temporal coefficients, shape (T, K)
    est_d:  estimated dictionary matrix, shape (K, D)

    Returns
    -------
    dual op: sum(Sigma) , scalar 
    """
    K = true_a.shape[1]
    p_idx = I(K)

    eva_metric = np.zeros(math.factorial(K))
    for idx, m in enumerate (itertools.permutations(p_idx)):
        r_matrix = np.array(m).reshape((K,K))
        a_dist = np.sum((true_a.T - np.einsum('ij,jk->ik',r_matrix, est_a.T )) **2)
        
        d_dist = 0
        temp = np.einsum('ij,jk->ik', r_matrix, est_d)
        #signed permutation
        for j, vec in enumerate(true_d):
            a = np.sum((vec-temp[j])**2)
            b = np.sum((vec+temp[j])**2)
            d_dist += np.min((a,b))
        eva_metric[idx] = a_dist + d_dist
    
    return np.min(eva_metric)

def dual_permutation_separate(true_a, true_d, est_a, est_d):
    """
    Parameters
    -----------
    true_a: ground truth temporal coefficients, shape (T, K) 
    true_d: ground truth dictionary matrix, shape (K, D)
    est_a:  estimated temporal coefficients, shape (T, K)
    est_d:  estimated dictionary matrix, shape (K, D)

    Returns
    -------
    dual op: sum(Sigma) , scalar 
    """
    K = true_a.shape[1]
    p_idx = I(K)
    a_list = []
    d_list = []
    eva_metric = np.zeros(math.factorial(K))
    for idx, m in enumerate (itertools.permutations(p_idx)):
        r_matrix = np.array(m).reshape((K,K))
        a_dist = np.sum((true_a.T - np.einsum('ij,jk->ik',r_matrix, est_a.T )) **2)
        #print('permute_d',np.einsum('ij,jk->ik', r_matrix, est_d))
        #print('true_d',true_d)
        d_dist = 0
        temo = np.einsum('ij,jk->ik', r_matrix, est_d)
        for j, vec in enumerate(true_d):
            a = np.sum((vec-temo[j])**2)
            b = np.sum((vec+temo[j])**2)
            d_dist += np.min((a,b))
        
        temp1 =  np.sum((true_d - np.einsum('ij,jk->ik', r_matrix, est_d)) **2)
        temp2 =  np.sum((true_d + np.einsum('ij,jk->ik', r_matrix, est_d)) **2)
        est_C  = np.einsum('ki,kj->ij', est_d, est_d)
        true_C = np.einsum('ki,kj->ij', true_d, true_d)
       
        #d_dist = linalg.norm(true_d - np.einsum('ij,jk->ik', r_matrix, est_d),2)
        eva_metric[idx] = a_dist + d_dist
        a_list.append(a_dist)
        d_list.append(d_dist)
    a_list = np.array(a_list)
    d_list = np.array(d_list)
    #print('c',np.sum((est_C-true_C)**2))
    min_idx = np.argmin(eva_metric)

    return np.min(eva_metric), np.min(d_list), np.min(a_list)


def dual_permutation_matrix(true_a, true_d, est_a, est_d):
    """
    Parameters
    -----------
    true_a: ground truth temporal coefficients, shape (T, K) 
    true_d: ground truth dictionary matrix, shape (K, D)
    est_a:  estimated temporal coefficients, shape (T, K)
    est_d:  estimated dictionary matrix, shape (K, D)

    Returns
    -------
    dual op: [sum(Sigma), rotation_matrix], [scalar, matrix shape (K,K)] 
    """
    K = true_a.shape[1]
    p_idx = I(K)

    eva_metric = np.zeros(math.factorial(K))
    for idx, m in enumerate (itertools.permutations(p_idx)):
        r_matrix = np.array(m).reshape((K,K))

        a_dist = np.sum((true_a.T - np.einsum('ij,jk->ik',r_matrix, est_a.T )) **2)
        d_dist = 0
        temp = np.einsum('ij,jk->ik', r_matrix, est_d)
        #signed permutation
        for j, vec in enumerate(true_d):
            a = np.sum((vec-temp[j])**2)
            b = np.sum((vec+temp[j])**2)
            d_dist += np.min((a,b))
        eva_metric[idx] = a_dist + d_dist

    r = np.argmin(eva_metric)
    for idx, m in enumerate (itertools.permutations(p_idx)):
        if idx == r:
            r_matrix = np.array(m).reshape((K,K))
    return np.min(eva_metric), r_matrix.T

def large_scale_permutation(true_a, true_d, est_a, est_d):
    """
    Parameters
    -----------
    true_a: ground truth temporal coefficients, shape (T, K) 
    true_d: ground truth dictionary matrix, shape (K, D)
    est_a:  estimated temporal coefficients, shape (T, K)
    est_d:  estimated dictionary matrix, shape (K, D)

    Returns
    -------
    dual op: sum(Sigma) , scalar 
    """
    K = true_a.shape[1] 
    r_matrix = cp.Variable((K,K))
    objective = cp.Minimize(cp.sum_squares(true_a - est_a@r_matrix) + cp.sum_squares(true_d - r_matrix@est_d))
    constraints = []
    constraints.append(r_matrix >= 0.)
    constraints.append(r_matrix <= 1.)
    for i in range(K):
        constraints.append(cp.sum(r_matrix[:,i]) == 1.)
        constraints.append(cp.sum(r_matrix[i,:]) == 1.)
    prob = cp.Problem(objective, constraints)
    prob.solve()
    r_matrix = np.where(r_matrix.value <0. ,0., r_matrix.value)
    r_matrix = np.where(r_matrix>1., 1., r_matrix)
    
    result = birkhoff_von_neumann_decomposition(r_matrix)
    coefficients, permutations= zip(*result)
    r_matrix = permutations[np.argmax(np.array(coefficients))]
    a_dist = np.sum((true_a - np.einsum('ij,jk->ik', est_a, r_matrix)) **2)
    d_dist = np.sum((true_d - np.einsum('ij,jk->ik', r_matrix, est_d)) **2)
    return a_dist + d_dist

def large_scale_signed_permutation(true_a, true_d, est_a, est_d):
    """
    Parameters
    -----------
    true_a: ground truth temporal coefficients, shape (T, K) 
    true_d: ground truth dictionary matrix, shape (K, D)
    est_a:  estimated temporal coefficients, shape (T, K)
    est_d:  estimated dictionary matrix, shape (K, D)

    Returns
    -------
    dual op: sum(Sigma) , scalar 
    """
    K = true_a.shape[1] 
    r_matrix = cp.Variable((K,K))
    objective = cp.Minimize(cp.sum_squares(true_a - est_a@r_matrix))
    constraints = []
    constraints.append(r_matrix >= 0.)
    constraints.append(r_matrix <= 1.)
    for i in range(K):
        constraints.append(cp.sum(r_matrix[:,i]) == 1.)
        constraints.append(cp.sum(r_matrix[i,:]) == 1.)
    prob = cp.Problem(objective, constraints)
    prob.solve()
    r_matrix = np.where(r_matrix.value <0. ,0., r_matrix.value)
    r_matrix = np.where(r_matrix>1., 1., r_matrix)
    
    result = birkhoff_von_neumann_decomposition(r_matrix)
    coefficients, permutations= zip(*result)
    r_matrix = permutations[np.argmax(np.array(coefficients))]
    a_dist = np.sum((true_a - np.einsum('ij,jk->ik', est_a, r_matrix)) **2)
    d_dist = np.sum((np.einsum('ij,ik->jk', true_d, true_d) - np.einsum('ij,ik->jk', est_d, est_d)) **2)/(2*(np.sqrt(2)-1))
    return a_dist + d_dist


def large_scale_signed_permutation_matrix(true_a, true_d, est_a, est_d):
    """
    Parameters
    -----------
    true_a: ground truth temporal coefficients, shape (T, K) 
    true_d: ground truth dictionary matrix, shape (K, D)
    est_a:  estimated temporal coefficients, shape (T, K)
    est_d:  estimated dictionary matrix, shape (K, D)

    Returns
    -------
    dual op: sum(Sigma) , scalar 
    """
    K = true_a.shape[1] 
    r_matrix = cp.Variable((K,K))
    objective = cp.Minimize(cp.sum_squares(true_a - est_a@r_matrix))
    constraints = []
    constraints.append(r_matrix >= 0.)
    constraints.append(r_matrix <= 1.)
    for i in range(K):
        constraints.append(cp.sum(r_matrix[:,i]) == 1.)
        constraints.append(cp.sum(r_matrix[i,:]) == 1.)
    prob = cp.Problem(objective, constraints)
    prob.solve()
    r_matrix = np.where(r_matrix.value <0. ,0., r_matrix.value)
    r_matrix = np.where(r_matrix>1., 1., r_matrix)
    
    result = birkhoff_von_neumann_decomposition(r_matrix)
    coefficients, permutations= zip(*result)
    r_matrix = permutations[np.argmax(np.array(coefficients))]
    a_dist = np.sum((true_a - np.einsum('ij,jk->ik', est_a, r_matrix)) **2)
    d_dist = np.sum((np.einsum('ij,ik->jk', true_d, true_d) - np.einsum('ij,ik->jk', est_d, est_d)) **2)/(2*(np.sqrt(2)-1))
    return a_dist + d_dist, r_matrix