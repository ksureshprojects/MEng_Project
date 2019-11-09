import numpy as np
from helpers import *
from PCA import *
from progressbar import ProgressBar

def soft_thresh(x, l):
    return np.array(list(map(lambda v: [np.sign(v[0]) * max(abs(v[0]) - l, 0)], x)))

def opt_thresh(x, t, tol):
    norm = np.linalg.norm(x, ord=1)
    if norm < t:
        return x
    
    l_u = np.max(np.abs(x))
    l_l = 0
    x_l = x
    while abs(norm - t) > tol:
        l_mid = (l_u + l_l)/2
        x_l = soft_thresh(x, l_mid)
        norm = np.linalg.norm(x_l, ord=1)
        if norm > t:
            l_l = l_mid
        else:
            l_u = l_mid
    
    return x_l
    

def sparse_rank1(X, t, tol):
    v = np.array([[np.random.rand() for i in range(X.shape[1])]]).T
    v = v/(np.linalg.norm(v, ord=2) + 0.00000001)
    # v[0,0] = 1
    count = 0
    while count < 100:
        Xv = np.dot(X, v)
        u = Xv / (np.linalg.norm(Xv, ord=2) + 0.00000001)
        S_XTu = opt_thresh(np.dot(X.T,u), t, tol)
        v_n = S_XTu / (np.linalg.norm(S_XTu, ord=2) + 0.00000001)
        v_diff = np.linalg.norm(v - v_n, ord=1)
        if v_diff < tol:
            break
        v = v_n
        count += 1
    
    return v


def optimal_t(X, tol, non_zero):
    
    if non_zero > X.shape[1]:
        raise ValueError("Invalid number of components")
    
    t = 1
    v = sparse_rank1(X, t, tol)
    zero_norm = np.linalg.norm(v.flatten(), ord=0)
    
    while zero_norm < non_zero:
        t += 1
        v = sparse_rank1(X, t, tol)
        zero_norm = np.linalg.norm(v.flatten(), ord=0)
        
    if zero_norm != non_zero:
        t_u = t
        t_l = t - 1
        count = 0
        while count < 100:
            t_mid = (t_u + t_l)/2
            v = sparse_rank1(X, t_mid, tol)
            zero_norm = np.linalg.norm(v.flatten(), ord=0)
            if zero_norm > non_zero:
                t_u = t_mid
            elif zero_norm < non_zero:
                t_l = t_mid
            else:
                break
            count += 1
    
    return v, t
            
def sparse(X, t, tol):
    v = np.array([[np.random.rand() for i in range(X.shape[1])]]).T
    v = v/(np.linalg.norm(v, ord=2) + 0.00000001)
    # v[0,0] = 1
    count = 0
    while count < 100:
        Xv = np.dot(X, v)
        u = Xv / (np.linalg.norm(Xv, ord=2) + 0.00000001)
        S_XTu = opt_thresh(np.dot(X.T,u), t, tol)
        v_n = S_XTu / (np.linalg.norm(S_XTu, ord=2) + 0.00000001)
        v_diff = np.linalg.norm(v - v_n, ord=1)
        if v_diff < tol:
            break
        v = v_n
        count += 1
    
    return v, u

def sparse_rank_n_uv(X, t, tol, r):
    X_ = X.copy()
    (u_, s, vh_) = np.linalg.svd(X)
    
    Uh = []
    Vh = []
    
    for i in range(r):
        v, u = sparse(normalise_(X_), t, tol)
        Uh.append(u.T[0])
        Vh.append(v.T[0])
        X_ = X_ - s[i] * np.dot(u, v.T)
    
    return np.array(Vh).T

def sparse_rank_n_vv(X, t, tol, r):
    X_ = X.copy()
#     (u_, s, vh_) = np.linalg.svd(X)
    
#     Uh = []
    Vh = []
    
    for i in range(r):
        v, u = sparse(normalise_(X_), t, tol)
#         Uh.append(u.T[0])
        Vh.append(v.T[0])
        X_ = X_ - np.dot(X_, np.dot(v, v.T))
    
    return np.array(Vh).T

def normalise_(X_):
    X_n = X_.T
    for i in range(len(X_n)):
        ran = np.max(X_n[i]) - np.min(X_n[i])
        X_n[i] = X_n[i]/(ran/2 + 0.000001)
        X_n[i] = X_n[i] - np.mean(X_n[i])
    return X_n.T
        
def sparsity(ml):
    pbar = ProgressBar()
    non_zero = []
    mse_ = []
    ts = np.linspace(5,105,20)
    for t in pbar(ts):
        
        V = sparse_rank_n_vv(normalise(ml).T, t, 10 ** -4, 2)
        ml_sparse = normalise(np.dot(normalise(ml).T, V).T)
        non_zero.append(np.sum(V.T[0] != 0))
        mse_.append(mse(analytical(ml_sparse[0], ml_sparse[1]), ml_sparse[0], ml_sparse[1]))
    
    return non_zero, mse_