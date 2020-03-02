import numpy as np

def soft_thresh(x, l):
    # return np.array(list(map(lambda v: [np.sign(v[0]) * max(abs(v[0]) - l, 
    #                                                         0)], x)))
    x_ = np.invert((x < l) * (x > - l)) * x
    return x_ - (x_ > l) * l + (x_ < -l) * l

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

def sparse(X, t, tol, r=False):
    # Shuffle times
    if r:
        random = np.random.RandomState(seed=0)
        v = np.array([[random.rand() for i in range(X.shape[1])]]).T
    else:
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

def sparse_rank_n_uv(X, t=20, tol=10e-4, r=2, scl=False, std=False, r=False):
    X_ = X.copy()
    Vh = []
    X_r = []
    for i in range(r):
        v, u = sparse(normalise_(X_, scl=scl, std=std), t, tol, r=r)
        Vh.append(v.T[0])
        X_r.append(np.dot(X_, v).T[0])
        X_ = X_ - np.dot(np.dot(u.T, X_),v) * np.dot(u, v.T)
    return np.array(Vh).T, np.array(X_r)

def normalise_(X_, scl=False, std=False):
    X_n = X_.T
    # Center data
    X_n = X_n - np.mean(X_n, axis=1, keepdims=True)
    if std:
        # Standardise every gene series to have variance of 1
        X_n = X_n / np.std(X_n, axis=1, keepdims=True)
    elif scl:
        # Scale range so that range = 2
        X_n = X_n / (np.max(X_n, axis=1, keepdims=True) - 
                            np.min(X_n, axis=1, keepdims=True))
        X_n = X_n * 2
        # Recenter so that all gene values lie between [-1,1]
        X_n = X_n - np.mean(X_n, axis=1, keepdims=True)
    return X_n.T