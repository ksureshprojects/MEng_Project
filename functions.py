from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go


# def circular_coef(x, y):
#     """
#     Function that calculates modified spearman's 
#     rank to evaluate ellipse.

#     Parameters:
#     x, y - Vectors corresponding to angular positions 
#            and real collection times of samples.
#     Returns:
#     max_result - Modified correlation value.
#     p - Window size that returned best result.
#     """
#     # Storing vector length
#     l = x.shape[0]
#     # Variable to store best result
#     max_result = -np.inf
#     min_result = np.inf
#     # Variable to store turning point
#     # at best result
#     p = 0
#     p_m = 0
#     # Iterate over different window sizes
#     for i in range(2, l):
#         coefficients = np.array([])
#         # Calculate spearman correlation for
#         # each window of size i
#         for j in range(l//i):
#             c = stats.spearmanr(x[j*i:(j+1)*i], y[j*i:(j+1)*i]).correlation
#             coefficients = np.block([coefficients, c])
#         # Calculate mean result
#         mean = np.mean(coefficients)
#         # Update max/min if new best found
#         if mean > max_result:
#             max_result = mean
#             p = i
#         if mean < min_result:
#             min_result = mean
#             p_m = i
#     if abs(max_result) > abs(min_result):
#         return max_result, p
#     else:
#         return abs(min_result), p_m


# def circular_coef(x, y, spp=24):
#     """
#     Function that calculates modified spearman's 
#     rank to evaluate ellipse.

#     Parameters:
#     x, y - Vectors corresponding to angular positions 
#            and real collection times of samples.
#     Returns:
#     max_result - Modified correlation value.
#     p - Window size that returned best result.
#     """
#     # Storing vector length
#     l = x.shape[0]
#     # Store coefficients
#     coefficients = np.array([])
#     # Calculate spearman correlation for
#     # each window of size spp
#     for j in range(l//spp):
#         x_ = ((-2*np.pi - y[j*spp:(j+1)*spp][0]) + y[j*spp:(j+1)*spp])
#         y_ =  x_ - np.floor(x_/(2*np.pi))*2*np.pi
#         c = np.abs(stats.spearmanr(x[j*spp:(j+1)*spp], y_).correlation)
#         coefficients = np.block([coefficients, c])
#     # Calculate mean result
#     mean = np.mean(coefficients)
#     return mean, spp

def circular_coef(x, y, spp=24):
    """
    Function that calculates modified spearman's 
    rank to evaluate ellipse.

    Parameters:
    x, y - Vectors corresponding to angular positions 
           and real collection times of samples.
    Returns:
    max_result - Modified correlation value.
    p - Window size that returned best result.
    """
    # Storing vector length
    l = x.shape[0]
    # Store rank errors
    errors = np.array([])
    # Find rank error
    for j in range(l//spp):
        x_ = np.argsort(x[j*spp : (j+1)*spp])
        y_ = np.argsort(y[j*spp : (j+1)*spp])
        start = np.argwhere(y_ == 0)[0,0]
        y_ = np.block([y_[start:], y_[:start]])
        e = np.mean(np.abs(y_ - x_ + ((y_ - x_) < -spp/2)*spp - ((y_ - x_) > spp/2)*spp))
        errors = np.block([errors, e])
    # Calculate mean result
    mean = np.mean(errors)

    # Store rank errors
    errors = np.array([])
    # Find rank error
    for j in range(l//spp):
        x_ = np.argsort(x[j*spp : (j+1)*spp])
        y_ = np.argsort(-y[j*spp : (j+1)*spp])
        start = np.argwhere(y_ == 0)[0,0]
        y_ = np.block([y_[start:], y_[:start]])
        e = np.mean(np.abs(y_ - x_ + ((y_ - x_) < -spp/2)*spp - ((y_ - x_) > spp/2)*spp))
        errors = np.block([errors, e])
     # Calculate mean result
    mean_opp = np.mean(errors)

    if mean < mean_opp:
        return mean, spp, True
    else:
        return mean_opp, spp, False
    
def original_plot(X_orig, g, save=None):
    # No of samples
    l = X_orig.shape[1]
    # Initialise plot figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[i for i in range(l)], 
                             y=(X_orig[g] - np.mean(X_orig[g])),
                             mode='markers',
                             name="Gene {}".format(g),
                             hovertext=[i for i in range(l)]))
    # Format plot layout
    fig.update_layout(title="True Plot",
                      xaxis_title='Index',
                      yaxis_title='Gene Value',
                      showlegend=True)
    # Save plot
    if save is not None:
        fig.write_image(save + ".png")
    # Return plot
    fig.show()

def fit_ellipse(X, Y, plot=False, labels=None, save=None):
    """
    Solve Ax^2 + Bxy + Cy^2 + Dx +Ey = 1 to
    fit best ellipse to a plot.

    Parameters:
    X,Y - Vectors that contain coordinates
          that form elliptical plot.
    plot - Boolean value that plots result if True
    labels - Array with [title, xlabel, ylabel] 
    save - File to save plot
    Returns:
    MSE value and ellipse width if plot is False.
    """
    # Formulate and solve the least squares problem ||Ax - b ||^2
    X = X.reshape((X.shape[0], 1))
    Y = Y.reshape((Y.shape[0], 1))
    A = np.hstack([X**2, X * Y, Y**2, X, Y])
    b = np.ones_like(X)
    x = np.linalg.lstsq(A, b, rcond=None)[0].squeeze()
    # min(A,C)/max(A,C) - measure of ellipse width
    r = min(x[0], x[2])/max(x[0], x[2])
    if not plot:
        # Mean squared error value and ellipse width
        return (np.linalg.norm(np.dot(A, x) - b, ord=2) ** 2)/(b.shape[0]), r

    # Plot the least squares ellipse
    x_coord = np.linspace(np.min(X) - 0.5, np.max(X) + 0.5, 300)
    y_coord = np.linspace(np.min(Y) - 0.5, np.max(Y) + 0.5, 300)
    X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
    Z_coord = x[0] * X_coord ** 2 + x[1] * X_coord * Y_coord + \
        x[2] * Y_coord**2 + x[3] * X_coord + x[4] * Y_coord
    mse = round((np.linalg.norm(np.dot(A, x) - b, ord=2) ** 2)/(b.shape[0]), 2)
    plt.figure()
    plt.scatter(X, Y)
    plt.contour(X_coord, Y_coord, Z_coord, levels=[
                1], colors=('r'), linewidths=2)
    if labels is not None:
        plt.title(labels[0] + 
                  '. MSE Ellipse Fit: {}'.format(mse))
        plt.xlabel(labels[1])
        plt.ylabel(labels[2])
    if save is not None:
        plt.savefig(save)
    plt.show()
    print("MSE general elliptical fit: {}".format(mse))


def order_ellipse(X, Y, time, spp=24):
    """
    Parameters:
    X,Y - Vectors that contain coordinates
          that form elliptical plot.
    time - Real time stamps corresponding to 
           coordinates X,Y

    Returns:
    Index values of reordered data, quality of
    reoredring and window size used to calculate
    correlation coefficient.
    """
    # Solve least squares problem
    X = X.reshape((X.shape[0], 1))
    Y = Y.reshape((Y.shape[0], 1))
    A = np.hstack([X**2, X * Y, Y**2, X, Y])
    b = np.ones_like(X)
    x = np.linalg.lstsq(A, b, rcond=None)[0].squeeze()
    # Evaluate ellipse equation at all points
    x_coord = np.linspace(np.min(X) - 0.5, np.max(X) + 0.5, 300)
    y_coord = np.linspace(np.min(Y) - 0.5, np.max(Y) + 0.5, 300)
    X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
    Z_coord = x[0] * X_coord ** 2 + x[1] * X_coord * Y_coord + \
        x[2] * Y_coord**2 + x[3] * X_coord + x[4] * Y_coord
    # Identify contour where ellipse equation is equal to one
    X_coord = X_coord[np.where(np.abs(Z_coord - 1) < 10 ** -3)]
    Y_coord = Y_coord[np.where(np.abs(Z_coord - 1) < 10 ** -3)]
    # Identify ellipse center
    c_x = np.mean(X_coord)
    c_y = np.mean(Y_coord)

    # Create table with time, angular positions and indicies
    df = pd.DataFrame(columns=['Time', 'Angle', 'Index'])
    # Find angular position of each point
    for i in range(X.shape[0]):
        x = X[i][0] - c_x
        y = Y[i][0] - c_y

        if x == 0:
            if y > 0:
                a = np.pi/2
            else:
                a = np.pi/2 + np.pi
        elif y == 0:
            if x > 0:
                a = 0
            else:
                a = np.pi
        elif x > 0 and y > 0:
            a = np.arctan(y/x)
        elif x > 0 and y < 0:
            a = np.arctan(y/x) + 2*np.pi
        else:
            a = np.arctan(y/x) + np.pi

        line = pd.DataFrame(data=[[time[i], -a, i]],
                            columns=['Time', 'Angle', 'Index'])
        df = df.append(line, ignore_index=True)
    # Sort table by time
    df = df.sort_values(by='Time')
    # Calculate reordering quality
    quality, p, asc = circular_coef(df['Time'].values, df['Angle'].values, spp=spp)
    # Return index column after sorting by angle, 
    # quality metric and window size
    return df.sort_values(by='Angle', ascending=asc)['Index'].values.astype('int'), quality, p


# def plot_scatter(X_, time, genes, title, save=None):
def plot_scatter(X_, genes, title, save=None):
    """
    Create scatter plot of gene expression series
    at uniform intervals.

    Parameters:
    X - Gene expression data set
    genes - Indicies of genes to plot or 
            array of tuples of form
            (gene index, reordered indicies)
    title - Plot title
    """
    # Center Data
    X = X_ - np.mean(X_, axis=1, keepdims=True)

    t = X.shape[1]
    plt.figure()
    for g in genes:
        if isinstance(g, int):
            l = g
        else:
            l = g[0]
        plt.scatter([i for i in range(t)], X[g], 
                    label="Gene {}".format(l))
    plt.xlabel('Index')
    plt.ylabel('Gene Value')
    plt.title(title)
    plt.legend()
    plt.savefig(save)

def analyse(X, data, top_g1, top_g2, time, spp, save, p1=None, p2=None, plot=True):
    ind, q, w = order_ellipse(X[top_g1], X[top_g2], time, spp=spp)
    print('Reordering Rank Error: {}, Samples per Period: {}'.format(round(q,5), w))
    if p1 is None or p2 is None:
        p1 = top_g1
        p2 = top_g2
    if plot:
        plot_scatter(data,
                [(p1), (p2)],
            'Unordered Data', save=save + '_unordered')
        plot_scatter(data,
                [(p1, ind), (p2, ind)],
            'Reordered Data', save=save + '_reordered')
    else:
        return q

#     # Initialise plot figure
#     fig = go.Figure()
    
#     for g in genes:
#         if isinstance(g, int):
#             l = g
#         else:
#             l = g[0]
#             if len(g) > 1:
#                 time = time[g[1]]
        
#         fig.add_trace(go.Scatter(x=[i for i in range(t)], y=X[g],
#                                          mode='markers',
#                                          name="Gene {}".format(l),
#                                          hovertext=[t for t in time]))
#     # Format plot layout
#     fig.update_layout(title=title,
#                       xaxis_title='Index',
#                       yaxis_title='Gene Value',
#                       showlegend=True)
#     # Return plot
#     fig.show()
    

########################################################################
########################################################################
########################################################################

def OSCOPE(X_, time, break_point=100, med=False, best=True, save=None, spp=24, cen=True, an2=None):
    """
    Carry out OSCOPE method to identify
    pair of genes that form best elliptical
    plot.

    Parameters:
    X - Data set of gene expression series
    br - Number of genes to consider
    time - Corresponding array of real collection
           times.
    """
    if cen:
        # Center Data
        X = X_ - np.mean(X_, axis=1, keepdims=True)
    else:
        X = X_.copy()
    # Number of genes
    l = X.shape[0]
    
    if not med:
        X_by_variance = np.argsort(np.var(X, axis=1))[::-1]
        X = X_ - np.mean(X_, axis=1, keepdims=True)
    else:
        h = l//2 - 1
        # Sort gene series by median,
        # store indicies
        X_by_median = np.argsort(np.median(X, axis=1))[h:]
        X = X_ - np.mean(X_, axis=1, keepdims=True)
        # Sort indicies, by variance, 
        # in descending order
        X_by_variance = X_by_median[np.argsort(np.var(X[X_by_median, :], axis=1))[::-1]]
        l = X_by_variance.shape[0]

    # Test every pair of genes for elliptical fit
    col = ['Gene 1', 'Gene 2', 'Ellipse Width', 'MSE']
    df = pd.DataFrame(columns=col)
    # Range to explore
    r = min(break_point,l)
    for i in range(r):
        # Show progress
        print('\r {}/{}'.format(i+1, min(break_point,l)), end='')
        for j in range(i + 1, r):
            g1, g2 = (X_by_variance[i], X_by_variance[j])
            mse_, est = fit_ellipse(X[g1], X[g2])
            line = pd.DataFrame(data=[[g1, g2, est, mse_]], 
                                columns=col)
            df = df.append(line, ignore_index=True)
    # Sort pairs by elliptical fit quality
    df.sort_values(by=['MSE'], inplace=True, ascending=True)
    df = df.reset_index(drop=True)

    top_g1 = df['Gene 1'][0]
    top_g2 = df['Gene 2'][0]
    
    if an2:
        fit_ellipse(X[top_g1], 
            X[top_g2], 
            plot=True, 
            labels=['Best Ellipse - ' + an2,
                    'Gene {}'.format(top_g1),
                    'Gene {}'.format(top_g2)], 
            save=save)
        analyse(X, X_, top_g1, top_g2, time, spp, save, plot=False)
        return None

    if best:
        # Ensure filename exists
        if save is None:
            save = 'OSCOPE_' + \
                   str(int((np.random.uniform()*100)//1))
        # Find best elliptical fit
        fit_ellipse(X[top_g1], 
            X[top_g2], 
            plot=True, 
            labels=['Best Ellipse',
                    'Gene {}'.format(top_g1),
                    'Gene {}'.format(top_g2)], 
            save=save)
        # Use ellipse to reorder data
        analyse(X, X_, top_g1, top_g2, time, spp, save)

    # Classify genes by group
    class Node():
        def __init__(self, gene):
            self.g = gene
            self.n = []

    node_list = {}
    for i, row in df.iterrows():
        if i == break_point:
            break
        if row['Gene 1'] not in node_list:
            node_list[row['Gene 1']] = Node(row['Gene 1'])
            node_list[row['Gene 1']].n.append(row['Gene 2'])
        else:
            node_list[row['Gene 1']].n.append(row['Gene 2'])

        if row['Gene 2'] not in node_list:
            node_list[row['Gene 2']] = Node(row['Gene 2'])
            node_list[row['Gene 2']].n.append(row['Gene 1'])
        else:
            node_list[row['Gene 2']].n.append(row['Gene 1'])

    seen = {}
    for g in node_list:
        if g not in seen:
            for n in node_list[g].n:
                seen[n] = ''

    for n in seen:
        node_list.pop(n)

    # Return table of gene pairs and
    # dictionary classifying gene groups
    return df, {i: sorted(node_list[key].n + [key]) \
                for i, key in enumerate(list(node_list))}

########################################################################
########################################################################
########################################################################

def PCA_ellipse(X_, time, save=None, g1=None, g2=None, spp=24):
    """
    Plot first two principal components of data, 
    where genes are variables (columns) and samples
    are inputs (rows), against one another.
    """
    # Center data so mean of all gene series are zero,
    # Take transpose of data so genes > columns
    X = X_ - np.mean(X_, axis=1, keepdims=True)

    # image: m x n, U: m x m, s: min(n, m) vector, V: n x n
    U, s, Vh = np.linalg.svd(X.T)

    # First n_component rows of VT. Rows of VT are principal component directions.
    Vh = Vh[:2, :]

    # Reduced Data
    U2 = X.dot(Vh.T).T
    
    # Ensure filename exists
    if save is None:
        save = 'PCA_E_' + \
                str(int((np.random.uniform()*100)//1))
    # Find best elliptical fit
    fit_ellipse(U2[0], 
        U2[1], 
        plot=True, 
        labels=['PCA - Elliptical Plot',
                'First Component',
                'Second Component'], 
        save=save)
    # Use ellipse to reorder data
    analyse(U2, X, 0, 1, time, spp, save, p1=g1, p2=g2)

def OSCOPE_PCA(X_, time, break_point=20, spp=24, med=False, best=True, save=None):
    """
    Carry out OSCOPE method to identify
    pair of genes that form best elliptical
    plot. Use first PCA component loading
    vector values to identify OSCOPE subset.

    Parameters:
    X - Data set of gene expression series
    br - Number of genes to consider
    time - Corresponding array of real collection
           times.
    """
    # Center data so mean of all gene series are zero,
    # Take transpose of data so genes > columns
    X = X_ - np.mean(X_, axis=1, keepdims=True)
    X_P = X.T
    # Number of genes
    l = X.shape[0]
    # image: m x n, U: m x m, s: min(n, m) vector, V: n x n
    U, s, Vh = np.linalg.svd(X_P)

    # First n_component rows of VT. Rows of VT are principal component directions.
    Vh = Vh[0]
    X_by_variance = np.argsort(np.abs(Vh))[::-1]

    # Test every pair of genes for elliptical fit
    col = ['Gene 1', 'Gene 2', 'Ellipse Width', 'MSE']
    df = pd.DataFrame(columns=col)
    # Range to explore
    r = min(break_point,l)
    for i in range(r):
        # Show progress
        print('\r {}/{}'.format(i+1, min(break_point,l)), end='')
        for j in range(i + 1, r):
            g1, g2 = (X_by_variance[i], X_by_variance[j])
            mse_, est = fit_ellipse(X[g1], X[g2])
            line = pd.DataFrame(data=[[g1, g2, est, mse_]], 
                                columns=col)
            df = df.append(line, ignore_index=True)
    # Sort pairs by elliptical fit quality
    df.sort_values(by=['MSE'], inplace=True, ascending=True)
    df = df.reset_index(drop=True)

    top_g1 = df['Gene 1'][0]
    top_g2 = df['Gene 2'][0]
    if best:
        # Ensure filename exists
        if save is None:
            save = 'OSCOPE_PCA_' + \
                   str(int((np.random.uniform()*100)//1))
        # Find best elliptical fit
        fit_ellipse(X[top_g1], 
            X[top_g2], 
            plot=True, 
            labels=['OSCOPE_PCA - Best Ellipse',
                    'Gene {}'.format(top_g1),
                    'Gene {}'.format(top_g2)], 
            save=save)
        # Use ellipse to reorder data
        analyse(X, X, top_g1, top_g2, time, spp, save)

    # Classify genes by group
    class Node():
        def __init__(self, gene):
            self.g = gene
            self.n = []

    node_list = {}
    for i, row in df.iterrows():
        if i == break_point:
            break
        if row['Gene 1'] not in node_list:
            node_list[row['Gene 1']] = Node(row['Gene 1'])
            node_list[row['Gene 1']].n.append(row['Gene 2'])
        else:
            node_list[row['Gene 1']].n.append(row['Gene 2'])

        if row['Gene 2'] not in node_list:
            node_list[row['Gene 2']] = Node(row['Gene 2'])
            node_list[row['Gene 2']].n.append(row['Gene 1'])
        else:
            node_list[row['Gene 2']].n.append(row['Gene 1'])

    seen = {}
    for g in node_list:
        if g not in seen:
            for n in node_list[g].n:
                seen[n] = ''

    for n in seen:
        node_list.pop(n)

    # Return table of gene pairs and
    # dictionary classifying gene groups
    return df, {i: sorted(node_list[key].n + [key]) \
                for i, key in enumerate(list(node_list))}

########################################################################
########################################################################
########################################################################

def center(X_):
    return X_ - np.mean(X_, axis=1, keepdims=True)

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

# def sparse_rank_n_uv(X, t=20, tol=10e-4):
#     X_ = X.copy().T
    
#     Vh = []
#     X_r = []
#     for i in range(2):
#         X_ = center(X_.T).T
#         v, u = sparse(X_, t, tol)
#         Vh.append(v.T[0])
#         X_r.append(np.dot(X_, v).T[0])
#         X_ = X_ - np.dot(np.dot(u.T, X_),v) * np.dot(u, v.T)
    
#     return np.array(Vh), np.array(X_r)

def sparse_rank_n_uv(X, t=20, tol=10e-4, r=2, scl=True):
    X_ = X.copy()
    
#     Uh = []
    Vh = []
    X_r = []
    for i in range(r):
        v, u = sparse(normalise_(X_, scl=scl), t, tol)
#         Uh.append(u.T[0])
        Vh.append(v.T[0])
        X_r.append(np.dot(X_, v).T[0])
        X_ = X_ - np.dot(np.dot(u.T, X_),v) * np.dot(u, v.T)
    
    return np.array(Vh).T, np.array(X_r)

def normalise_(X_, scl=True):
    X_n = X_.T
    for i in range(len(X_n)):
        if scl:
            ran = np.max(X_n[i]) - np.min(X_n[i])
            X_n[i] = X_n[i]/(ran/2 + 0.000001)
        X_n[i] = X_n[i] - np.mean(X_n[i])
    return X_n.T

def SPCA_ellipse(X_, time, t=20, spp=24, save=None, g1=None, g2=None, scl=True, med=False, 
                 an=False, cen=True, an2=False):
    """
    Plot first two sparse principal components of data, 
    where genes are variables (columns) and samples
    are inputs (rows), against one another.
    """
    if cen:
        # Center Data
        X = X_ - np.mean(X_, axis=1, keepdims=True)
    else:
        X = X_.copy()
    # Number of genes
    l = X.shape[0]
    if med:
        h = l//2 - 1
        # Sort gene series by median,
        # store indicies
        X_by_median = np.argsort(np.median(X, axis=1))[h:]
        X = X[X_by_median, :]

    # Run Sparse PCA
    Vh, U2 = sparse_rank_n_uv(X.T, t=t, scl=scl)
    
    # Ensure filename exists
    if save is None:
        save = 'SPCA_E_' + \
                str(int((np.random.uniform()*100)//1))
    if an2:
        fit_ellipse(U2[0], 
            U2[1], 
            plot=True, 
            labels=['SPCA - ' + an2,
                    'First Component',
                    'Second Component'], 
            save=save)
        # Use ellipse to reorder data
        r = analyse(U2, X_, 0, 1, time, spp, save, p1=g1, p2=g2, plot=False)
        return None
        
    # Find best elliptical fit
    if an:
        mse, w = fit_ellipse(U2[0], 
            U2[1], 
            plot=(not an), 
            labels=['SPCA - Elliptical Plot',
                    'First Component',
                    'Second Component'], 
            save=save)
    else:
        fit_ellipse(U2[0], 
            U2[1], 
            plot=(not an), 
            labels=['SPCA - Elliptical Plot',
                    'First Component',
                    'Second Component'], 
            save=save)
    # Use ellipse to reorder data
    r = analyse(U2, X_, 0, 1, time, spp, save, p1=g1, p2=g2, plot=(not an))

    if an:
        return mse, r, w

def OSCOPE_SPCA(X_, time, t=20, spp=24, break_point=20, med=False, best=True, 
                scl=True, save=None, cen=True):
    """
    Carry out OSCOPE method to identify
    pair of genes that form best elliptical
    plot. Use first SPCA component loading
    vector values to identify OSCOPE subset.

    Parameters:
    X - Data set of gene expression series
    br - Number of genes to consider
    time - Corresponding array of real collection
           times.
    """
    if cen:
        # Center Data
        X = X_ - np.mean(X_, axis=1, keepdims=True)
    else:
        X = X_.copy()
    # Number of genes
    l = X.shape[0]
    if med:
        h = l//2 - 1
        # Sort gene series by median,
        # store indicies
        X_by_median = np.argsort(np.median(X, axis=1))[h:]
        X = X[X_by_median, :]
        # Center Data
        X = X_ - np.mean(X_, axis=1, keepdims=True)

    # Run Sparse PCA
    V, U2 = sparse_rank_n_uv(X.T, t=t, scl=scl)

    # First n_component rows of VT. Rows of VT are principal component directions.
    V1 = V.T[0]
    X_by_variance = np.argsort(np.abs(V1))[::-1]

    # Test every pair of genes for elliptical fit
    col = ['Gene 1', 'Gene 2', 'Ellipse Width', 'MSE']
    df = pd.DataFrame(columns=col)
    # Range to explore
    r = min(break_point,l)
    for i in range(r):
        # Show progress
        print('\r {}/{}'.format(i+1, min(break_point,l)), end='')
        for j in range(i + 1, r):
            g1, g2 = (X_by_variance[i], X_by_variance[j])
            mse_, est = fit_ellipse(X[g1], X[g2])
            line = pd.DataFrame(data=[[g1, g2, est, mse_]], 
                                columns=col)
            df = df.append(line, ignore_index=True)
    # Sort pairs by elliptical fit quality
    df.sort_values(by=['MSE'], inplace=True, ascending=True)
    df = df.reset_index(drop=True)

    top_g1 = df['Gene 1'][0]
    top_g2 = df['Gene 2'][0]
    if best:
        # Ensure filename exists
        if save is None:
            save = 'OSCOPE_SPCA_' + \
                   str(int((np.random.uniform()*100)//1))
        # Find best elliptical fit
        fit_ellipse(X[top_g1], 
            X[top_g2], 
            plot=True, 
            labels=['OSCOPE_SPCA - Best Ellipse',
                    'Gene {}'.format(top_g1),
                    'Gene {}'.format(top_g2)], 
            save=save)
        # Use ellipse to reorder data
        analyse(X, X, top_g1, top_g2, time, spp, save)

    # Classify genes by group
    class Node():
        def __init__(self, gene):
            self.g = gene
            self.n = []

    node_list = {}
    for i, row in df.iterrows():
        if i == break_point:
            break
        if row['Gene 1'] not in node_list:
            node_list[row['Gene 1']] = Node(row['Gene 1'])
            node_list[row['Gene 1']].n.append(row['Gene 2'])
        else:
            node_list[row['Gene 1']].n.append(row['Gene 2'])

        if row['Gene 2'] not in node_list:
            node_list[row['Gene 2']] = Node(row['Gene 2'])
            node_list[row['Gene 2']].n.append(row['Gene 1'])
        else:
            node_list[row['Gene 2']].n.append(row['Gene 1'])

    seen = {}
    for g in node_list:
        if g not in seen:
            for n in node_list[g].n:
                seen[n] = ''

    for n in seen:
        node_list.pop(n)

    # Return table of gene pairs and
    # dictionary classifying gene groups
    return df, {i: sorted(node_list[key].n + [key]) \
                for i, key in enumerate(list(node_list))}