from matplotlib import pyplot as plt
from functools import reduce
import operator as op
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import sys
from scipy.optimize import minimize
from ellipse import *

def ncr(n, r):
    """
    Function that computes n choose r. 
    """
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom


def plot_X(X, ind, time, title):
        """
        Function that plots gene expression series against time. 
        
        Parameters:
            X - Dataset
            ind - Indicies of X to plot.
            time - Time values.
            title - Plot title.
        Returns:
            Plotly figure object.
        """
        # Check if input paramaters are valid and consistent.
        if len(time) != X.shape[1]:
            raise ValueError("Length of time array and length of series dont match")
        if len(ind) > X.shape[0] or max(ind) > X.shape[0] - 1 or min(ind) < 0:
            raise ValueError("Index doesn't exist")
        # Initialise plot figure
        fig = go.Figure()
        # Create scatter plot for each gene expression series referenced in ind.
        for i in ind:
            fig.add_trace(go.Scatter(x=time, y=X[i],
                                     mode='markers',
                                     name="Gene {}".format(i + 1)))
        # Format plot layout
        fig.update_layout(title=title,
                          xaxis_title="Time",
                          yaxis_title="Gene Expression Value",
                          showlegend=True)
        # Return plot
        return fig
        
        
def plot_ellipse(X, ind, title, axes=None):
    """
    Function that plots pairs of gene expression series against one another. 
        
    Parameters:
        X - Dataset
        ind - Pairs of indicies of X
    """
    if axes is None:
        x_axes = "First gene in pair"
        y_axes = "Second gene in pair"
    else:
        x_axes = axes[0]
        y_axes = axes[1]
    
    # Check if input paramaters are valid and consistent.
    g = X.shape[0]
    if len(ind) > ncr(g, 2):
        raise ValueError("Too many pair values specified")
    # Initialise plot figure
    fig = go.Figure()
    # Create scatter plot for each pair of gene expression series referenced in ind.
    for tup in ind:
        i, j = tup
        if 0 <= i < g and 0 <= j < g:
            fig.add_trace(go.Scatter(x=X[i], y=X[j],
                                     mode='markers',
                                     name="Row {} & Row {}".format(i, j),
                                     hovertext=[ind for ind in range(len(X[i]))]))
        else:
            raise ValueError("Indicies out of range")
    # Format plot layout
    fig.update_layout(title=title,
                      xaxis_title=x_axes,
                      yaxis_title=y_axes,
                      showlegend=True)
    # Return plot
    return fig
             
            
def plot_mse(X, ind):
    """
    Function that plots mse vs phase difference plot for pairs of gene expression series. 
        
    Parameters:
        X - Dataset
        ind - Pairs of indicies of X
    """
    phases = np.linspace(0, 2 * np.pi, 200)
    fig = go.Figure()
    for tup in ind:
        i, j = tup
        mse_val = [mse(psi, X[i], X[j]) for psi in phases]
        fig.add_trace(go.Scatter(x=phases, y=mse_val,
                                 mode='lines',
                                 name="Gene {} & Gene {} (min mse: {})".format(i + 1, j + 1, np.round(min(mse_val),5))))
    fig.update_layout(title="MSE vs Phase",
                   xaxis_title="Phase",
                   yaxis_title="MSE")
    return fig

    
def mse(psi, X1, X2):
    """ 
    Calculate the MSE term given value of psi.
    
    Parameters:
        X1, X2 - Pair of gene expression series to fit elliptical equation.
    Returns:
        mse value.
    """
    if len(X1) != len(X2):
        raise ValueError("Lengths of input don't match")
    
    s_vec = (np.square(X1) + np.square(X2) - 1
             - 2 * np.multiply(X1, X2) * np.cos(psi)
             + np.cos(psi) ** 2)
    
    return np.dot(s_vec, s_vec)

# def mse(psi, Xg1, Xg2):
#     """ Calculate the MSE term given this value of psi """
#     if len(Xg1) != len(Xg2):
#         # Check that both genes have an equal number of expressions
#         raise ValueError
    
#     mse = 0
#     for i in range(len(Xg1)):
#         # Sum the squared term for each cell
#         mse += (Xg1[i]**2 + Xg2[i]**2 - 2*Xg1[i]*Xg2[i]*np.cos(psi) - np.sin(psi)**2)**2
        
#     return mse


def analytical(X1, X2):
    """
    Function that uses analytical method to calculate optimal phase difference between  
    two gene expression series that should fit an elliptical plot.
    
    Parameters:
        X1, X2 - Two gene series datasets that are part of the same oscillatory group.
    
    Returns:
        Array of possible optimal phase difference values between input gene expression series.
    """
    if len(X1) != len(X2):
        raise ValueError("Lengths of input don't match")
    a = len(X1)
    b = -3 * np.dot(X1, X2)
    c = (2 * np.dot(np.square(X1), np.square(X2))
         + np.dot(X1, X1) + np.dot(X2, X2) - len(X1))
    d = -np.dot(np.multiply(X1, X2), 
                np.square(X1) + np.square(X2) - 1)
    
    return min([np.arccos(float(i)) for i in np.roots([a, b, c, d]) if np.isreal(i) and - 1 < float(i) < 1], key=lambda psi: mse(psi, X1, X2))


def ellipse_eqn(psi, X1, X2):
    """
    Evaluate ellipse gradient equation.
    
    Parameters:
        X1, X2 - Two gene series datasets that are part of the same oscillatory group.
        psi - Phase difference
    
    Returns:
        Result of ellipse gradient equation.
    """
    if len(X1) != len(X2):
        raise ValueError("Lengths of input don't match")
    a = len(X1)
    b = 3 * np.dot(X1, X2)
    c = (2 * np.dot(np.square(X1), np.square(X2))
         + np.dot(X1, X1) + np.dot(X2, X2) - len(X1))
    d = -np.dot(np.multiply(X1, X2), 
                np.square(X1) + np.square(X2) - 1)
    x = np.cos(psi)
    
    return a * (x ** 3) + b * (x ** 2) + c * (x) + d


def numerical(X1, X2):
    """
    Function that uses line search method to calculate optimal phase difference between  
    two gene expression series that should fit an elliptical plot.
    
    Parameters:
        X1, X2 - Two gene series datasets that are part of the same oscillatory group.
    
    Returns:
        Optimal phase difference values between input gene expression series.
    """
    psi0 = 0
    for j in range(5):
        i = int(np.floor(np.random.random() * len(X1)))
        a = 1
        b = -2 * X1[i] * X2[i]
        c = X1[i] ** 2 + X2[i] ** 2 - 1
        psi0 += np.arccos((-b + (b ** 2 - 4 * a * c) ** 0.5)/(2 * a))
    psi0 = psi0/5
    
    opt = minimize(mse, psi0, args=(X1, X2))['x'][0]
    return min(opt, 2 * np.pi - opt)


def validate(method, osc_groups, X, true_psi):
    """
    Function that validates the results of search methods.
    
    Parameters:
        method - Function object corresponding to method to be tested. 
        osc_groups - Dictionary with oscilatory groups as keys and gene indicies arrays as values.
        X - Gene expression dataset.
        true_psi - Array with simulation phase of each gene expression series.
    Returns:
        DataFrame which outlines whether true phase difference and estimated phase differences are similar.
    """
    col = ['Gene 1', 'Gene 2', 'Group', 'True Phase Difference', 'Estimated Difference','Similar','MSE']
    df = pd.DataFrame(columns=col)
    for group in osc_groups:
        if group == 0:
            continue
        osc_ind = osc_groups[group]
        for i in range(len(osc_ind)):
            for j in range(i + 1, len(osc_ind)):
                gene1 = osc_ind[i]
                gene2 = osc_ind[j]
                true = min(2 * np.pi - abs(true_psi[gene1] - true_psi[gene2]), abs(true_psi[gene1] - true_psi[gene2]))
                est = method(X[gene1], X[gene2])
                mse_ = mse(est, X[gene1], X[gene2])
                similar = np.isclose(true, est)
                line = pd.DataFrame(data=[[gene1 + 1, gene2 + 1, group, true, est, similar, mse_]], columns=col)
                df = df.append(line, ignore_index=True)
                
    df.sort_values(by=['MSE'], inplace=True)
    return df

def cluster_1D(arr):
    # Clustering 1D Sorted Array into two clusters By Minimising MSE
    min_e = sys.float_info.max
    for i in range(len(arr)):
        # Minimising weighted sum of variances of cluster square of variation values.
        # curr_e = ((i - 1) * np.var(np.square(np.array(arr[:i]) - np.mean(arr[:i]))) +
                    # (len(arr) - i + 1) * np.var(np.square(np.array(arr[i:]) - np.mean(arr[i:]))))
        # Minimising euclidean distance between cluster values and cluster means 
        # with double penalty in non-oscillatory cluster.
        curr_e = ((i - 1) * np.var(arr[:i]) +
                  2 * (len(arr) - i + 1) * np.var(arr[i:]))
        if curr_e < min_e:
            min_e = curr_e
            break_point = i
    return break_point

def sort_by_variance(X):
    X_by_variance = [i for i in range(X.shape[0])]
    X_by_variance.sort(key=lambda i: np.var(X[i]), reverse=True)
    std_devs = [np.std(X[i]) for i in X_by_variance]
    return X_by_variance, std_devs

def OSCOPE(X_noisy, method, br):

    X_by_variance = [i for i in range(len(X_noisy))]
    X_by_variance.sort(key=lambda i: np.var(X_noisy[i]), reverse=True)
    std_devs = [np.std(X_noisy[i]) for i in X_by_variance]
    
    break_point = br
    
    col = ['Gene 1', 'Gene 2', 'Estimated Phase Difference','MSE']
    df = pd.DataFrame(columns=col)
    for i in range(break_point):
        for j in range(i + 1, break_point):
            g1, g2 = (X_by_variance[i], X_by_variance[j])
#             est = method(X_noisy[g1], X_noisy[g2])
#             mse_ = mse(est, X_noisy[g1], X_noisy[g2])
            mse_, est = fit_ellipse(X_noisy[g1], X_noisy[g2])
            line = pd.DataFrame(data=[[g1, g2, est, mse_]], columns=col)
            df = df.append(line, ignore_index=True)
                
    df.sort_values(by=['MSE'], inplace=True, ascending=True)
    df = df.reset_index(drop=True)
    
    break_point = cluster_1D(df['MSE'].values)
    
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
        
#         if i == df.shape[0] - 1:
#             break
#         if i > 0 and (df['MSE'].iloc[i + 1] - row['MSE'])/(row['MSE'] - df['MSE'].iloc[i - 1]) > mse_tol:
#             break
    
    seen = {}
    for g in node_list:
        if g not in seen:
            for n in node_list[g].n:
                seen[n] = ''
                
    for n in seen:
        node_list.pop(n)
    
    return df, {i + 1:sorted(node_list[key].n + [key]) for i, key in enumerate(list(node_list))}

def accuracy(groups, osc):
    t_count = 0
    total = 0
    for t in osc:
        if t != 0:
            t_count += 1
            max_score = sys.float_info.min
            for r in groups:
                plus_count = 0
                gr = 0
                gt = 0
                r_l = len(groups[r])
                t_l = len(osc[t])
                while gt < t_l:
                    if gr == r_l:
                        break
                    if groups[r][gr] == osc[t][gt]:
                        gr += 1
                        gt += 1
                        plus_count += 1
                    elif groups[r][gr] > osc[t][gt]:
                        gt += 1
                    else:
                        gr += 1
                score = (plus_count - abs(r_l - t_l))/t_l
                if score > max_score:
                    max_score = score
            total += max_score
            
    return total/t_count

def centre_ellipse(X_):
    X = X_.copy()
    # Find centre
    x_center = (np.max(X[0]) + np.min(X[0]))/2
    y_center = (np.max(X[1]) + np.min(X[1]))/2
    # Center Data
    X[0] = X[0] - x_center
    X[1] = X[1] - y_center
    
    return X