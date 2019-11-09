from matplotlib import pyplot as plt
from functools import reduce
import operator as op
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import sys

def SVD(X_, n_component=None, V=None):
    # If loading matrix V supplied, only returns principal components for input set. 
    # If V not supplied, V matrix first calculated by performing SVD on entire input set.
    if V is None:
        # image: m x n, U: m x m, s: min(n, m) vector, V: n x n
        U, s, Vh = np.linalg.svd(X_)

        # First n_component rows of VT. Rows of VT are principal component directions.
        Vh = Vh[:n_component, :]

        # Reduced Data
        return X_.dot(Vh.T), Vh.T, s
    else:
        return X_.dot(V)
    
def plot_loadings(v, osc, c):
    colors = []
    for i in range(len(v)):
        if i in osc[1]:
            colors.append('crimson')
        elif i in osc[2]:
            colors.append('blue')
        else:
            colors.append('lightslategray')

    fig = go.Figure(data=[go.Bar(
        x=[i + 1 for i in range(len(v))],
        y=v,
        marker_color=colors # marker color can be a single color value or an iterable
    )])
    fig.update_layout(title_text='Loadings of component: {}'.format(c + 1),
                      xaxis_title="Gene",
                      yaxis_title="Loading value")
    return fig

def normalise(X_):
    X_n = X_.copy()
    for i in range(len(X_n)):
        ran = np.max(X_n[i]) - np.min(X_n[i])
        X_n[i] = X_n[i]/(ran/2 + 0.000001)
        X_n[i] = X_n[i] - np.mean(X_n[i])

    return X_n