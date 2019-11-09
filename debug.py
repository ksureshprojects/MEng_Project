import numpy as np
import random
from matplotlib import pyplot as plt
import pandas as pd
import operator as op
from functools import reduce
from scipy.optimize import minimize
from Data import *
from helpers import *
from PCA import *
from Sparse import *
import warnings
from sklearn.cluster import KMeans
import sys

# n_genes = 100
# n_cells = 50
# groups = 3 # one non oscilating group and two non-oscilating groups.
# data = Data(n_genes, n_cells)
# data.osc_groups(groups)

# X, osc = data.gen_data(t_dist_normal=False)
# time = data.t
# print("Shape of Data: {}".format(X.shape))
# print("Group Frequencies: {}".format(data.groups))
# print("Genes per group:")

X = np.loadtxt("foo.csv", delimiter=",")

v_n_n_n = optimal_t(normalise(X).T, 0.0001, 50)
