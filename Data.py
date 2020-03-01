import numpy as np
import random
from matplotlib import pyplot as plt

class Data:
    def __init__(self, n_genes, n_cells):
        """
        Constructor that creates data generator class. 
        
        Parameters:
            n_genes - Number of genes in data set
            n_cells - Number of collection samples.
        """
        self.ng = n_genes
        self.nc = n_cells
        self.psi = [random.random()*np.pi for i in range(self.ng)]
        self.groups = None
        self.t = None
        
    def osc_groups(self, n_groups, arr=None):
        """
        Function that assigns self.groups property with an array specifying the frequencies of each group. 

        Parameters:
            n_groups - To specify number of oscillatory groups wanted in data.
            arr - Option to individually specify each group frequency in an array.
        """
        if arr is not None:
            if len(arr) == n_groups:
                if all(isinstance(x, (float)) for x in arr):
                    self.groups = arr
                    # Append frequency value for non-oscillatory genes.
                    self.groups.insert(0, 0.0)
                else:
                    raise ValueError('Invalid frequency value(s) specified')
            else:
                raise ValueError('Number of groups specified does not match length of frequency list')
        else:
            # Frequency values are evenly distributed [0,1)
            self.groups = [min(max(0.25, 0.5 + np.random.normal(scale=0.2)), 0.75) for i in range(n_groups - 1)]
            self.groups.insert(0, 0.0) 
    
    def gen_data(self, t_dist_normal=True, sample_mean=np.pi * 2, 
                 st_dev=3, tol=0.9):
        """
        Function that assigns self.groups property with an array specifying the frequencies of each group. 
        
        Parameters:
            t_dist_normal - Option to specify how sample times are distributed. True for normal, False for uniform.
            sample_mean - Mean sample collection time.
            st_dev - Sample collection time standard deviation for sampling from normal distribution.
        Returns:
            X - Generated data matrix size n_genes x n_cells
            group_ind - Dictionary with key being oscillatory group number (or -1 for non-osciallatory group) and values
                        being all row indicies in dataset belonging to group corresponding to key.
        """
        if self.groups is None:
            raise ValueError('Run osc_groups method before gen_data')
        
        # Collection times of samples are not generally uniformly distributed.
        if t_dist_normal:
            self.t = [sample_mean + np.random.normal(scale=st_dev) for i in range(self.nc)]
        else:
            # Option to sample uniformly for testing purposes
            # Samples only generated from one cycle
            self.t = np.array([i for i in range(self.nc)])
            np.random.shuffle(self.t)
            
        X = []
        group_ind = {}
        for i in range(self.ng):
            gene = []
            # 90% of genes are non-oscillatory
            if np.random.random() < tol:
                # Store index values of genes of non-oscillatory group
                if 0 in group_ind:
                    group_ind[0].append(i)
                else:
                    group_ind[0]= [i]
                # Generate gene expression value for all cell sample collection times
                for t in self.t:
                    gene.append(np.sin(self.groups[0] * t + self.psi[i]))
            else:
                # Determine oscillatory group of gene
                g = int(np.floor(np.random.random() * (len(self.groups) - 1)) + 1)
                # Store index values of genes and corresponding oscillatory group
                if g in group_ind:
                    group_ind[g].append(i)
                else:
                    group_ind[g]= [i]
                # Generate gene expression value for all cell sample collection times
                for t in self.t:
                    gene.append(np.sin(self.groups[g] * t + self.psi[i]))
            # Add gene row to data matrix
            X.append(gene)
        
        # Return n_genes x n_cells matrix
        return np.array(X), group_ind
            
    def corrupt(self, X, st_dev=0.05):
        """
        Function that adds noise to data set. 
        
        Parameters:
            X - Dataset
            st_dev - Gaussian noise standard deviation.
        Returns:
            X_noisy - Noisy data set.
        """
        X_noisy = np.copy(X)
        # Add gaussian noise to every data entry
        for row in X_noisy:
            for i in range(len(row)):
                row[i] += np.random.normal(scale=st_dev)
                
        return X_noisy
    
    def plot_X(self, X, group_ind, save=None):
        """
        Function that plots gene expression series against time. 
        
        Parameters:
            X - Dataset
            group_ind - Array of tuples where each tuple contains X row index and oscillation group.
        """
        for i, g in group_ind:
            plt.scatter(self.t, X[i], label='Gene {}, Group {}'.format(i, g))

        plt.ylabel("Gene Expression")
        plt.xlabel("Time")
        plt.title("Gene Expressions as a function of Time")
        plt.legend(loc="lower right")
        if save is not None:
            plt.savefig(save)
        
    def plot_ellipse(self, X, group_ind_X, group_ind_Y, save=None):
        """
        Function that plots pairs of gene expression series against one another. 
        
        Parameters:
            X - Dataset
            group_ind_X - Array of tuples where each tuple contains X row index and oscillation group.
            group_ind_Y - Array of tuples where each tuple contains X row index and oscillation group.
        """
        for (i1,g1),(i2,g2) in zip(group_ind_X, group_ind_Y):
            plt.figure()
            plt.scatter(X[i1], X[i2])
            plt.xlabel('Gene {}, Group {}'.format(i1, g1))
            plt.ylabel('Gene {}, Group {}'.format(i2, g2))
            plt.title("Gene {} against Gene {}".format(i2, i1))
            plt.savefig(save + '_{}_{}'.format(i1,i2))
