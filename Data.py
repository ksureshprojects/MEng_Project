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
        self.psi = [random.random()*2*np.pi for i in range(self.ng)]
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
    
    def gen_data(self, t_dist_normal=True, sample_mean=np.pi * 2, st_dev=3):
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
            self.t = np.linspace(0, sample_mean / (self.groups[-1] + 0.0000001), self.nc)
            np.random.shuffle(self.t)
            
        X = []
        group_ind = {}
        for i in range(self.ng):
            gene = []
            # 90% of genes are non-oscillatory
            if np.random.random() < 0.9:
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
    
    def plot_X(self, X, group_ind, groups, n_non):
        """
        Function that plots gene expression series against time. 
        
        Parameters:
            X - Dataset
            group_ind - Dictionary with key being oscillatory group number (or -1 for non-osciallatory group) and values
                        being all row indicies in dataset belonging to group corresponding to key.
            groups - Array of tuples where each tuple has two values. First value represents oscillatory group, 
                     second value represents number of genes to plot from this group.
        """
        if n_non > len(group_ind[0]):
            raise ValueError('Number of non-oscilating genes to plot greater than actual number of non-oscilating genes.')
        
        plt.figure()
        for tup in groups:
            if tup[1] > len(group_ind[tup[0]]):
                raise ValueError('Number of oscilating genes in specified group is less than desired number of plots.')
            for i in range(tup[1]):
                plt.scatter(self.t, X[group_ind[tup[0]][i]], label='Gene {}, Group {}'.format(group_ind[tup[0]][i] + 1, tup[0]))
            
        for i in range(n_non):
            plt.scatter(self.t, X[group_ind[0][i]], label='Gene {}, Group {}'.format(group_ind[0][i] + 1, 0))

        plt.ylabel("Normalised Gene Expression")
        plt.xlabel("Time")
        plt.title("Gene Expressions as a function of Time")
        plt.legend(loc="lower right")
        
    def plot_ellipse(self, X, group_ind, groups):
        """
        Function that plots pairs of gene expression series against one another. 
        
        Parameters:
            X - Dataset
            group_ind - Dictionary with key being oscillatory group number (or -1 for non-osciallatory group) and values
                        being all row indicies in dataset belonging to group corresponding to key.
            groups - Array of tuples where each tuple has two values. First value represents oscillatory group, 
                     second value represents number of pairs genes to plot from this group.
        """
        for tup in groups:
            count = 0
            if tup[1] > ncr(len(group_ind[tup[0]]), 2):
                raise ValueError('Not enough number of oscilating genes for desired number of pairs.')
            for i in range(len(group_ind[tup[0]])):
                for j in range(i + 1, len(group_ind[tup[0]])):
                    plt.figure()
                    plt.scatter(X[group_ind[tup[0]][i]], X[group_ind[tup[0]][j]])
                    plt.xlabel('Gene {}, Group {}'.format(group_ind[tup[0]][i] + 1, tup[0]))
                    plt.ylabel('Gene {}, Group {}'.format(group_ind[tup[0]][j] + 1, tup[0]))
                    plt.title("Gene {} as a function of Gene {}".format(group_ind[tup[0]][j] + 1, group_ind[tup[0]][i] + 1))
                    count += 1
                    if count == tup[1]:
                        break
                if count == tup[1]:
                        break