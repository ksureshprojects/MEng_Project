import sys
from matplotlib import pyplot as plt
from SPCA import *
import numpy as np
import pandas as pd

class Analyse:
    def __init__(self):
        self.reset()

    def reset(self):
        # Samples per period
        self.spp = None
        # Original data
        self.X_ = None
        # Processed Data
        self.X = None
        # Processed data indicies
        self.proc_ind = None
        # True time labels
        self.time = None
        # Gene labels
        self.labels = None
        # Reordered time array indices
        self.reordered = None
        # Reordering Rank Error
        self.rre = None
        # Ellipse MSE
        self.mse = None
        # Ellipse Width
        self.width = None
        # Plots
        self.fig, self.axs = (None, None)

        self.kwargs = {
        # Analysis Parameters
            'save': None,
            'plots': None,
        # Gene Filtering Methods
            'med': None,
            'var': None,
        # Gene pre-processing Methods
            'std': None,
            'scl': None,
            'cen': None
        }

    def setProperties(self, spp, X_, time, labels, kw, plot=True):
        """
        Reset and set class properties.

        Parameters:
        spp - Samples per period
        X_ - Original data
        time - True time labels
        kw - Keyword arguments indicating analysis parameters, gene 
             filtering methods, and gene pre-processing methods.
        """
        # Reset Properties
        self.reset()
        if plot:
            self.fig, self.axs = plt.subplots(1, 3, figsize=[19.2,4])
        self.spp = spp
        self.X_ = X_.copy()
        self.time = time.copy()
        if labels is not None:
            self.labels = labels.copy()
        else:
            self.labels = np.array([i for i in range(self.X_.shape[0])])
        for key in self.kwargs:
            self.kwargs[key] = kw.get(key)

    def preProcess(self):
        """
        Pre-process data prior to analysis.

        Updates:
        self.X
        """
        if self.kwargs['cen']:
            # Center data to have zero mean
            self.X = self.X - np.mean(self.X, axis=1, keepdims=True)
        
        if self.kwargs['std']:
            # Standardise every gene series to have variance of 1
            self.X = self.X / np.std(self.X, axis=1, keepdims=True)
        elif self.kwargs['scl']:
            # Scale range so that all values lie between [-1,1]
            self.X = self.X / np.maximum(np.max(self.X, 
                                                axis=1, keepdims=True), 
                                  np.abs(np.min(self.X, 
                                                axis=1, keepdims=True)))
            # self.X = self.X * 2
            # # Recenter so that all gene values lie between [-1,1]
            # if self.kwargs['cen']:
            #     self.X = self.X - np.mean(self.X, axis=1, keepdims=True)


    def filter(self):
        """
        Select a subset a of genes upon which analysis 
        can be conducted.

        Updates:
        self.X, self.proc_ind
        """
        if self.kwargs['var'] and self.kwargs['med']:
            l = self.X_.shape[0]
            h = l//2 - 1
            # Sort gene series by median, keep top half
            arg_x = np.argsort(np.median(self.X_, axis=1))[h:]
            # Sort indicies by variance in descending order
            arg_med = np.argsort(np.var(self.X_[arg_x, :], axis=1))
            arg_x = arg_x[arg_med[::-1]]
        elif self.kwargs['var']:
            # Sort indicies by variance in descending order
            arg_x = np.argsort(np.var(self.X_, axis=1))[::-1]
        elif self.kwargs['med']:
            l = self.X_.shape[0]
            h = l//2 - 1
            # Sort gene series by median, keep top half
            arg_x = np.argsort(np.median(self.X_, axis=1))[h:]
        

        if self.kwargs['var'] or self.kwargs['med']:
            # Store processed data and index map
            self.X = self.X_[arg_x, :]
            self.proc_ind = arg_x
        else:
            self.X = self.X_.copy()
            self.proc_ind = np.array([i for i in range(self.X.shape[0])])

    def fitEllipse(self, i1, i2, mse_only=False, comp=False, plot=True):
        """
        Method to create elliptical plot, find best elliptical fit, 
        calculate MSE value.

        Parameters:
        i1, i2 - Gene indicies for filtered data
        Updates:
        self.reordered, self.axs[0]
        Returns:
        mse, ellipse width if needed during Oscope
        """
        # Extract gene series
        X = self.X[i1]
        Y = self.X[i2]
        # Formulate and solve the least squares problem ||Ax - b ||^2
        X = X.reshape((X.shape[0], 1))
        Y = Y.reshape((Y.shape[0], 1))
        A = np.hstack([X**2, X * Y, Y**2, X, Y])
        b = np.ones_like(X)
        x = np.linalg.lstsq(A, b, rcond=None)[0].squeeze()
        # min(A,C)/max(A,C) - measure of ellipse width
        r = min(x[0], x[2])/max(x[0], x[2])
        # Mean squared error
        mse = (np.linalg.norm(np.dot(A, x) - b, ord=2) ** 2)/(b.shape[0])
        mse = round(mse, 2)
        # Returning only MSE and ellipse width
        if mse_only:
            return mse, r
        else:
            # Store results
            self.mse = mse
            self.width = r
        # Plot the least squares ellipse
        x_coord = np.linspace(np.min(X) - 0.5, np.max(X) + 0.5, 300)
        y_coord = np.linspace(np.min(Y) - 0.5, np.max(Y) + 0.5, 300)
        X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
        Z_coord = x[0] * X_coord ** 2 + x[1] * X_coord * Y_coord + \
                  x[2] * Y_coord**2 + x[3] * X_coord + x[4] * Y_coord
        
        if plot:
            # Plot ellipse
            self.axs[0].scatter(X, Y)
            self.axs[0].contour(X_coord, Y_coord, Z_coord, levels=[1], 
                                colors=('r'), linewidths=2)
            self.axs[0].set_title('MSE Ellipse Fit: {}'.format(mse))
            # Check calling fn
            if comp:
                x_label = self.labels[self.proc_ind[i1]]
                y_label = self.labels[self.proc_ind[i2]]
            else:
                x_label = "Gene {}".format(self.labels[self.proc_ind[i1]])
                y_label = "Gene {}".format(self.labels[self.proc_ind[i2]])
            self.axs[0].set_xlabel(x_label)
            self.axs[0].set_ylabel(y_label)
        
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

            true_time = self.time[i]
            line = pd.DataFrame(data=[[true_time, -a, i]],
                                columns=['Time', 'Angle', 'Index'])
            df = df.append(line, ignore_index=True)
        # Sort table by time
        df = df.sort_values(by='Time')
        # Calculate reordering quality
        asc = self.rankError(df['Time'].values, 
                                       df['Angle'].values)
        # Return index column after sorting by angle, 
        # quality metric and window size
        self.reordered = df.sort_values(by='Angle', ascending=asc)['Index']
        self.reordered = self.reordered.values.astype('int')
    
    def rankError(self, x, y):
        """
        Function that calculates modified spearman's 
        rank to evaluate ellipse.

        Parameters:
        x, y - Vectors corresponding to angular positions 
               and real collection times of samples.
        Updates:
        self.rre
        Returns:
        Direction of reordering
        """
        # Storing vector length
        l = x.shape[0]
        # Retrieving samples per period
        spp = self.spp
        # Arrays to store errors from each period segment
        errors_p = np.array([])
        errors_n = np.array([])
        # Find rank error
        for j in range(l//spp):
            # Scale time array ranks to be between 0 and 1
            x_ = np.argsort(x[j*spp : (j+1)*spp]) * 1/spp
            # Scale angular position ranks and negative angular 
            # position to be between 0 and 1
            y_p = np.argsort(y[j*spp : (j+1)*spp]) * 1/spp
            y_n = np.argsort(-y[j*spp : (j+1)*spp]) * 1/spp
            # Record best error for every rank
            min_e_p = np.inf
            min_e_n = np.inf
            # Anchor angular position array at every value
            for i in range(spp):
                # Find anchored angular position arrays
                y_a_p = np.block([y_p[i:], y_p[:i]])
                y_a_n = np.block([y_n[i:], y_n[:i]])
                # Calculate circular mean absolute error
                e_p = np.mean(np.abs(y_a_p - x_ + 
                                     ((y_a_p - x_) < -1/2) - 
                                     ((y_a_p - x_) > 1/2)))
                e_n = np.mean(np.abs(y_a_n - x_ + 
                                     ((y_a_n - x_) < -1/2) - 
                                     ((y_a_n - x_) > 1/2)))
                # Update min value seen
                if e_p < min_e_p:
                    min_e_p = e_p
                if e_n < min_e_n:
                    min_e_n = e_n
            # Store error values from segment
            errors_p = np.block([errors_p, min_e_p])
            errors_n = np.block([errors_n, min_e_n])
        # Calculate mean result
        mean_p = np.mean(errors_p)
        mean_n = np.mean(errors_n)

        # Return best result
        if mean_p < mean_n:
            self.rre = round(mean_p, 3)
            return True
        else:
            self.rre = round(mean_n, 3)
            return False

    def plotGenes(self, i1, i2):
        """
        Create plots of true order of gene series and reordered gene 
        series.

        Updates:
        self.axs[1], self.axs[0]
        """
        # Extract gene series
        X1 = self.X[i1]
        X2 = self.X[i2]
        # Extract number of samples
        t = X1.shape[0]
        # Gene labels
        x1_label = self.labels[self.proc_ind[i1]]
        x2_label = self.labels[self.proc_ind[i2]]
        # True Order
        self.axs[1].scatter([i for i in range(t)], X1[np.argsort(self.time)],
                            label="Gene {}".format(x1_label))
        self.axs[1].scatter([i for i in range(t)], X2[np.argsort(self.time)],
                            label="Gene {}".format(x2_label))
        self.axs[1].set_xlabel("Index")
        self.axs[1].set_ylabel("Gene Value")
        self.axs[1].set_title("True Sample Order")
        self.axs[1].legend()
        # Reordered Samples
        x1_label = self.labels[self.proc_ind[i1]]
        x2_label = self.labels[self.proc_ind[i2]]
        self.axs[2].scatter([i for i in range(t)], X1[self.reordered],
                            label="Gene {}".format(x1_label))
        self.axs[2].scatter([i for i in range(t)], X2[self.reordered],
                            label="Gene {}".format(x2_label))
        self.axs[2].set_xlabel("Index")
        self.axs[2].set_ylabel("Gene Value")
        self.axs[2].set_title("Reordered Samples")
        self.axs[2].legend()

    def OSCOPE(self, X_, time, labels=None, break_point=100, spp=24, 
               PCA=False, SPCA=False, t=20, kw=None, plot=True):
        """
        Carry out OSCOPE method to identify pair of genes that form best
        elliptical
        plot.

        Parameters:
        X_ - Data set of gene expression series
        break_point - Number of genes to consider
        time - Corresponding array of real collection times.
        save - Filename base
        spp - Samples per period
        kw - Keyword arguments indicating analysis parameters, gene 
             filtering methods, and gene pre-processing methods.
        """
        # Set class properties
        self.setProperties(spp, X_, time, labels, kw, plot=plot)
        # Filter data
        self.filter()
        # Pre process data
        self.preProcess()

        if PCA:
            if not self.kwargs['cen']:
                raise ValueError("Data needs to be centred for PCA")
            # image: m x n, U: m x m, s: min(n, m) vector, V: n x n
            U, s, Vh = np.linalg.svd(self.X.T)
            # First n_component rows of VT. Rows of VT are principal
            # component directions.
            V1 = Vh[0]
            # Re-filter data by loading vector values
            arg_x = np.argsort(np.abs(V1))[::-1]
            # Store processed data and index map
            self.X = self.X[arg_x, :]
            self.proc_ind = arg_x
        elif SPCA:
            if not self.kwargs['cen']:
                raise ValueError("Data needs to be centred for SPCA")
            # Run Sparse PCA
            V, U2 = sparse_rank_n_uv(self.X.T, t=t, scl=self.kwargs['scl'],
                                      std=self.kwargs['std'])
            # First n_component rows of VT. Rows of VT are principal
            # component directions.
            V1 = V.T[0]
            # Re-filter data by loading vector values
            arg_x = np.argsort(np.abs(V1))[::-1]
            # Store processed data and index map
            self.X = self.X[arg_x, :]
            self.proc_ind = arg_x


        # Test every pair of genes for elliptical fit
        col = ['Gene 1', 'Gene 2', 'Ellipse Width', 'MSE']
        df = pd.DataFrame(columns=col)
        # Max number of genes
        l = self.X.shape[0]
        # Range to explore
        r = min(break_point,l)
        for i in range(r):
            # Show progress
            print('\r {}/{}'.format(i+1, min(break_point,l)), end='')
            for j in range(i + 1, r):
                mse_, width = self.fitEllipse(i, j, mse_only=True)
                line = pd.DataFrame(data=[[i, j, width, mse_]], 
                                    columns=col)
                df = df.append(line, ignore_index=True)
        # Sort pairs by elliptical fit quality
        df.sort_values(by=['MSE'], inplace=True, ascending=True)
        df = df.reset_index(drop=True)

        i1 = df['Gene 1'][0]
        i2 = df['Gene 2'][0]

        # Find Best Ellipse and Reorder Data
        self.fitEllipse(i1, i2, plot=plot)
        # Plot genes
        if plot:
            self.plotGenes(i1, i2)
            # Display figure and error
            # self.fig.show()
            print('\r Reordering Rank Error: {}'.format(self.rre))
            # Savefig
            if self.kwargs['save'] is not None:
                self.fig.savefig(self.kwargs['save'])
        
        return self.rre, self.mse, self.width

    def pComponents(self, X_, time, labels=None, break_point=100, spp=24, 
                    t=20, SPCA=False, PCA=False, kw=None, plot=True):
        # Set class properties
        self.setProperties(spp, X_, time, labels, kw, plot=plot)
        # Filter data
        self.filter()
        # Pre process data
        self.preProcess()

        if not self.kwargs['cen']:
                raise ValueError("Data needs to be centred for PCA")

        if PCA:
            # image: m x n, U: m x m, s: min(n, m) vector, V: n x n
            U, s, Vh = np.linalg.svd(self.X.T)
            # First n_component rows of VT. Rows of VT are principal
            # component directions.
            Vh = Vh[:2, :]
            # Reduced Data
            U2 = (self.X.T).dot(Vh.T).T
        elif SPCA:
            # Run Sparse PCA
            V, U2 = sparse_rank_n_uv(self.X.T, t=t, scl=self.kwargs['scl'],
                                      std=self.kwargs['std'])
            Vh = V.T
        else:
            raise ValueError('PCA or SPCA flag not set')

        # Replace processed data
        temp_data = self.X.copy()
        self.X = U2
        temp_labels = self.labels
        self.labels = np.array(["First Component", "Second Component"])
        temp_ind = self.proc_ind
        self.proc_ind = np.array([i for i in range(self.X.shape[0])])
        # Find Best Ellipse and Reorder Data
        self.fitEllipse(0, 1, comp=True, plot=plot)
        # Restore processed data
        self.X = temp_data
        self.labels = temp_labels
        self.proc_ind = temp_ind
        # Pick top genes
        i1 = np.argmax(np.abs(Vh[0]))
        Vh[0, i1] = 0
        i2 = np.argmax(np.abs(Vh[0]))
        # Plot genes
        if plot:
            self.plotGenes(i1, i2)
            # Display figure and error
            # self.fig.show()
            print('\r Reordering Rank Error: {}'.format(self.rre))
            # Savefig
            if self.kwargs['save'] is not None:
                self.fig.savefig(self.kwargs['save'])

        return self.rre, self.mse, self.width