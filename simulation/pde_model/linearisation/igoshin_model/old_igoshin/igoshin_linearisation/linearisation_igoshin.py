import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize 
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
from joblib import Parallel, delayed
import parameters_igoshin


class Linearisation:
    """
    On adore calculer les valeurs propres
    
    """

    def __init__(self, fontsize, figsize):
        
        self.par = parameters_igoshin.Parameters()
        self.fontsize = fontsize
        self.figsize = figsize


    def w_1(self, rho_bar, q):
        """
        Function \omega_1
        
        """
        return self.par.w_n * rho_bar**q / (rho_bar**q + self.par.rho_w**q)
    

    def K_1(self, rho_bar, q):
        """
        Function K_1
        
        """
        num = q * self.par.w_0**2 * self.w_1(rho_bar, q) * (1 - self.w_1(rho_bar, q) / self.par.w_n)
        den = (self.w_1(rho_bar, q) * self.par.delta_phi + np.pi * self.par.w_0) * (self.par.w_0 + self.w_1(rho_bar, q))

        return num / den


    def matrix_igoshin(self, rho_bar, q, xi):
        """
        Construct the matrix from the igoshin linearised model

        """
        g_plus = np.zeros((self.par.n, self.par.n), dtype='cfloat')
        g_a = np.zeros((self.par.n, self.par.n), dtype='cfloat')
        g_minus = np.zeros((self.par.n, self.par.n), dtype='cfloat')
        index = np.arange(self.par.n - 1)

        # Diag
        diag = np.zeros(self.par.n, dtype='cfloat')
        diag[:] = -1j*xi - self.par.w_0 / self.par.d_phi
        diag[self.par.n_rp-1:] -= self.w_1(rho_bar, q) / self.par.d_phi
        diag[self.par.n_rp-1] -= self.w_1(rho_bar, q) / self.par.d_phi

        # g_plus
        np.fill_diagonal(g_plus, diag)
        g_plus[index+1, index] = self.par.w_0 / self.par.d_phi
        g_plus[(index+1)[self.par.n_rp-1:], index[self.par.n_rp-1:]] += self.w_1(rho_bar, q) / self.par.d_phi # -1 because Python start index to 0

        # g_minus
        diag[:] += 2j*xi # update for g_minus
        np.fill_diagonal(g_minus, diag)
        g_minus[index+1, index] = self.par.w_0 / self.par.d_phi
        g_minus[(index+1)[self.par.n_rp-1:], index[self.par.n_rp-1:]] += self.w_1(rho_bar, q) / self.par.d_phi # -1 because Python start index to 0

        # g_a
        g_a[0, -1] = self.par.w_0 / self.par.d_phi + self.w_1(rho_bar, q) / self.par.d_phi
        g_a[self.par.n_rp-1, :] = -self.K_1(rho_bar, q)
        
        g = np.vstack((np.hstack((g_plus, g_a)), np.hstack((g_a, g_minus))))

        return g, g_plus, g_minus, g_a


    def compute_eigeinvalues_main(self, values):
        """
        Compute the eigeinvalues of a matrix for different parameters value
        
        """
        g, __, __, __ = self.matrix_igoshin(values[0], values[1], values[2])

        e, __ = np.linalg.eig(g)

        return np.max(e.real)
    

    def compute_eigeinvalues(self):
        """
        Parallelize compute_eigeinvalue
        
        """
        array = Parallel(n_jobs=self.par.n_jobs)(delayed(self.compute_eigeinvalues_main)(values) for values in tqdm(self.par.combined_array))
        # print(array)
        map_eigenvalues = np.max(np.array(array).reshape(self.par.rho_bar_grid.shape), axis=2)
        # num_err = 10e-8
        # array_P[array_P<num_err] = np.nan
        # array_R[array_R<num_err] = np.nan

        return map_eigenvalues
    

    def initialize_csv(self, path_file):
        """
        Initialize csv file.
        In case the file exist first remove it and write the columns name.
        
        """
        if os.path.isfile(path_file):
            os.remove(path_file)  # Supprimer le fichier existant s'il existe
        with open(path_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)


    def append_to_csv(self, path_file, data):
        """
        Write in a csv file 
        
        """
        with open(path_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data)

    def map_eigenvalues(self, eigen_map, rho_array, q_array, path_save):
        """
        Plot the map of the eigenvalues
        
        """
        cmap = plt.get_cmap('plasma_r')
        xmin = rho_array[0]
        xmax = rho_array[-1]
        ymin = q_array[0]
        ymax = q_array[-1]
        lambda_max = np.nanmax(eigen_map)
        eigen_map_plot = eigen_map.copy()
        eigen_map_plot[eigen_map_plot < 10e-8] = np.nan

        fig, ax = plt.subplots(figsize=self.figsize)
        # ax = fig.add_subplot(111)
        im = plt.imshow(eigen_map_plot, extent=[xmin,xmax,ymin,ymax], origin='lower', cmap=cmap, aspect='auto', vmin=0, vmax=lambda_max)
        ax.tick_params(labelsize=self.fontsize/1.5)
        ax.set_xlabel(r'$\bar{\rho}}$', fontsize=self.fontsize)
        ax.set_ylabel(r'$q$', fontsize=self.fontsize)
        ax.set_xlim(0.9, xmax)
        ax.set_ylim(1.9, ymax)
        # plt.xticks(np.arange(0, xmax, step=2), fontsize=self.fontsize/1.5)
        # plt.yticks(np.arange(0, ymax, step=0.2), fontsize=self.fontsize)
        # self.forceAspect(ax=ax, aspect=1)
        v1 = np.linspace(0, lambda_max, 4, endpoint=True)
        cb = plt.colorbar(ticks=v1, shrink=1)
        # cbar.ax.tick_params(labelsize=12)
        cb.ax.set_yticklabels(["{:.1e}".format(i) for i in v1], fontsize=self.fontsize/1.5)
        # plt.gca().invert_yaxis()
        cb.set_label(r"$\lambda$", fontsize=self.fontsize)
        # plt.legend(fontsize=self.fontsize*0.7)
        plt.show()
        fig.savefig(path_save+'map.png', bbox_inches="tight", dpi=100)
        fig.savefig(path_save+'map.svg', bbox_inches="tight", dpi=100)