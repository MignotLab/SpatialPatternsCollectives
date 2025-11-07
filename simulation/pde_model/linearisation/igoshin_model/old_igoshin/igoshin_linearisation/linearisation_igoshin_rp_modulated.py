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
    

    def phi_r(self, rho_bar, rho_t):
        """
        Modulated refractory period for igoshin
        
        """
        return self.par.delta_phi * rho_t / rho_bar
    

    def K_2(self, rho_bar, rho_t):
        """
        Function K
        
        """
        num = -self.phi_r(rho_bar, rho_t) * self.par.w_n**2 * self.par.w_0
        den = (self.par.w_n * self.phi_r(rho_bar, rho_t) + np.pi * self.par.w_0) * (self.w_0 + self.w_n)

        return num / den

    def matrix_igoshin_rp(self, rho_bar, rho_t, xi):
        """
        Construct the matrix from the revisited igoshin linearised model (modulation of the refractory period)

        """
        l_plus = np.zeros((self.par.n, self.par.n), dtype='cfloat')
        l_a = np.zeros((self.par.n, self.par.n), dtype='cfloat')
        l_minus = np.zeros((self.par.n, self.par.n), dtype='cfloat')
        index = np.arange(self.par.n - 1)

        # Diag
        diag = np.zeros(self.par.n, dtype='cfloat')
        diag[:] = -1j*self.par.v*xi - self.par.w_0 / self.par.d_phi
        diag[self.par.n_rp-1:] -= self.w_n / self.par.d_phi
        diag[self.par.n_rp-1] -= self.w_n / self.par.d_phi

        # l_plus
        np.fill_diagonal(l_plus, diag)
        l_plus[index+1, index] = self.par.w_0 / self.par.d_phi
        l_plus[(index+1)[self.par.n_rp-1:], index[self.par.n_rp-1:]] += self.par.w_n / self.par.d_phi # -1 because Python start index to 0

        # l_minus
        diag[:] += 2j*xi # update for g_minus
        np.fill_diagonal(l_minus, diag)
        l_minus[index+1, index] = self.par.w_0 / self.par.d_phi
        l_minus[(index+1)[self.par.n_rp-1:], index[self.par.n_rp-1:]] += self.par.w_n / self.par.d_phi # -1 because Python start index to 0

        # l_a
        l_a[0, -1] = self.par.w_0 / self.par.d_phi + self.par.w_n / self.par.d_phi
        l_a[self.par.n_rp-1:self.par.n_rp+1, :] = -self._K(rho_bar, rho_t)

        l = np.vstack((np.hstack((l_plus, l_a)), np.hstack((l_a, l_minus))))

        return l, l_plus, l_minus, l_a


    def compute_eigeinvalues_main(self, values):
        """
        Compute the eigeinvalues of a matrix for different parameters value
        
        """
        g, __, __, __ = self.matrix_igoshin_rp(values[0], values[1], values[2])

        e, __ = np.linalg.eig(g)

        return np.max(e.real)
    

    def compute_eigeinvalues(self):
        """
        Parallelize compute_eigeinvalue
        
        """
        array = Parallel(n_jobs=self.par.n_jobs)(delayed(self.compute_eigeinvalues_main)(values) for values in tqdm(self.par.combined_array_rp_modulated))
        map_eigenvalues = np.max(np.array(array).reshape(self.par.rho_bar_grid_rp.shape), axis=2)

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

    def map_eigenvalues(self, eigen_map, rho_array, rho_t_array, path_save):
        """
        Plot the map of the eigenvalues
        
        """
        cmap = plt.get_cmap('plasma_r')
        xmin = rho_array[0]
        xmax = rho_array[-1]
        ymin = rho_t_array[0]
        ymax = rho_t_array[-1]
        lambda_max = np.nanmax(eigen_map)
        eigen_map_plot = eigen_map.copy()
        eigen_map_plot[eigen_map_plot < 10e-8] = np.nan

        fig, ax = plt.subplots(figsize=self.figsize)
        # ax = fig.add_subplot(111)
        im = plt.imshow(eigen_map_plot, extent=[xmin,xmax,ymin,ymax], origin='lower', cmap=cmap, aspect='auto', vmin=0, vmax=lambda_max)
        ax.tick_params(labelsize=self.fontsize/1.5)
        ax.set_xlabel(r'$\bar{\rho}}$', fontsize=self.fontsize)
        ax.set_ylabel(r'$\rho_t$', fontsize=self.fontsize)
        ax.set_xlim(-0.1, xmax)
        ax.set_ylim(-0.1, ymax)
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