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


    def phi_r(self, rho_bar):
        """
        Modulated refractory period for igoshin
        
        """
        return self.par.delta_phi * self.par.rho_t / (rho_bar+1e-8)
    

    def _K(self, rho_bar):
        """
        Function K
        
        """
        num = self.par.w_0 * self.par.w_n * self.phi_r(rho_bar)
        den = self.par.w_n * self.phi_r(rho_bar) + np.pi * self.par.w_0

        return num / den

    def matrix_igoshin_rp(self, rho_bar, xi):
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
        diag[self.par.n_rp-1:] -= self.par.w_n / self.par.d_phi
        diag[self.par.n_rp-1] -= self.par.w_n

        # l_plus
        np.fill_diagonal(l_plus, diag)
        l_plus[index+1, index] = self.par.w_0 / self.par.d_phi
        l_plus[(index+1)[self.par.n_rp-1:], index[self.par.n_rp-1:]] += self.par.w_n / self.par.d_phi # -1 because Python start index to 0

        # l_minus
        diag[:] += 2j*self.par.v*xi # update for g_minus
        np.fill_diagonal(l_minus, diag)
        l_minus[index+1, index] = self.par.w_0 / self.par.d_phi
        l_minus[(index+1)[self.par.n_rp-1:], index[self.par.n_rp-1:]] += self.par.w_n / self.par.d_phi # -1 because Python start index to 0

        # l_a
        l_a[0, -1] = self.par.w_0 / self.par.d_phi + self.par.w_n / self.par.d_phi
        l_a[self.par.n_rp-1:self.par.n_rp+1, :] = -self._K(rho_bar)

        l = np.vstack((np.hstack((l_plus, l_a)), np.hstack((l_a, l_minus))))

        return l, l_plus, l_minus, l_a


    def compute_eigeinvalues_main(self, values):
        """
        Compute the eigeinvalues of a matrix for different parameters value
        
        """
        g, __, __, __ = self.matrix_igoshin_rp(values[0], values[1])

        e, __ = np.linalg.eig(g)

        return np.max(e.real)
    

    def compute_eigeinvalues(self):
        """
        Parallelize compute_eigeinvalue
        
        """
        array = Parallel(n_jobs=self.par.n_jobs)(delayed(self.compute_eigeinvalues_main)(values) for values in tqdm(self.par.combined_array))
        map_eigenvalues = np.max(np.array(array).reshape(self.par.rho_bar_grid.shape), axis=0)

        return map_eigenvalues
    

    def initialize_directory_or_file(self, path, columns=None):
        """
        Create the parent directories specified in the path (if they don't already exist).
        If the path contains a file name, create the file or reset it if it already exists.
        If the path does not contain a file name, create only the parent directories.

        Parameters
        ----------
        path : str
            The full path of the file or directories to initialize.

        columns : list or None, optional
            A list of column names to write as the header if the path contains a file name.
            Default is None, which means no header will be written.

        Returns
        -------
        None

        Notes
        -----
        This function creates the parent directories for the specified path, and if the path
        includes a file name (with extension), it will create or reset the file.

        Examples
        --------
        >>> initialize_directory_or_file("data/images/image.jpg")
        # Creates the 'data/images' directory if it doesn't exist and creates/reset the 'image.jpg' file.

        >>> initialize_directory_or_file("data/results/")
        # Creates the 'data/results' directory if it doesn't exist.

        >>> initialize_directory_or_file("data/data.csv", columns=["Name", "Age", "City"])
        # Creates/reset the 'data.csv' file with the specified columns as header.
        """
        # Get the parent directory of the path
        folder_path = os.path.dirname(path)

        # Check if the parent directory exists, if not, create it
        if folder_path and not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Check if the path contains a file name
        if os.path.splitext(path)[1]:  # If the path has an extension (it's a file)
            # Open the file in write mode and write the columns as header if provided
            with open(path, 'w', newline='') as file:
                if columns:
                    header = ",".join(columns)
                    file.write(header + '\n')


    def fill_csv(self, path_csv_file, column_names, *data):
        # Transpose data to ensure equal length if there are multiple columns
        if len(data) > 1:
            # Open the CSV file in append mode
            with open(path_csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                # Write the column names as the first row
                writer.writerow(column_names)
                transposed_data = zip(*data)
                # Write the data using writerows
                writer.writerows(transposed_data)
        else:
            np.savetxt(path_csv_file, data[0], delimiter=',', header=column_names, comments="")
            
            


    def plot_eigenvalues(self, data_array, path_save):
        """
        Plot signal in function of the eigenvalues
        
        """
        xmin = data_array[0, 1]
        xmax = data_array[-1, 1]
        eps = 10e-8
        cond_instabilities = data_array[:, 0] > eps

        fig, ax = plt.subplots(figsize=self.figsize)
        # ax = fig.add_subplot(111)
        ax.plot(data_array[:, 1], data_array[:, 0], color='k', linewidth=2, zorder=0)
        ax.scatter(data_array[~cond_instabilities, 1], data_array[~cond_instabilities, 0], c='b', marker='v', s=25, label='values $\leq 0$', zorder=1, alpha=0.5)
        ax.scatter(data_array[cond_instabilities, 1], data_array[cond_instabilities, 0], c='r', marker='^', s=25, label='values $> 0$', zorder=1, alpha=0.5)
        ax.tick_params(labelsize=self.fontsize/1.5)
        ax.set_xlabel(r'$\bar{\rho}}$', fontsize=self.fontsize)
        ax.set_ylabel(r'$\lambda$ (min$^{-1}$)', fontsize=self.fontsize)
        ax.set_xlim(0, xmax)
        ax.legend(fontsize=self.fontsize/1.5, markerscale=4)
        plt.show()
        fig.savefig(path_save+'map.png', bbox_inches="tight", dpi=100)
        fig.savefig(path_save+'map.svg', bbox_inches="tight", dpi=100)