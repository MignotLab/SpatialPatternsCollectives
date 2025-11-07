from tqdm import tqdm
import numpy as np
from itertools import product
from joblib import Parallel, delayed

from parameters import Parameters
from igoshin_matrices import IgoshinMatrix


class MatrixTypeError(Exception):
    pass


class Eigeinvalues:

    def __init__(self, xi, rho_bar, q, K, matrix_igoshin='main'):

        self.par = Parameters()
        self.ig = IgoshinMatrix()

        if matrix_igoshin == 'main':
            self.matrix_igoshin = self.ig.main_matrix
        elif matrix_igoshin == 'rp':
            self.matrix_igoshin = self.ig.rp_matrix
        else:
            raise MatrixTypeError()
        
        if isinstance(K, np.ndarray):
            variables = [xi, rho_bar, q, K]
        elif (K == 'function') and (matrix_igoshin == 'main'):
            variables = [xi, rho_bar, q]
        else:
            variables = [xi, rho_bar, q]

        # Appliquer la fonction lambda à chaque variable
        processed_variables = list(map(self.process_variable, variables))
        # Générer toutes les combinaisons possibles des valeurs des variables
        combinaisons = list(product(*processed_variables))
        # Construire le tableau combiné
        self.combined_array = np.array(combinaisons)

        if K == 'function' and (matrix_igoshin == 'main'):
            self.combined_array = np.concatenate((self.combined_array, self.ig.K_1(self.combined_array[:, 1], self.combined_array[:, 2])[:, np.newaxis]), axis=1)
        elif K == 'function' and (matrix_igoshin == 'rp'):
            self.combined_array = np.concatenate((self.combined_array, self.ig.K_2(self.combined_array[:, 1], self.combined_array[:, 2])[:, np.newaxis]), axis=1)


    def process_variable(self, var):
        """
        Allow to build the good object to build the array with all variable combinaison in __init__
        
        """
        if isinstance(var, np.ndarray):
            return np.atleast_1d(var)
        else:
            return [var]


    def compute_eigeinvalues_main(self, values):
        """
        Parallelize compute_eigeinvalue
        
        """
        g, __, __, __ = self.matrix_igoshin(values[0], values[1], values[2], values[3])
        e, __ = np.linalg.eig(g)

        return np.max(e.real)
    

    def compute_eigeinvalues(self):
        """
        Parallelize compute_eigeinvalue
        
        """
        array = Parallel(n_jobs=self.par.n_jobs)(delayed(self.compute_eigeinvalues_main)(values) for values in tqdm(self.combined_array))
        map_eigenvalues = np.max(np.array(array).reshape(self.par.rho_bar_grid.shape), axis=0) # Along the xi dimension

        return map_eigenvalues