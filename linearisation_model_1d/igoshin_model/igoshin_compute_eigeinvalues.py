from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed
import scipy as sp

from igoshin_matrices import IgoshinMatrix


class MatrixTypeError(Exception):
    pass


class Eigeinvalues:

    def __init__(self, values, grid, inst_par):
        
        self.par = inst_par
        self.mat = IgoshinMatrix(inst_par)
        self.values = values
        self.grid = grid


    def compute_eigeinvalues_xi_S_Phi_R(self, values):
        """
        Parallelize compute_eigeinvalue
        
        """
        g, __, __, __ = self.mat.main_matrix(xi=values[0],
                                             S=values[1],
                                             Phi_R=values[2]
                                             )
        e, __ = np.linalg.eig(g)

        return np.max(e.real)


    def compute_eigeinvalues_rp_xi_S_Phi_R(self, values):
        """
        Parallelize compute_eigeinvalue
        
        """
        l, __, __, __, __ = self.mat.rp_matrix_L(xi=values[0],
                                                 S=values[1],
                                                 Phi_R=values[2]
                                                 )
        b, __, __ = self.mat.rp_matrix_B(S=values[1],
                                         Phi_R=values[2]
                                         )

        e, __ = sp.linalg.eig(l, b, left=False, right=True)

        return np.max(e.real)
    

    def compute_eigeinvalues(self, function):
        """
        Parallelize compute_eigeinvalue
        
        """
        array = Parallel(n_jobs=self.par.n_jobs)(delayed(function)(values) for values in tqdm(self.values))

        array_e_reshaped = np.array(array).reshape(self.grid.shape)

        # Extract the maximum eigenvalues among the first axis (here the xi dimension)
        map_eigenvalues = np.max(array_e_reshaped, axis=0)

        return map_eigenvalues