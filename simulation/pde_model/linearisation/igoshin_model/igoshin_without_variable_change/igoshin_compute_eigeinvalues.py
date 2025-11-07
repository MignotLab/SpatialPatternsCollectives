from tqdm import tqdm
import numpy as np
from itertools import product
from joblib import Parallel, delayed

from parameters import Parameters
from igoshin_matrices import IgoshinMatrix


class MatrixTypeError(Exception):
    pass


class Eigeinvalues:

    def __init__(self, values, grid):
        
        self.par = Parameters()
        self.mat = IgoshinMatrix()
        self.values = values
        self.grid = grid


    def compute_eigeinvalues_xi_rho(self, values):
        """
        Parallelize compute_eigeinvalue
        
        """
        g, __, __, __ = self.mat.main_matrix(xi=values[0],
                                             w_1=self.mat.w_1(rho_bar=values[1], q=self.par.q_value_constant),
                                             K_1=self.mat.K_1(rho_bar=values[1], q=self.par.q_value_constant)
                                             )
        e, __ = np.linalg.eig(g)

        return np.max(e.real)


    def compute_eigeinvalues_rp_xi_rho(self, values):
        """
        Parallelize compute_eigeinvalue
        
        """
        phi_r = self.mat.phi_r(rho_bar=values[1])
        K_2 = self.mat.K_2(phi_r=phi_r)
        g, __, __, __ = self.mat.rp_matrix(xi=values[0],
                                           phi_r=phi_r,
                                           K_2=K_2
                                          )
        e, __ = np.linalg.eig(g)

        return np.max(e.real)


    def compute_eigeinvalues_xi_w_1_K_1(self, values):
        """
        Parallelize compute_eigeinvalue
        
        """
        g, __, __, __ = self.mat.main_matrix(xi=values[0],
                                             w_1=values[1],
                                             K_1=values[2]
                                            )
        e, __ = np.linalg.eig(g)

        return np.max(e.real)


    def compute_eigeinvalues_rp_xi_phi_r_K_2(self, values):
        """
        Parallelize compute_eigeinvalue
        
        """
        g, __, __, __ = self.mat.rp_matrix(xi=values[0],
                                           phi_r=values[1],
                                           K_2=values[2]
                                           )
        e, __ = np.linalg.eig(g)

        return np.max(e.real)
    

    def compute_eigeinvalues(self, function):
        """
        Parallelize compute_eigeinvalue
        
        """
        array = Parallel(n_jobs=self.par.n_jobs)(delayed(function)(values) for values in tqdm(self.values))
        print(np.max(array))
        map_eigenvalues = np.max(np.array(array).reshape(self.grid.shape), axis=0) # Along the xi dimension

        return map_eigenvalues