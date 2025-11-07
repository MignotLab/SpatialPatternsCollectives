from tqdm import tqdm
import numpy as np
from itertools import product
from joblib import Parallel, delayed
import scipy as sp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs

from parameters import Parameters
from igoshin_matrices import IgoshinMatrix


class MatrixTypeError(Exception):
    pass


class Eigeinvalues:

    def __init__(self, values, grid, inst_par):
        
        self.par = inst_par
        self.mat = IgoshinMatrix(inst_par)
        self.values = values
        self.grid = grid


    def compute_eigeinvalues_xi_S(self, values, dp):
        """
        Parallelize compute_eigeinvalue
        
        """
        g, __, __, __ = self.mat.main_matrix(xi=values[0],
                                             S=values[1],
                                             K_1=self.mat.K_1(S=values[1], q=self.par.q_value_constant),
                                             dp=dp)
        e, v = np.linalg.eig(g)
        index_e_max = np.argmax(e.real)
        norm_v_max = np.linalg.norm(v[index_e_max])

        return np.max(e.real), norm_v_max


    def compute_eigeinvalues_xi_S_linear(self, values, dp):
        """
        Parallelize compute_eigeinvalue
        
        """
        g, __, __, __ = self.mat.main_matrix(xi=values[0],
                                             S=values[1],
                                             K_1=self.mat.K_1_linear(S=values[1]),
                                             dp=dp)
        e, v = np.linalg.eig(g)
        index_e_max = np.argmax(e.real)
        norm_v_max = np.linalg.norm(v[index_e_max])

        return np.max(e.real), norm_v_max
    

    def compute_eigeinvalues_xi_rho(self, values, dp):
        """
        Parallelize compute_eigeinvalue
        
        """
        S = self.mat.w_1(values[1], q=self.par.q_value_constant) / self.par.w_0
        g, __, __, __ = self.mat.main_matrix(xi=values[0],
                                             S=S,
                                             K_1=self.mat.K_1(S=S, q=self.par.q_value_constant),
                                             dp=dp)
        e, v = np.linalg.eig(g)
        index_e_max = np.argmax(e.real)
        norm_v_max = np.linalg.norm(v[index_e_max])

        return np.max(e.real), norm_v_max


    def compute_eigeinvalues_xi_S_K_1(self, values, dp):
        """
        Parallelize compute_eigeinvalue
        
        """
        g, __, __, __ = self.mat.main_matrix(xi=values[0],
                                             S=values[1],
                                             K_1=values[2],
                                             dp=dp)
        e, v = np.linalg.eig(g)
        index_e_max = np.argmax(e.real)
        norm_v_max = np.linalg.norm(v[index_e_max])

        return np.max(e.real), norm_v_max


    # def compute_eigeinvalues_rp_xi_S_K_1(self, values):
    #     """
    #     Parallelize compute_eigeinvalue
        
    #     """
    #     l, __, __, __, __ = self.mat.rp_matrix_L(xi=values[0],
    #                                              S=values[1],
    #                                              K_1=values[2]
    #                                             )
    #     b, __, __ = self.mat.rp_matrix_B(K_1=values[2])

    #     # Convert it to sparse matrix
    #     l = csr_matrix(l)
    #     b = csr_matrix(b)

    #     e = eigs(l, k=10, M=b, return_eigenvectors=False)

    #     return np.max(e.real)
    

    def compute_eigeinvalues_rp_xi_S(self, values, dp):
        """
        Parallelize compute_eigeinvalue
        
        """
        l, __, __, __, __ = self.mat.rp_matrix_L(xi=values[0],
                                                 S=values[1],
                                                 K_1=self.mat.K_1_rp(S=values[1]),
                                                 dp=dp)
        b, __, __ = self.mat.rp_matrix_B(K_1=self.mat.K_1_rp(S=values[1]), dp=dp)

        e, v = sp.linalg.eig(l, b, left=False, right=True)
        index_e_max = np.argmax(e.real)
        norm_v_max = np.linalg.norm(v[index_e_max])

        return np.max(e.real), norm_v_max


    def compute_eigeinvalues_rp_xi_S_K_1(self, values, dp):
        """
        Parallelize compute_eigeinvalue
        
        """
        l, __, __, __, __ = self.mat.rp_matrix_L(xi=values[0],
                                                 S=values[1],
                                                 K_1=values[2],
                                                 dp=dp)
        b, __, __ = self.mat.rp_matrix_B(K_1=values[2], dp=dp)

        e, v = sp.linalg.eig(l, b, left=False, right=True)
        index_e_max = np.argmax(e.real)
        norm_v_max = np.linalg.norm(v[index_e_max])

        return np.max(e.real), norm_v_max
    

    def compute_eigeinvalues(self, function, dp):
        """
        Parallelize compute_eigeinvalue
        
        """
        array = Parallel(n_jobs=self.par.n_jobs)(delayed(function)(values, dp) for values in tqdm(self.values))

        array_e_reshaped = np.array(array)[:, 0].reshape(self.grid.shape)
        array_v_reshaped = np.array(array)[:, 1].reshape(self.grid.shape)

        # Extract the maximum eigenvalues among the first axis (here the xi dimension)
        map_eigenvalues = np.max(array_e_reshaped, axis=0)

        # Build a condition to extract the element in array_v as the maximum in array_e
        cond_array_e_max = array_e_reshaped == map_eigenvalues
        # Normalize in case of several element are equal along the selected axis
        normalisation_array_e_max = np.sum(cond_array_e_max, axis=0)
        map_eigenvectors = np.sum(array_v_reshaped * cond_array_e_max, axis=0) / normalisation_array_e_max

        return map_eigenvalues, map_eigenvectors