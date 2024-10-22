import numpy as np
import scipy as sp
from tqdm import tqdm
from joblib import Parallel, delayed
from parameters import Parameters
from matrices import Matrix


class Eigeinvalues:

    def __init__(self, values, grid, a=1):
        
        self.par = Parameters()
        self.mat = Matrix()
        self.values = values
        self.grid = grid
        self.a = a
    

    def compute_eigeinvalues_main_xi_S(self, values, loc_or_dir):
        """
        Compute the eigeinvalues of a matrix for different parameters value
        
        """
        if loc_or_dir =='loc':
            C_R = self.mat.C_R_local(S=values[1])
            C_P = self.mat.C_P_local(S=values[1])
        elif loc_or_dir =='dir':
            C_R = self.mat.C_R_directional(S=values[1])
            C_P = self.mat.C_P_directional(S=values[1])
        else:
            return 'loc_or_dir parameter should be "loc" or "dir"'
        R, __, __, __ = self.chosen_f_matrix_r(values[0], values[1], C_R)
        P, __, __, __ = self.chosen_f_matrix_p(values[0], values[1], C_P)

        w1, __ = np.linalg.eig(R)
        w2, __ = np.linalg.eig(P)

        return np.max(w1.real), np.max(w2.real)
    

    def compute_eigeinvalues_main_xi_S_C(self, values, loc_or_dir):
        """
        Compute the eigeinvalues of a matrix for different parameters value
        
        """
        R, __, __, __ = self.chosen_f_matrix_r(values[0], values[1], values[2])
        P, __, __, __ = self.chosen_f_matrix_p(values[0], values[1], values[2])

        w1, __ = np.linalg.eig(R)
        w2, __ = np.linalg.eig(P)

        return np.max(w1.real), np.max(w2.real)
    

    def compute_eigeinvalues(self, function, matrix_p, matrix_r, loc_or_dir):
        """
        Parallelize compute_eigeinvalue
        
        """
        self.chosen_f_matrix_p = matrix_p
        self.chosen_f_matrix_r = matrix_r
        arrays = Parallel(n_jobs=self.par.n_jobs)(delayed(function)(values, loc_or_dir) for values in tqdm(self.values))
        self.arrays = np.array(arrays)
        array_R = np.max(self.arrays[:, 0].reshape(self.grid.shape), axis=0)
        array_P = np.max(self.arrays[:, 1].reshape(self.grid.shape), axis=0)

        return array_R, array_P