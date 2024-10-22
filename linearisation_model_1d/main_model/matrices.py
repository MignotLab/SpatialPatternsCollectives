import numpy as np
import scipy.sparse as sp
from parameters import Parameters


class Matrix:

    def __init__(self, a=1):
        
        self.par = Parameters()
        self.a = a
        

    def C_P_directional(self, S):
        return S / (1 + S)

    def C_R_directional(self, S):
        return self.a / (1 + S)
    
    def C_P_local(self, S):
        return 0.5 * S / (1 + S)


    def C_R_local(self, S):
        return 0.5 * self.a / (1 + S)


    def matrix_r_directional(self, xi, S, C):
        """
        Construct the matrix M_R which correspond to the 1D model when the reversal dependence 
        is on the rate of reversal, so the parameter S = /bar{F}R*
        """
        S_ind = int(S / self.par.ds)
        R_pos = np.zeros((self.par.ns, self.par.ns), dtype='cfloat')
        R_A = np.zeros((self.par.ns, self.par.ns), dtype='cfloat')
        R_neg = np.zeros((self.par.ns, self.par.ns), dtype='cfloat')
        # A and D diagonal
        diag = np.zeros(self.par.ns, dtype='cfloat')
        diag[:] = -1j*xi
        diag[:-1] -= self.par.delta
        diag[S_ind:] -= 1
        np.fill_diagonal(R_pos, diag)
        diag[:] += 2j*xi
        np.fill_diagonal(R_neg, diag)
        # A and D  super-diagonal
        index = np.arange(self.par.ns-1)
        R_pos[index+1, index] = self.par.delta
        R_neg[index+1, index] = self.par.delta
        # First row
        R_pos[0, :] += C
        R_neg[0, :] += C
        R_A[0, S_ind:] = 1
        # B and C bottom of the matrix
        renorm = np.sum(np.exp(-np.maximum(self.par.s - S_ind, 0) * self.par.ds) * (self.par.s >= S_ind) * self.par.ds)
        R_A[S_ind:, :] = -C * np.exp(-(np.tile(self.par.s[S_ind:], (self.par.ns, 1)).T - S_ind) * self.par.ds) * self.par.ds / renorm
        
        R = np.block([[R_pos, R_A], [R_A, R_neg]])

        return R, R_pos, R_neg, R_A
    

    # Parameters of the matrices
    def matrix_r_local(self, xi, S, C):
        """
        Construct the matrix M_R which correspond to the 1D model when the reversal dependence is on the rate of reversal, so the parameter S = /bar{F}R*
        """
        S_ind = int(S / self.par.ds)
        __, R_pos, R_neg, R_A = self.matrix_r_directional(xi, S, C)
        # Last rows until S_ind
        renorm = np.sum(np.exp(-np.maximum(self.par.s - S_ind, 0) * self.par.ds) * (self.par.s >= S_ind) * self.par.ds)
        R_pos[S_ind:, :] -= C * np.exp(-(np.tile(self.par.s[S_ind:], (self.par.ns, 1)).T - S_ind) * self.par.ds) * self.par.ds / renorm
        R_neg[S_ind:, :] -= C * np.exp(-(np.tile(self.par.s[S_ind:], (self.par.ns, 1)).T - S_ind) * self.par.ds) * self.par.ds / renorm
        R_A[0, :] += C
        
        R = np.block([[R_pos, R_A], [R_A, R_neg]])

        return R, R_pos, R_neg, R_A
            
            
    def matrix_p_directional(self, xi, S, C):
        """
        Construct the matrix M_P which correspond to the 1D model when the reversal dependence is on the refractory period, so the parameter S = F*/bar{R}
        """
        S_ind = int(S / self.par.ds)
        P_pos = np.zeros((self.par.ns, self.par.ns), dtype='cfloat')
        P_A = np.zeros((self.par.ns, self.par.ns), dtype='cfloat')
        P_neg = np.zeros((self.par.ns, self.par.ns), dtype='cfloat')
        # A and D diagonal
        diag = np.zeros(self.par.ns, dtype='cfloat')
        diag[:] = -1j*xi
        diag[:-1] -= self.par.delta
        diag[S_ind:] -= 1
        np.fill_diagonal(P_pos, diag)
        diag[:] = 1j*xi
        diag[:-1] -= self.par.delta
        diag[S_ind:] -= 1 
        np.fill_diagonal(P_neg, diag)
        # A and D super-diagonal
        index = np.arange(self.par.ns - 1)
        P_pos[index+1, index] = self.par.delta
        P_neg[index+1, index] = self.par.delta
        # First row
        P_pos[0, :] += C
        P_neg[0, :] += C
        P_A[0, S_ind:] = 1
        # B and C middle of the matrix
        P_A[S_ind, :] = -C
        P = np.vstack((np.hstack((P_pos, P_A)), np.hstack((P_A, P_neg))))
        
        return P, P_pos, P_neg, P_A
    

    def matrix_p_local(self, xi, S, C):
        """
        Construct the matrix M_P which correspond to the 1D model when the reversal dependence is on the refractory period, so the parameter S = F*/bar{R}
        """
        S_ind = int(S / self.par.ds)
        __, P_pos, P_neg, P_A = self.matrix_p_directional(xi, S, C)
        P_pos[S_ind, :] -= C
        P_neg[S_ind, :] -= C
        P_A[0, :] += C

        P = np.vstack((np.hstack((P_pos, P_A)), np.hstack((P_A, P_neg))))
        
        return P, P_pos, P_neg, P_A