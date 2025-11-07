import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize 
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
from joblib import Parallel, delayed
import parameters


class Linearisation:

    def __init__(self, a=1):
        
        self.par = parameters.Parameters()
        self.a = a

    # Parameters of the matrices
    def matrix_r_directional(self, S, C, xi):
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
        diag[:] = 1j*xi
        diag[:-1] -= self.par.delta
        diag[S_ind:] -= 1 
        np.fill_diagonal(R_neg, diag)
        # A and D  super-diagonal
        index = np.arange(self.par.ns-1)
        R_pos[index+1, index] = self.par.delta
        R_neg[index+1, index] = self.par.delta
        # First row
        # R_pos[0, :] += C * np.sum(np.exp(-np.maximum(self.par.s - S, 0)) * (self.par.s >= S) * self.par.ds)
        # R_neg[0, :] += C * np.sum(np.exp(-np.maximum(self.par.s - S, 0)) * (self.par.s >= S) * self.par.ds)
        R_pos[0, :] += C
        R_neg[0, :] += C
        R_A[0, S_ind:] = 1
        # B and C bottom of the matrix
        # renorm = np.sum(np.exp(-np.maximum(self.par.s - S, 0)) * (self.par.s >= S) * self.par.ds)
        renorm = np.sum(np.exp(-np.maximum(self.par.s - S_ind, 0) * self.par.ds) * (self.par.s >= S_ind) * self.par.ds)
        # R_A[S_ind:, :] = -C * np.exp(-(np.tile(self.par.s[S_ind:], (self.par.ns, 1)).T - S)) * self.par.ds
        # R_A[S_ind:, :] = -C * np.exp(-(np.tile(self.par.s[S_ind:], (self.par.ns, 1)).T - S)) * self.par.ds / renorm
        R_A[S_ind:, :] = -C * np.exp(-(np.tile(self.par.s[S_ind:], (self.par.ns, 1)).T - S_ind) * self.par.ds) * self.par.ds / renorm
        
        R = np.vstack((np.hstack((R_pos, R_A)), np.hstack((R_A, R_neg))))

        return R, R_pos, R_neg, R_A
    

    # Parameters of the matrices
    def matrix_r_local(self, S, C, xi):
        """
        Construct the matrix M_R which correspond to the 1D model when the reversal dependence is on the rate of reversal, so the parameter S = /bar{F}R*
        """
        S_ind = int(S / self.par.ds)
        __, R_pos, R_neg, R_A = self.matrix_r_directional(S, C, xi)
        # Last rows until S_ind
        # renorm = np.sum(np.exp(-np.maximum(self.par.s - S, 0)) * (self.par.s >= S) * self.par.ds)
        renorm = np.sum(np.exp(-np.maximum(self.par.s - S_ind, 0) * self.par.ds) * (self.par.s >= S_ind) * self.par.ds)
        # R_pos[S_ind:, :] -= C * np.exp(-(np.tile(self.par.s[S_ind:], (self.par.ns, 1)).T - S)) * self.par.ds
        # R_neg[S_ind:, :] -= C * np.exp(-(np.tile(self.par.s[S_ind:], (self.par.ns, 1)).T - S)) * self.par.ds
        # R_pos[S_ind:, :] -= C * np.exp(-(np.tile(self.par.s[S_ind:], (self.par.ns, 1)).T - S)) * self.par.ds / renorm
        # R_neg[S_ind:, :] -= C * np.exp(-(np.tile(self.par.s[S_ind:], (self.par.ns, 1)).T - S)) * self.par.ds / renorm
        R_pos[S_ind:, :] -= C * np.exp(-(np.tile(self.par.s[S_ind:], (self.par.ns, 1)).T - S_ind) * self.par.ds) * self.par.ds / renorm
        R_neg[S_ind:, :] -= C * np.exp(-(np.tile(self.par.s[S_ind:], (self.par.ns, 1)).T - S_ind) * self.par.ds) * self.par.ds / renorm
        R_A[0, :] += C
        
        R = np.vstack((np.hstack((R_pos, R_A)), np.hstack((R_A, R_neg))))

        return R, R_pos, R_neg, R_A
            
            
    def matrix_p_directional(self, S, C, xi):
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
    

    def matrix_p_local(self, S, C, xi):
        """
        Construct the matrix M_P which correspond to the 1D model when the reversal dependence is on the refractory period, so the parameter S = F*/bar{R}
        """
        S_ind = int(S / self.par.ds)
        __, P_pos, P_neg, P_A = self.matrix_p_directional(S, C, xi)
        P_pos[S_ind, :] -= C
        P_neg[S_ind, :] -= C
        P_A[0, :] += C

        P = np.vstack((np.hstack((P_pos, P_A)), np.hstack((P_A, P_neg))))
        
        return P, P_pos, P_neg, P_A
        

    def C_P(self, S):
        return 0.5 * S / (1 + S)


    def C_R(self, S):
        return 0.5 * self.a / (1 + S)
    

    def compute_eigeinvalues_main(self, values):
        """
        Compute the eigeinvalues of a matrix for different parameters value
        
        """
        P, __, __, __ = self.chosen_f_matrix_p(values[0], values[1], values[2])
        R, __, __, __ = self.chosen_f_matrix_r(values[0], values[1], values[2])

        w1, __ = np.linalg.eig(P)
        w2, __ = np.linalg.eig(R)

        return np.max(w1.real), np.max(w2.real), values[0].copy()
    

    def compute_eigeinvalues(self, f_matrix_p, f_matrix_r):
        """
        Parallelize compute_eigeinvalue
        
        """
        self.chosen_f_matrix_p = f_matrix_p
        self.chosen_f_matrix_r = f_matrix_r
        arrays = Parallel(n_jobs=self.par.n_jobs)(delayed(self.compute_eigeinvalues_main)(values) for values in tqdm(self.par.combined_array))
        arrays = np.array(arrays)
        array_P = np.max(arrays[:, 0].reshape(self.par.S_grid.shape), axis=2)
        array_R = np.max(arrays[:, 1].reshape(self.par.S_grid.shape), axis=2)
        array_S = np.max(arrays[:, 2].reshape(self.par.S_grid.shape), axis=2)
        # num_err = 10e-8
        # array_P[array_P<num_err] = np.nan
        # array_R[array_R<num_err] = np.nan

        return array_P, array_R, array_S
    

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

# # %%
# ds = 0.1
# n = 15
# ns = int(n / ds)
# lin = Linearisation(n=n, ns=ns, ds=ds, delta=1/ds, n_jobs=24, a=1)

# max_eigenvalues = []
# S_array = np.arange(ds, n, 2*ds)
# C_array = np.arange(0, 0.6, 0.02)
# xi_array = np.arange(0, 4, 0.05)
# # Création des grilles pour chaque tableau
# S_grid, C_grid, xi_grid = np.meshgrid(S_array, C_array, xi_array)
# # Empilement des grilles pour obtenir un tableau 2D
# combined_array = np.vstack((S_grid.flatten(), C_grid.flatten(), xi_grid.flatten())).T

# val = 10
# P, P_pos, P_neg, P_A = lin.M_P(2, 1, 1)
# R, R_pos, R_neg, R_A = lin.M_R(2, 1, 1)
# print(np.sum(np.real(P), axis=0))
# print(np.sum(np.real(R), axis=0))

# # %%
# arrays = lin.compute_eigeinvalues(combined_array=combined_array)

# array_P = np.max(arrays[:, 0].reshape(S_grid.shape), axis=2)
# array_R = np.max(arrays[:, 1].reshape(S_grid.shape), axis=2)
# array_S = np.max(arrays[:, 2].reshape(S_grid.shape), axis=2)
# num_err = 10e-8
# array_P[array_P<num_err] = np.nan
# array_R[array_R<num_err] = np.nan

# # %%
# path_save = "results/eigenvalue_map_directional_density/"

# name_save = "refractory_period_modulation_v2.png"
# cmap=plt.get_cmap('plasma_r')
# xmin = S_array[0]
# xmax = S_array[-1]
# ymin = C_array[0]
# ymax = C_array[-1]
# labelsize = 15
# lambda_max = np.nanmax(array_P)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# im = plt.imshow(array_P, extent=[xmin,xmax,ymax,ymin], cmap=cmap, aspect=10, vmin=0, vmax=lambda_max)
# ax.plot(S_array, lin.C_P(S_array), color='lime', linewidth=1, label=r'$\~C=\frac{0.5\times\bar{S}}{1+\bar{S}}$')
# ax.tick_params(labelsize=labelsize)
# ax.set_xlabel(r'$\bar{S}=R^*\times \bar{F}$', fontsize = labelsize)
# ax.set_ylabel(r'$\~C$', fontsize=labelsize)
# plt.xticks(np.arange(0, xmax, step=2), fontsize=labelsize)
# plt.yticks(np.arange(0, ymax, step=0.2), fontsize=labelsize)
# lin.forceAspect(ax, aspect=1)
# v1 = np.linspace(0, lambda_max, 4, endpoint=True)
# cb = plt.colorbar(ticks=v1, shrink=1)
# cb.ax.set_yticklabels(["{:3.1f}".format(i) for i in v1], fontsize='15')
# plt.gca().invert_yaxis()
# cb.set_label("$\lambda$",fontsize=labelsize)
# plt.legend(fontsize=labelsize*0.7)
# fig.savefig(path_save+name_save, bbox_inches="tight", dpi=200)

# name_save = "reversal_rate_modulation_a="+str(lin.a)+"_v2.png"
# cmap=plt.get_cmap('plasma_r')
# xmin = S_array[0]
# xmax = S_array[-1]
# ymin = C_array[0]
# ymax = C_array[-1]
# labelsize = 15
# lambda_max = np.nanmax(array_P)
# array_c_r = lin.C_R(S_array).copy()
# cond = array_c_r < ymax

# fig = plt.figure()
# ax = fig.add_subplot(111)
# im = plt.imshow(array_R, extent=[xmin,xmax,ymax,ymin], cmap=cmap, aspect=10, vmin=0, vmax=lambda_max)
# ax.plot(S_array[cond], array_c_r[cond], color='lime', linewidth=1, label=r'$\~C=\frac{0.5}{1+\bar{S}}$')
# ax.tick_params(labelsize=labelsize)
# ax.set_xlabel(r'$\bar{S}=\bar{R}\times F^*$', fontsize = labelsize)
# ax.set_ylabel(r'$\~C$', fontsize=labelsize)
# plt.xticks(np.arange(0, xmax, step=2), fontsize=labelsize)
# plt.yticks(np.arange(0, ymax, step=0.2), fontsize=labelsize)
# lin.forceAspect(ax, aspect=1)
# v1 = np.linspace(0, lambda_max, 4, endpoint=True)
# cb = plt.colorbar(ticks=v1, shrink=1)
# cb.ax.set_yticklabels(["{:3.1f}".format(i) for i in v1], fontsize='15')
# plt.gca().invert_yaxis()
# cb.set_label("$\lambda$",fontsize=labelsize)
# plt.legend(fontsize=labelsize*0.7)
# fig.savefig(path_save+name_save, bbox_inches="tight", dpi=200)

# # %%
# import os
# import csv
# def initialize_csv(filename):
#     """
#     Initialize csv file.
#     In case the file exist first remove it and write the columns name.
    
#     """
#     if os.path.isfile(filename):
#         os.remove(filename)  # Supprimer le fichier existant s'il existe
#     with open(filename, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)

# def append_to_csv(filename,data):
#     """
#     Write in a csv file
    
#     """
#     with open(filename, 'a', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerows(data)

# path_save = "W:/jb/python/model_1d_linearisation/results/eigenvalue_map_directional_density/"
# filename1 ="data_eigenvalues_directional_density_refractory_period.csv"
# initialize_csv(path_save+filename1)
# append_to_csv(filename=path_save+filename1, data=array_P)
# filename2 ="data_eigenvalues_directional_density_rate_reversal.csv"
# initialize_csv(path_save+filename2)
# append_to_csv(filename=path_save+filename2, data=array_R)

# # %%
# # VINCENT
# xi = 0
# U = 1
# dr = 1
# delta = 1/dr
# n = 10
# nr = int(n/dr)
# r = (np.arange(0,n,dr)).astype(int)
# rmax = 3 # (F * R)
# A = np.zeros((nr, nr), dtype='cfloat')
# B = np.zeros((nr, nr), dtype='cfloat')
# C = np.zeros((nr, nr), dtype='cfloat')
# D = np.zeros((nr, nr), dtype='cfloat')

# for i in range(nr):
#     A[i, i] = A[i, i] - 1j * xi - 1 / dr * (i < nr-1) - (r[i] > rmax)

#     D[i, i] = D[i, i] + 1j * xi - 1 / dr * (i < nr-1) - (r[i] > rmax)
#     if i > 0:
#         A[i, i - 1] = 1 / dr

#         D[i, i - 1] = 1 / dr

#     if i == 0:
#         for j in range(nr):
#             A[i, j] = A[i, j] + U * np.sum(np.exp(-np.maximum(r - rmax, 0)) * (r >= rmax) * dr)
#             B[i, j] = B[i, j] + (r[j] >= rmax) + 0 * U * np.sum(np.exp(-np.maximum(r - rmax, 0)) * (r >= rmax) * dr)

#             C[i, j] = C[i, j] + (r[j] >= rmax) + 0 * U * np.sum(np.exp(-np.maximum(r - rmax, 0)) * (r >= rmax) * dr)

#             D[i, j] = D[i, j] + U * np.sum(np.exp(-np.maximum(r - rmax, 0)) * (r >= rmax) * dr)

#     for j in range(nr):
#         A[i, j] = A[i, j] - 0 * U * np.exp(-np.maximum(r[i] - rmax, 0)) * (r[i] >= rmax) * dr

#         B[i, j] = B[i, j] - U * np.exp(-np.maximum(r[i] - rmax, 0)) * (r[i] >= rmax) * dr

#         C[i, j] = C[i, j] - U * np.exp(-np.maximum(r[i] - rmax, 0)) * (r[i] >= rmax) * dr

#         D[i, j] = D[i, j] - 0 * U * np.exp(-np.maximum(r[i] - rmax, 0)) * (r[i] >= rmax) * dr

# L = np.vstack((np.hstack((A, B)), np.hstack((C, D))))
# print((np.sum(np.real(L), axis=0)))