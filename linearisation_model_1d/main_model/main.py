# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import pickle

from compute_eigeinvalues import Eigeinvalues
from parameters import Parameters
from matrices import Matrix
from tools import Tools

par = Parameters()
tool = Tools()
plo = Plot()
mat = Matrix()


# %% XI and RHO_BAR
values = par.combined_array_1
grid = par.xi_grid_1
eig = Eigeinvalues(values, grid)
array_eigenvalues_R_local, array_eigenvalues_P_local = eig.compute_eigeinvalues(function=eig.compute_eigeinvalues_main_xi_S,
                                                                                matrix_p=mat.matrix_p_local,
                                                                                matrix_r=mat.matrix_r_local,
                                                                                loc_or_dir='loc')
array_eigenvalues_R_directional, array_eigenvalues_P_directional = eig.compute_eigeinvalues(function=eig.compute_eigeinvalues_main_xi_S,
                                                                                            matrix_p=mat.matrix_p_directional,
                                                                                            matrix_r=mat.matrix_r_directional,
                                                                                            loc_or_dir='dir')

# Save
path_save = "results/csv/"
filename_R_local, filename_P_local = "data_eigenvalues_R_local_xi_S.csv", "data_eigenvalues_P_local_xi_S.csv"
filename_R_directional, filename_P_directional = "data_eigenvalues_R_directional_xi_S.csv", "data_eigenvalues_P_directional_xi_S.csv"
columns_name = ["eigenvalues", "S"]
tool.initialize_directory_or_file(path=path_save+filename_R_local)
tool.initialize_directory_or_file(path=path_save+filename_P_local)
tool.initialize_directory_or_file(path=path_save+filename_R_directional)
tool.initialize_directory_or_file(path=path_save+filename_P_directional)
tool.fill_csv(path_save+filename_R_local, columns_name, array_eigenvalues_R_local, par.S_array)
tool.fill_csv(path_save+filename_P_local, columns_name, array_eigenvalues_P_local, par.S_array)
tool.fill_csv(path_save+filename_R_directional, columns_name, array_eigenvalues_R_directional, par.S_array)
tool.fill_csv(path_save+filename_P_directional, columns_name, array_eigenvalues_P_directional, par.S_array)