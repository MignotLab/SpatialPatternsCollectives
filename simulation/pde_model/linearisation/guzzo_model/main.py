# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import pickle

from compute_eigeinvalues import Eigeinvalues
from parameters import Parameters
from matrices import Matrix
from tools import Tools
from plot import Plot

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
# %% Plot 1D XI and RHO_BAR LOCAL
path = "results/csv/"
filename_R_local, filename_P_local = "data_eigenvalues_R_local_xi_S.csv","data_eigenvalues_P_local_xi_S.csv"
filename_R_directional, filename_P_directional = "data_eigenvalues_R_directional_xi_S.csv","data_eigenvalues_P_directional_xi_S.csv"
path_save = "results/fig/"
filename_save_R_local, filename_save_P_local = "plot_eigenvalues_R_local_xi_S", "plot_eigenvalues_P_local_xi_S"
filename_save_R_directional, filename_save_P_directional = "plot_eigenvalues_R_directional_xi_S", "plot_eigenvalues_P_directional_xi_S"
x_label_r = r"$\bar S_F$"
y_label_r = r"$\lambda$"
x_label_p = r"$\bar S_R$"
y_label_p = r"$\lambda$"
# Read csv
data_array_R_local, data_array_P_local = pd.read_csv(path+filename_R_local).values, pd.read_csv(path+filename_P_local).values
data_array_R_directional, data_array_P_directional = pd.read_csv(path+filename_R_directional).values, pd.read_csv(path+filename_P_directional).values
# Plot
plo.plot_eigenvalues_1d(data_array=data_array_R_local, 
                        path_save=path_save,
                        filename=filename_save_R_local,
                        x_label=x_label_r,
                        y_label=y_label_r)
plo.plot_eigenvalues_1d(data_array=data_array_P_local, 
                        path_save=path_save,
                        filename=filename_save_P_local,
                        x_label=x_label_p,
                        y_label=y_label_p)
plo.plot_eigenvalues_1d(data_array=data_array_R_directional, 
                        path_save=path_save,
                        filename=filename_save_R_directional,
                        x_label=x_label_r,
                        y_label=y_label_r)
plo.plot_eigenvalues_1d(data_array=data_array_P_directional, 
                        path_save=path_save,
                        filename=filename_save_P_directional,
                        x_label=x_label_p,
                        y_label=y_label_p)
# %%









# %% XI, S and C LOCAL AND DIRECTIONAL
values = par.combined_array_2
grid = par.xi_grid_2
eig = Eigeinvalues(values, grid)
array_eigenvalues_R_local, array_eigenvalues_R_local = eig.compute_eigeinvalues(function=eig.compute_eigeinvalues_main_xi_S_C,
                                                                                matrix_p=mat.matrix_p_local,
                                                                                matrix_r=mat.matrix_r_local,
                                                                                loc_or_dir='loc')
array_eigenvalues_R_directional, array_eigenvalues_P_directional = eig.compute_eigeinvalues(function=eig.compute_eigeinvalues_main_xi_S_C,
                                                                                            matrix_p=mat.matrix_p_directional,
                                                                                            matrix_r=mat.matrix_r_directional,
                                                                                            loc_or_dir='dir')
# Save
path_save = "results/csv/"
filename_R_local, filename_P_local = "data_eigenvalues_R_local_xi_S_C.pkl", "data_eigenvalues_P_local_xi_S_C.pkl"
filename_R_directional, filename_P_directional = "data_eigenvalues_R_directional_xi_S_C.pkl", "data_eigenvalues_P_directional_xi_S_C.pkl"
tool.initialize_directory_or_file(path=path_save)
data_R_local = [array_eigenvalues_R_local, par.S_array, par.C_array]
data_P_local = [array_eigenvalues_R_local, par.S_array, par.C_array]
data_R_directional = [array_eigenvalues_R_directional, par.S_array, par.C_array]
data_P_directional = [array_eigenvalues_R_directional, par.S_array, par.C_array]
with open(path_save+filename_R_local, 'wb') as pickle_file:
    pickle.dump(data_R_local, pickle_file)
with open(path_save+filename_P_local, 'wb') as pickle_file:
    pickle.dump(data_P_local, pickle_file)
with open(path_save+filename_R_directional, 'wb') as pickle_file:
    pickle.dump(data_R_directional, pickle_file)
with open(path_save+filename_P_directional, 'wb') as pickle_file:
    pickle.dump(data_P_directional, pickle_file)
# %% Plot XI, S and C LOCAL AND DIRECTIONAL
path = "results/csv/"
filename_R_local, filename_P_local = "data_eigenvalues_R_local_xi_S_C.pkl", "data_eigenvalues_P_local_xi_S_C.pkl"
filename_R_directional, filename_P_directional = "data_eigenvalues_R_directional_xi_S_C.pkl", "data_eigenvalues_P_directional_xi_S_C.pkl"
path_save = "results/fig/"
filename_save_R_local, filename_save_P_local = "plot_eigenvalues_R_local_xi_S_C", "plot_eigenvalues_P_local_xi_S_C"
filename_save_R_directional, filename_save_P_directional = "plot_eigenvalues_R_directional_xi_S_C", "plot_eigenvalues_P_directional_xi_S_C"
x_label = r"$\bar S$"
y_label = r"$\bar\tilde{C}$"
# Read pkl
with open(path+filename_R_local, 'rb') as pickle_file:
    data_array_R_local = pickle.load(pickle_file)
with open(path+filename_P_local, 'rb') as pickle_file:
    data_array_P_local = pickle.load(pickle_file)
with open(path+filename_R_directional, 'rb') as pickle_file:
    data_array_R_directional = pickle.load(pickle_file)
with open(path+filename_P_directional, 'rb') as pickle_file:
    data_array_P_directional = pickle.load(pickle_file)
# Plot (both even if the matrices are equivalennt to verify if it's the case)
plo.plot_eigenvalues_2d(data_array=data_array_R_local,
                        path_save=path_save,
                        filename=filename_save_R_local,
                        x_label=x_label,
                        y_label=y_label,
                        linewidth=3,
                        plot_curve=True,
                        loc_or_dir='loc')
plo.plot_eigenvalues_2d(data_array=data_array_P_local,
                        path_save=path_save,
                        filename=filename_save_P_local,
                        x_label=x_label,
                        y_label=y_label,
                        linewidth=3,
                        plot_curve=True,
                        loc_or_dir='loc')
plo.plot_eigenvalues_2d(data_array=data_array_R_directional,
                        path_save=path_save,
                        filename=filename_save_R_directional,
                        x_label=x_label,
                        y_label=y_label,
                        linewidth=3,
                        plot_curve=True,
                        loc_or_dir='dir')
plo.plot_eigenvalues_2d(data_array=data_array_P_directional,
                        path_save=path_save,
                        filename=filename_save_P_directional,
                        x_label=x_label,
                        y_label=y_label,
                        linewidth=3,
                        plot_curve=True,
                        loc_or_dir='dir')
# %%
