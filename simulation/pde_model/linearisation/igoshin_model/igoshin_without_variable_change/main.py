# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import pickle

import igoshin_compute_eigeinvalues as igoshin_compute_eigeinvalues
from parameters import Parameters
from igoshin_matrices import IgoshinMatrix
from tools import Tools
from plot import Plot

par = Parameters()
tool = Tools()
plo = Plot()
mat = IgoshinMatrix()



# %% XI and RHO_BAR
values = par.combined_array_1
grid = par.xi_grid_1
eig = igoshin_compute_eigeinvalues.Eigeinvalues(values, grid)
array_eigenvalues = eig.compute_eigeinvalues(function=eig.compute_eigeinvalues_xi_rho)
# Save
path_save = "results/csv/"
filename = "data_eigenvalues_igoshin_xi_rho.csv"
columns_name = ["eigenvalues", "rho_bar"]
tool.initialize_directory_or_file(path=path_save+filename)
tool.fill_csv(path_save+filename, columns_name, array_eigenvalues, par.rho_bar_array)
# %% Plot 1D XI and RHO_BAR
path = "results/csv/"
filename = "data_eigenvalues_igoshin_xi_rho.csv"
path_save = "results/fig/"
filename_save = "plot_eigenvalues_igoshin_xi_rho"
x_label = r"$\bar\rho$"
y_label = r"Eigeinvalue"
# Read csv
data_array = pd.read_csv(path+filename).values
# Plot
plo.plot_eigenvalues_1d(data_array=data_array, 
                        path_save=path_save,
                        filename=filename_save,
                        x_label=x_label,
                        y_label=y_label)
# %%




# %% XI, w_1 and K_1
values = par.combined_array_3
grid = par.xi_grid_3
eig = igoshin_compute_eigeinvalues.Eigeinvalues(values, grid)
array_eigenvalues = eig.compute_eigeinvalues(function=eig.compute_eigeinvalues_xi_w_1_K_1)
# Save
path_save = "results/csv/"
filename = "data_eigenvalues_igoshin_xi_w_1_K_1.pkl"
tool.initialize_directory_or_file(path=path_save+filename)
data = [array_eigenvalues, par.w_1_array, par.K_1_array]
with open(path_save+filename, 'wb') as pickle_file:
    pickle.dump(data, pickle_file)
# %% Plot XI, w_1 and K_1
path = "results/csv/"
filename = "data_eigenvalues_igoshin_xi_w_1_K_1.pkl"
path_save = "results/fig/"
filename_save = "plot_eigenvalues_igoshin_xi_w_1_K_1"
x_label = r"$w_1$"
y_label = r"$K_1$"
# Read pkl
with open(path+filename, 'rb') as pickle_file:
    data_array = pickle.load(pickle_file)
# Plot
plo.plot_eigenvalues_2d(data_array=data_array,
                        path_save=path_save,
                        filename=filename_save,
                        x_label=x_label,
                        y_label=y_label,
                        type='main')
# %%




# %% XI and RHO_BAR rp modulation
values = par.combined_array_2
grid = par.xi_grid_2
eig = igoshin_compute_eigeinvalues.Eigeinvalues(values, grid)
array_eigenvalues = eig.compute_eigeinvalues(function=eig.compute_eigeinvalues_rp_xi_rho)
# Save
path_save = "results/csv/"
filename = "data_eigenvalues_igoshin_rp_xi_rho.csv"
columns_name = ["eigenvalues", "rho_bar"]
tool.initialize_directory_or_file(path=path_save+filename)
tool.fill_csv(path_save+filename, columns_name, array_eigenvalues, par.rho_bar_rp_array)
# %% Plot 1D XI and RHO_BAR rp modulation
path = "results/csv/"
filename = "data_eigenvalues_igoshin_rp_xi_rho.csv"
path_save = "results/fig/"
filename_save = "plot_eigenvalues_igoshin_rp_xi_rho"
x_label = r"$\rho$"
y_label = r"$\lambda$"
# Read csv
data_array = pd.read_csv(path+filename).values
# Plot
plo.plot_eigenvalues_1d(data_array=data_array, 
                        path_save=path_save,
                        filename=filename_save,
                        x_label=x_label,
                        y_label=y_label)
# %%




# %% XI, PHI_R and K_2 rp modulation
values = par.combined_array_4
grid = par.xi_grid_4
eig = igoshin_compute_eigeinvalues.Eigeinvalues(values, grid)
array_eigenvalues = eig.compute_eigeinvalues(function=eig.compute_eigeinvalues_rp_xi_phi_r_K_2)
# Save
path_save = "results/csv/"
filename = "data_eigenvalues_igoshin_rp_xi_phi_r_K_2.pkl"
tool.initialize_directory_or_file(path=path_save+filename)
data = [array_eigenvalues, par.phi_r_array, par.K_2_array]
with open(path_save+filename, 'wb') as pickle_file:
    pickle.dump(data, pickle_file)
# %% Plot XI, PHI_R and K_2 rp modulation
path = "results/csv/"
filename = "data_eigenvalues_igoshin_rp_xi_phi_r_K_2.pkl"
path_save = "results/fig/"
filename_save = "plot_eigenvalues_igoshin_rp_xi_phi_r_K_2"
x_label = r"$\Phi_R$"
y_label = r"$K_2$"
# Read pkl
with open(path+filename, 'rb') as pickle_file:
    data_array = pickle.load(pickle_file)
# Plot
plo.plot_eigenvalues_2d(data_array=data_array,
                        path_save=path_save,
                        filename=filename_save,
                        x_label=x_label,
                        y_label=y_label,
                        type='rp')














# %% TEST
delta_phi = []
K_2 = []
for rho_bar in par.rho_bar_array:
    delta_phi.append(mat.phi_r(rho_bar))
    K_2.append(mat.K_2(rho_bar))
plt.plot(delta_phi, K_2, linewidth=1)