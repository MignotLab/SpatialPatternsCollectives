# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import pickle

from igoshin_compute_eigeinvalues import Eigeinvalues
from parameters import Parameters
from igoshin_matrices import IgoshinMatrix
from tools import Tools
from plot import Plot

par = Parameters()
tool = Tools()
plo = Plot(par)
mat = IgoshinMatrix(par)



# %% XI, S and Phi_R
exp_name = "xi_S_Phi_R"
values = par.combined_array
grid = par.xi_grid
eig = Eigeinvalues(values, grid, par)
array_eigenvalues = eig.compute_eigeinvalues(function=eig.compute_eigeinvalues_xi_S_Phi_R)
# Save
path_save = "results_"+exp_name+"/csv/"
filename = "data_eigenvalues_igoshin_"+exp_name+"_q="+str(par.q_value_constant)+"_dp="+str(par.dp)+"_rho_w="+str(par.rho_w)+".pkl"
tool.initialize_directory_or_file(path=path_save+filename)
data = [array_eigenvalues, par.S_array, par.Phi_R_array]
with open(path_save+filename, 'wb') as pickle_file:
    pickle.dump(data, pickle_file)

# Plot XI, S and Phi_R
exp_name = "xi_S_Phi_R"
path = "results_"+exp_name+"/csv/"
filename = "data_eigenvalues_igoshin_"+exp_name+"_q="+str(par.q_value_constant)+"_dp="+str(par.dp)+"_rho_w="+str(par.rho_w)+".pkl"
path_save = "results_"+exp_name+"/fig/"
filename_save = "plot_eigenvalues_igoshin_"+exp_name+"_q="+str(par.q_value_constant)+"_dp="+str(par.dp)+"_rho_w="+str(par.rho_w)
x_label = r"$\bar S=\bar\omega_1/\omega_0$"
y_label = r"$\Phi_R$"
cbar_label = r"Eigenvalues $(\Lambda_G)$"
# Read pkl
with open(path+filename, 'rb') as pickle_file:
    data_array_freq = pickle.load(pickle_file)
# Plot
plo.plot_eigenvalues_2d(data_array=data_array_freq,
                        output_folder=path_save,
                        output_file_name=filename_save,
                        x_label=x_label,
                        y_label=y_label,
                        cbar_label=cbar_label,
                        cmap='plasma_r',
                        vmin=0, 
                        vmax=2, 
                        vstep=0.5)

# %% XI, S and Phi_R rp modulation
exp_name = "rp_xi_S_Phi_R"
values = par.combined_array
grid = par.xi_grid
eig = Eigeinvalues(values, grid, par)
array_eigenvalues = eig.compute_eigeinvalues(function=eig.compute_eigeinvalues_rp_xi_S_Phi_R)
# Save
path_save = "results_"+exp_name+"/csv/"
filename = "data_eigenvalues_igoshin_"+exp_name+"_q="+str(par.q_value_constant)+"_dp="+str(par.dp)+"_rho_w="+str(par.rho_w)+".pkl"
tool.initialize_directory_or_file(path=path_save+filename)
data = [array_eigenvalues, par.S_array, par.Phi_R_array]
with open(path_save+filename, 'wb') as pickle_file:
    pickle.dump(data, pickle_file)

# Plot XI, S and Phi_R rp modulation
exp_name = "rp_xi_S_Phi_R"
path = "results_"+exp_name+"/csv/"
filename = "data_eigenvalues_igoshin_"+exp_name+"_q="+str(par.q_value_constant)+"_dp="+str(par.dp)+"_rho_w="+str(par.rho_w)+".pkl"
path_save = "results_"+exp_name+"/fig/"
filename_save = "plot_eigenvalues_igoshin_"+exp_name+"_q="+str(par.q_value_constant)+"_dp="+str(par.dp)+"_rho_w="+str(par.rho_w)
x_label = r"$S=\omega_1/\omega_0$"
y_label = r"$\bar\Phi_R$"
cbar_label = r"Eigenvalues $(\Lambda_L)$"
# Read pkl
with open(path+filename, 'rb') as pickle_file:
    data_array_rp = pickle.load(pickle_file)
# Plot
plo.plot_eigenvalues_2d(data_array=data_array_rp,
                        output_folder=path_save,
                        output_file_name=filename_save,
                        x_label=x_label,
                        y_label=y_label,
                        cbar_label=cbar_label,
                        cmap='plasma_r',
                        vmin=0, 
                        vmax=20, 
                        vstep=5)


# %%
# Plot relative difference between the two maps
cond_both_zero = (data_array_rp[0] < par.dp) & (data_array_freq[0] < par.dp)
cond_freq_zero = data_array_freq[0] < par.dp
cond_rp_zero = data_array_rp[0] < par.dp
map_combine = (data_array_rp[0] - data_array_freq[0]) / (data_array_rp[0] + data_array_freq[0])
map_combine[cond_both_zero] = np.nan
# map_combine[cond_freq_zero] = 1

map_combine_plot = data_array_rp.copy()
map_combine_plot[0] = map_combine

exp_name = "relative_difference"
path = "results_"+exp_name+"/csv/"
path_save = "results_"+exp_name+"/fig/"
filename_save = "plot_eigenvalues_igoshin_"+exp_name+"_q="+str(par.q_value_constant)+"_dp="+str(par.dp)+"_rho_w="+str(par.rho_w)
x_label = r"$S=\omega_1/\omega_0$"
y_label = r"$\Phi_R$"
cbar_label = r"$\frac{\Lambda_G - \Lambda_L}{\Lambda_G + \Lambda_L}$"
cbar_label = "Relative Eigenvalues\n difference"
plo.plot_eigenvalues_2d(data_array=map_combine_plot,
                        output_folder=path_save,
                        output_file_name=filename_save,
                        x_label=x_label,
                        y_label=y_label,
                        cbar_label=cbar_label,
                        cmap='coolwarm_r',
                        vmin=-1, 
                        vmax=1, 
                        vstep=1,
                        uncertenties=False)
