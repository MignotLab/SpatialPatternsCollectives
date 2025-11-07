# %%
import numpy as np
import parameters
import compute_eigeinvalues
par = parameters.Parameters()

lin = compute_eigeinvalues.Linearisation(a=1)
P_local, P_pos_local, P_neg_local, P_A_local = lin.matrix_p_local(4.4, 0.2, 1.5)
R_local, R_pos_local, R_neg_local, R_A_local = lin.matrix_r_local(4.4, 0.2, 1.5)
print('LOCAL')
# print(np.sum(np.real(P_local), axis=0))
# print(np.sum(np.real(R_local), axis=0))
print(np.sum(np.abs(np.sum(np.real(P_local), axis=0))))
print(np.sum(np.abs(np.sum(np.real(R_local), axis=0))))

P_directional, P_pos_directional, P_neg_directional, P_A_directional = lin.matrix_p_directional(2, 0.2, 1.5)
R_directional, R_pos_directional, R_neg_directional, R_A_directional = lin.matrix_r_directional(2, 0.2, 1.5)
print('DIRECTIONAL')
# print(np.sum(np.real(P_directional), axis=0))
# print(np.sum(np.real(R_directional), axis=0))
print(np.sum(np.abs(np.sum(np.real(P_directional), axis=0))))
print(np.sum(np.abs(np.sum(np.real(R_directional), axis=0))))

# # print('P_pos')
# # print(np.real(P_pos_local))
# print('R_pos')
# print(np.real(R_pos_local))
# # print('P_neg')
# # print(np.real(P_neg_local))
# print('R_neg')
# print(np.real(R_neg_local))
# # print('P_A')
# # print(np.real(P_A_local))
# print('R_A')
# print(np.real(R_A_local))
# %%
array_P_local, array_R_local, __ = lin.compute_eigeinvalues(f_matrix_p=lin.matrix_p_local, f_matrix_r=lin.matrix_r_local)
array_P_directional, array_R_directional, __ = lin.compute_eigeinvalues(f_matrix_p=lin.matrix_p_directional, f_matrix_r=lin.matrix_r_directional)

# %%
################################# PLOT #################################
import pandas as pd
import numpy as np
path_C = "W:/jb/python/myxococcus_xanthus_simu1d/linearisation/results/eigenvalue_map_directional_density/C_values.csv"
C_array = np.concatenate(pd.read_csv(path_C, header=None).values)

path_S = "W:/jb/python/myxococcus_xanthus_simu1d/linearisation/results/eigenvalue_map_directional_density/S_values.csv"
S_array = np.concatenate(pd.read_csv(path_S, header=None).values)

path_P_directional = "W:/jb/python/myxococcus_xanthus_simu1d/linearisation/results/eigenvalue_map_directional_density/data_eigenvalues_directional_density_refractory_period.csv"
array_P_directional = pd.read_csv(path_P_directional, header=None).values

path_R_directional = "W:/jb/python/myxococcus_xanthus_simu1d/linearisation/results/eigenvalue_map_directional_density/data_eigenvalues_directional_density_reversal_rate.csv"
array_R_directional = pd.read_csv(path_R_directional, header=None).values

path_P_local = "W:/jb/python/myxococcus_xanthus_simu1d/linearisation/results/eigenvalue_map_local_density/data_eigenvalues_local_density_refractory_period.csv"
array_P_local = pd.read_csv(path_P_local, header=None).values

path_R_local = "W:/jb/python/myxococcus_xanthus_simu1d/linearisation/results/eigenvalue_map_local_density/data_eigenvalues_local_density_reversal_rate.csv"
array_R_local = pd.read_csv(path_R_local, header=None).values

# %%
import plot
import parameters
par = parameters.Parameters()
plo = plot.Plot(fontsize=40, figsize=(8,8))
# plo.multiple_map_eigenvalues(eigen_maps=[array_P_directional, array_P_local],
#                              S_array=S_array,
#                              C_array=C_array,
#                              alpha=1,
#                              linewidth=4,
#                              color_local='#B0E0E6',
#                              color_directional='#FFDAB9',
#                              path_save='results/maps_plot/map_eigenvelues_model_1d_local_directional_signal.png')

    # Rose pastel : '#FFB6C1'
    # Bleu pastel : '#B0E0E6'
    # Vert pastel : '#98FB98'
    # Jaune pastel : '#FFFFE0'
    # Violet pastel : '#E6E6FA'
    # Orange pastel : '#FFDAB9'
# %%
# Plot directional and local on different graph with a map color
plo.map_eigenvalues(eigen_map=array_P_directional, 
                    S_array=S_array, 
                    C_array=C_array, 
                    fct_plot=True, 
                    path_save='results/maps_plot/map_eigenvelues_model_1d_directional_signal_P.png',
                    linewidth=4,
                    legend=True)
plo.map_eigenvalues(eigen_map=array_R_directional, 
                    S_array=S_array, 
                    C_array=C_array, 
                    fct_plot=True, 
                    path_save='results/maps_plot/map_eigenvelues_model_1d_directional_signal_R.png', 
                    linewidth=4,
                    legend=False)
plo.map_eigenvalues(eigen_map=array_P_local, 
                    S_array=S_array, 
                    C_array=C_array, 
                    fct_plot=True, 
                    path_save='results/maps_plot/map_eigenvelues_model_1d_local_signal_P.png', 
                    linewidth=4,
                    legend=True)
plo.map_eigenvalues(eigen_map=array_R_local, 
                    S_array=S_array, 
                    C_array=C_array, 
                    fct_plot=True, 
                    path_save='results/maps_plot/map_eigenvelues_model_1d_local_signal_R.png', 
                    linewidth=4,
                    legend=False)

# Plot directional and local on different graph with a map color without the functions on it
plo.map_eigenvalues(eigen_map=array_P_directional, 
                    S_array=S_array, 
                    C_array=C_array, 
                    fct_plot=False, 
                    path_save='results/maps_plot/map_eigenvelues_model_1d_directional_signal_P_without_fct.png',
                    linewidth=4,
                    legend=True)
plo.map_eigenvalues(eigen_map=array_R_directional, 
                    S_array=S_array, 
                    C_array=C_array, 
                    fct_plot=False, 
                    path_save='results/maps_plot/map_eigenvelues_model_1d_directional_signal_R_without_fct.png', 
                    linewidth=4,
                    legend=False)
plo.map_eigenvalues(eigen_map=array_P_local, 
                    S_array=S_array, 
                    C_array=C_array, 
                    fct_plot=False, 
                    path_save='results/maps_plot/map_eigenvelues_model_1d_local_signal_P_without_fct.png', 
                    linewidth=4,
                    legend=True)
plo.map_eigenvalues(eigen_map=array_R_local, 
                    S_array=S_array, 
                    C_array=C_array, 
                    fct_plot=False, 
                    path_save='results/maps_plot/map_eigenvelues_model_1d_local_signal_R_without_fct.png', 
                    linewidth=4,
                    legend=False)

# %%
# Save csv file
path_save = "results/eigenvalue_map_local_density/"

filename ="data_eigenvalues_local_density_refractory_period.csv"
lin.initialize_csv(path_save+filename)
lin.append_to_csv(path_file=path_save+filename, data=array_P_local)

filename ="data_eigenvalues_local_density_reversal_rate.csv"
lin.initialize_csv(path_save+filename)
lin.append_to_csv(path_file=path_save+filename, data=array_R_local)

filename ="S_values.csv"
with open(path_save+filename, 'w') as f:
    np.savetxt(f, par.S_array, delimiter=',')
# lin.initialize_csv(path_save+filename)
# lin.append_to_csv(path_file=path_save+filename, data=par.S_array)

filename ="C_values.csv"
with open(path_save+filename, 'w') as f:
    np.savetxt(f, par.C_array, delimiter=',')
# lin.initialize_csv(path_save+filename)
# lin.append_to_csv(path_file=path_save+filename, data=par.C_array)

# Save csv file
path_save = "results/eigenvalue_map_directional_density/"

filename ="data_eigenvalues_directional_density_refractory_period.csv"
lin.initialize_csv(path_save+filename)
lin.append_to_csv(path_file=path_save+filename, data=array_P_directional)

filename ="data_eigenvalues_directional_density_reversal_rate.csv"
lin.initialize_csv(path_save+filename)
lin.append_to_csv(path_file=path_save+filename, data=array_R_directional)

filename ="S_values.csv"
with open(path_save+filename, 'w') as f:
    np.savetxt(f, par.S_array, delimiter=',')
# lin.initialize_csv(path_save+filename)
# lin.append_to_csv(path_file=path_save+filename, data=par.S_array)

filename ="C_values.csv"
with open(path_save+filename, 'w') as f:
    np.savetxt(f, par.C_array, delimiter=',')
# lin.initialize_csv(path_save+filename)
# lin.append_to_csv(path_file=path_save+filename, data=par.C_array)
# %%
