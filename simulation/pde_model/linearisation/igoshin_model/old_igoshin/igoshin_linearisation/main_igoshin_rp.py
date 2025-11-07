# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

import linearisation_igoshin_rp_modulated
import parameters_igoshin

lin_rp = linearisation_igoshin_rp_modulated.Linearisation(fontsize=40, figsize=(8,8))
par = parameters_igoshin.Parameters()

# %%
# Compute eigenvalues and save
map_eigenvalues_rp = lin_rp.compute_eigeinvalues()

path_save = "results/"
filename ="data_eigenvalues_igoshin_rp.csv"
lin_rp.initialize_csv(path_save+filename)

# Save
lin_rp.append_to_csv(path_file=path_save+filename, data=map_eigenvalues_rp)

filename ="rho_bar_values_rp.csv"
with open(path_save+filename, 'w') as f:
    np.savetxt(f, par.rho_bar_array_rp, delimiter=',')

filename ="rho_t_values_rp.csv"
with open(path_save+filename, 'w') as f:
    np.savetxt(f, par.rho_t_array_rp, delimiter=',')

# %%
import pandas as pd
import numpy as np

# Read_csv
path = "W:/jb/python/myxococcus_xanthus_simu1d/linearisation/igoshin_linearisation/results/"
filename_map_igoshin_rp = "data_eigenvalues_igoshin_rp.csv"
filename_rho_bar_rp = "rho_bar_values_rp.csv"
filename_rho_t_rp = "rho_t_values_rp.csv"

map_igoshin_rp = pd.read_csv(path+filename_map_igoshin_rp, header=None).values
rho_bar_array_rp = np.concatenate(pd.read_csv(path+filename_rho_bar_rp, header=None).values)
rho_t_array_rp = np.concatenate(pd.read_csv(path+filename_rho_t_rp, header=None).values)

lin_rp.map_eigenvalues(eigen_map=map_igoshin_rp, rho_array=rho_bar_array_rp, rho_t_array=rho_t_array_rp, path_save=path)

# %%
