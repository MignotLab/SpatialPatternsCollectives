# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

import linearisation_igoshin
import parameters_igoshin

lin = linearisation_igoshin.Linearisation(fontsize=40, figsize=(8,8))
par = parameters_igoshin.Parameters()

# %%
# Compute eigenvalues and save
map_eigenvalues = lin.compute_eigeinvalues()

path_save = "results/"
filename ="data_eigenvalues_igoshin.csv"
lin.initialize_csv(path_save+filename)

# Save
lin.append_to_csv(path_file=path_save+filename, data=map_eigenvalues)

filename ="rho_bar_values.csv"
with open(path_save+filename, 'w') as f:
    np.savetxt(f, par.rho_bar_array, delimiter=',')

filename ="q_values.csv"
with open(path_save+filename, 'w') as f:
    np.savetxt(f, par.q_array, delimiter=',')

# %%
import pandas as pd
import numpy as np

# Read_csv
path = "W:/jb/python/myxococcus_xanthus_simu1d/linearisation/igoshin_linearisation/results/"
filename_map_igoshin = "data_eigenvalues_igoshin.csv"
filename_rho_bar = "rho_bar_values.csv"
filename_q = "q_values.csv"

map_igoshin = pd.read_csv(path+filename_map_igoshin, header=None).values
rho_bar_array = np.concatenate(pd.read_csv(path+filename_rho_bar, header=None).values)
q_array = np.concatenate(pd.read_csv(path+filename_q, header=None).values)

lin.map_eigenvalues(eigen_map=map_igoshin, rho_array=rho_bar_array, q_array=q_array, path_save=path)
# %%
