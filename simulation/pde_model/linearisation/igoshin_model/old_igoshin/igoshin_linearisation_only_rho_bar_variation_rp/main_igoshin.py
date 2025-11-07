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
array_eigenvalues = lin.compute_eigeinvalues()
path_save = "results/"
filename ="data_eigenvalues_igoshin_1d_rp.csv"
columns_name = ["eigenvalues", "rho_bar"]
lin.initialize_directory_or_file(path=path_save+filename)
lin.fill_csv(path_save+filename, columns_name, array_eigenvalues, par.rho_bar_array)

# Save
filename ="rho_bar_values_1d_rp.csv"
column_name = "rho_bar"
lin.initialize_directory_or_file(path=path_save+filename)
lin.fill_csv(path_save+filename, column_name, array_eigenvalues)


# %%
import pandas as pd
import numpy as np

# Read_csv
path = "results/"
filename_data_igoshin = "data_eigenvalues_igoshin_1d_rp.csv"

data_array = pd.read_csv(path+filename_data_igoshin).values

lin.plot_eigenvalues(data_array=data_array, path_save=path)
# %%
