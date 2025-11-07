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



# %% XI and S
exp_name = "xi_S"
values = par.combined_array_1
grid = par.xi_grid_1
eig = Eigeinvalues(values, grid, par)
array_eigenvalues, array_eigenvectors = eig.compute_eigeinvalues(function=eig.compute_eigeinvalues_xi_S, dp=par.dp)
# Save
path_save = "results_"+exp_name+"/csv/"
filename = "data_eigenvalues_igoshin_"+exp_name+"_q="+str(par.q_value_constant)+"_dp="+str(par.dp)+"_rho_w="+str(par.rho_w)+".csv"
columns_name = ["eigenvalues", "S", "eigenvectors"]
tool.initialize_directory_or_file(path=path_save+filename)
tool.fill_csv(path_save+filename, columns_name, array_eigenvalues, par.S_array, array_eigenvectors)
# %% Plot 1D XI and S
exp_name = "xi_S"
path = "results_"+exp_name+"/csv/"
filename = "data_eigenvalues_igoshin_"+exp_name+"_q="+str(par.q_value_constant)+"_dp="+str(par.dp)+"_rho_w="+str(par.rho_w)+".csv"
path_save = "results_"+exp_name+"/fig/"
filename_save = "plot_eigenvalues_igoshin_"+exp_name+"_q="+str(par.q_value_constant)+"_dp="+str(par.dp)+"_rho_w="+str(par.rho_w)
x_label = r"$S$"
y_label = "Eigeinvalue"
# Read csv
data_array = pd.read_csv(path+filename).values
# Plot
plo.plot_eigenvalues_1d(data_array=data_array, 
                        path_save=path_save,
                        filename=filename_save,
                        x_label=x_label,
                        y_label=y_label)


# %% XI and S linear
exp_name = "xi_S_linear"
values = par.combined_array_1
grid = par.xi_grid_1
eig = Eigeinvalues(values, grid, par)
array_eigenvalues, array_eigenvectors  = eig.compute_eigeinvalues(function=eig.compute_eigeinvalues_xi_S_linear, dp=par.dp)
# Save
path_save = "results_"+exp_name+"/csv/"
filename = "data_eigenvalues_igoshin_"+exp_name+"_q="+str(par.q_value_constant)+"_dp="+str(par.dp)+"_rho_w="+str(par.rho_w)+".csv"
columns_name = ["eigenvalues", "S", "array_eigenvectors"]
tool.initialize_directory_or_file(path=path_save+filename)
tool.fill_csv(path_save+filename, columns_name, array_eigenvalues, par.S_array, array_eigenvectors)
# %% Plot 1D XI and S linear
exp_name = "xi_S_linear"
path = "results_"+exp_name+"/csv/"
filename = "data_eigenvalues_igoshin_"+exp_name+"_q="+str(par.q_value_constant)+"_dp="+str(par.dp)+"_rho_w="+str(par.rho_w)+".csv"
path_save = "results_"+exp_name+"/fig/"
filename_save = "plot_eigenvalues_igoshin_"+exp_name+"_q="+str(par.q_value_constant)+"_dp="+str(par.dp)+"_rho_w="+str(par.rho_w)
x_label = r"$S$"
y_label = "Eigeinvalue"
# Read csv
data_array = pd.read_csv(path+filename).values
# Plot
plo.plot_eigenvalues_1d(data_array=data_array, 
                        path_save=path_save,
                        filename=filename_save,
                        x_label=x_label,
                        y_label=y_label)
# %%
par = Parameters()
mat = IgoshinMatrix()
rho = np.linspace(0, 6, 100)
q = 4
rho_w = 2
w_n = 4
def w_1_fct(rho_bar, q, rho_w, w_n):
    """
    Function \omega_1
    
    """
    return w_n * rho_bar**q / (rho_bar**q + rho_w**q)

w_1 = w_1_fct(rho_bar=rho, q=q, rho_w=rho_w, w_n=w_n)
fig, ax = plt.subplots()
ax.plot(rho, w_1, label=r"$\omega_1$")
ax.set_xlabel(r"$\bar\rho$")
ax.set_ylabel(r"$\omega_1$")

def slope_w_1(rho_bar, q, rho_w, w_n):
    """
    Slope of w_1
    
    """
    num = q * w_n * rho_w * rho_bar**(q-1)
    den = (rho_bar**q + rho_w**q)**2

    return num / den

s_w_1 = slope_w_1(rho_bar=rho, q=q, rho_w=rho_w, w_n=w_n)
# plt.figure()
ax1 = plt.twinx()
ax1.plot(rho, s_w_1, color='k', label=r"slope of $\omega_1$")
ax1.set_ylabel("slope of $\omega_1$")
ax.legend()
ax1.legend()







# %% XI and rho
exp_name = "xi_rho"
values = par.combined_array_2
grid = par.xi_grid_2
eig = Eigeinvalues(values, grid, par)
array_eigenvalues, array_eigenvectors = eig.compute_eigeinvalues(function=eig.compute_eigeinvalues_xi_rho, dp=par.dp)
# Save
path_save = "results_"+exp_name+"/csv/"
filename = "data_eigenvalues_igoshin_"+exp_name+"_q="+str(par.q_value_constant)+"_dp="+str(par.dp)+"_rho_w="+str(par.rho_w)+".csv"
columns_name = ["eigenvalues", "rho", "array_eigenvectors"]
tool.initialize_directory_or_file(path=path_save+filename)
tool.fill_csv(path_save+filename, columns_name, array_eigenvalues, par.rho_bar_array, array_eigenvectors)
# %% Plot 1D XI and rho
exp_name = "xi_rho"
path = "results_"+exp_name+"/csv/"
filename = "data_eigenvalues_igoshin_"+exp_name+"_q="+str(par.q_value_constant)+"_dp="+str(par.dp)+"_rho_w="+str(par.rho_w)+".csv"
path_save = "results_"+exp_name+"/fig/"
filename_save = "plot_eigenvalues_igoshin_"+exp_name+"_q="+str(par.q_value_constant)+"_dp="+str(par.dp)+"_rho_w="+str(par.rho_w)
x_label = r"$\bar\rho$"
y_label = "Eigeinvalue"
# Read csv
data_array = pd.read_csv(path+filename).values
# Plot
plo.plot_eigenvalues_1d(data_array=data_array, 
                        path_save=path_save,
                        filename=filename_save,
                        x_label=x_label,
                        y_label=y_label)
# %% Plots multiple values 1D XI and RHO
exp_name = "xi_rho"
path = "results_"+exp_name+"/csv/"
q_names = [0.5, 1, 2, 3, 4]

filenames = []
data_arrays = []
for i in range(len(q_names)):
    filenames.append("data_eigenvalues_igoshin_"+exp_name+"_q="+str(q_names[i])+"_dp="+str(par.dp)+"_rho_w="+str(par.rho_w)+".csv")
    data_arrays.append(pd.read_csv(path+filenames[i]).values)

path_save = "results_"+exp_name+"/fig/"
filename_save = "plot_eigenvalues_igoshin_"+exp_name+"_q="+str(q_names)+"_dp="+str(par.dp)+"_rho_w="+str(par.rho_w)
x_label = r"$\bar\rho$"
y_label = "Eigeinvalue"
colors = ['k', 'r', 'g', 'b', 'violet']
legend_names = ['q=0.5', 'q=1', 'q=2', 'q=3', 'q=4']
# Plot
plo.plot_eigenvalues_1d_multiple(data_arrays=data_arrays, 
                                 path_save=path_save,
                                 filename=filename_save,
                                 legend_names=legend_names,
                                 colors=colors,
                                 x_label=x_label,
                                 y_label=y_label,
                                 xmax=12)


# %% XI, S and K_1
exp_name = "xi_S_K_1"
values = par.combined_array_4
grid = par.xi_grid_4
eig = Eigeinvalues(values, grid, par)
array_eigenvalues, array_eigenvectors = eig.compute_eigeinvalues(function=eig.compute_eigeinvalues_xi_S_K_1, dp=par.dp)
# Save
path_save = "results_"+exp_name+"/csv/"
filename = "data_eigenvalues_igoshin_"+exp_name+"_q="+str(par.q_value_constant)+"_dp="+str(par.dp)+"_rho_w="+str(par.rho_w)+".pkl"
tool.initialize_directory_or_file(path=path_save+filename)
data = [array_eigenvalues, par.S_array, par.K_1_array, array_eigenvectors]
with open(path_save+filename, 'wb') as pickle_file:
    pickle.dump(data, pickle_file)
# %% Plot XI, S and K_1
exp_name = "xi_S_K_1"
path = "results_"+exp_name+"/csv/"
filename = "data_eigenvalues_igoshin_"+exp_name+"_q="+str(par.q_value_constant)+"_dp="+str(par.dp)+"_rho_w="+str(par.rho_w)+".pkl"
path_save = "results_"+exp_name+"/fig/"
filename_save = "plot_eigenvalues_igoshin_"+exp_name+"_q="+str(par.q_value_constant)+"_dp="+str(par.dp)+"_rho_w="+str(par.rho_w)
x_label = r"$S$"
y_label = r"$\bar K_1$"
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












# %% XI and S rp modulation linear
exp_name = "rp_xi_S_linear"
values = par.combined_array_3
grid = par.xi_grid_3
eig = Eigeinvalues(values, grid, par)
array_eigenvalues, array_eigenvectors = eig.compute_eigeinvalues(function=eig.compute_eigeinvalues_rp_xi_S, dp=par.dp)
# Save
path_save = "results_"+exp_name+"/csv/"
filename = "data_eigenvalues_igoshin_"+exp_name+"_q="+str(par.q_value_constant)+"_dp="+str(par.dp)+"_rho_w="+str(par.rho_w)+".csv"
columns_name = ["eigenvalues", "S", "array_eigenvectors"]
tool.initialize_directory_or_file(path=path_save+filename)
tool.fill_csv(path_save+filename, columns_name, array_eigenvalues, par.S_array, array_eigenvectors)
# %% Plot 1D XI and S rp modulation linear
exp_name = "rp_xi_S_linear"
path = "results_"+exp_name+"/csv/"
filename = "data_eigenvalues_igoshin_"+exp_name+"_q="+str(par.q_value_constant)+"_dp="+str(par.dp)+"_rho_w="+str(par.rho_w)+".csv"
path_save = "results_"+exp_name+"/fig/"
filename_save = "plot_eigenvalues_igoshin_"+exp_name+"_q="+str(par.q_value_constant)+"_dp="+str(par.dp)+"_rho_w="+str(par.rho_w)
x_label = r"$S$"
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




# %% XI, S and K_1 rp modulation
exp_name = "rp_xi_S_K_1"
values = par.combined_array_4
grid = par.xi_grid_4
eig = Eigeinvalues(values, grid, par)
array_eigenvalues, array_eigenvectors = eig.compute_eigeinvalues(function=eig.compute_eigeinvalues_rp_xi_S_K_1, dp=par.dp)
# Save
path_save = "results_"+exp_name+"/csv/"
filename = "data_eigenvalues_igoshin_"+exp_name+"_q="+str(par.q_value_constant)+"_dp="+str(par.dp)+"_rho_w="+str(par.rho_w)+".pkl"
tool.initialize_directory_or_file(path=path_save+filename)
data = [array_eigenvalues, par.S_array, par.K_1_array, array_eigenvectors]
with open(path_save+filename, 'wb') as pickle_file:
    pickle.dump(data, pickle_file)
# %% Plot XI, S and K_1 rp modulation
exp_name = "rp_xi_S_K_1"
path = "results_"+exp_name+"/csv/"
filename = "data_eigenvalues_igoshin_"+exp_name+"_q="+str(par.q_value_constant)+"_dp="+str(par.dp)+"_rho_w="+str(par.rho_w)+".pkl"
path_save = "results_"+exp_name+"/fig/"
filename_save = "plot_eigenvalues_igoshin_"+exp_name+"_q="+str(par.q_value_constant)+"_dp="+str(par.dp)+"_rho_w="+str(par.rho_w)
x_label = r"$S$"
y_label = r"$\hat K_1$"
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



# %% Plot Eigenvectors norm for rp modulation and frequency modulation
# Main
exp_name = "xi_S_linear"
path = "results_"+exp_name+"/csv/"
filename = "data_eigenvalues_igoshin_"+exp_name+"_q="+str(par.q_value_constant)+"_dp="+str(par.dp)+"_rho_w="+str(par.rho_w)+".csv"
# Read csv
data_array_main = pd.read_csv(path+filename).values

# RP
exp_name = "rp_xi_S_linear"
path = "results_"+exp_name+"/csv/"
filename = "data_eigenvalues_igoshin_"+exp_name+"_q="+str(par.q_value_constant)+"_dp="+str(par.dp)+"_rho_w="+str(par.rho_w)+".csv"
x_label = r"$S$"
y_label = r"Eigeinvalue"
# Read csv
data_array_rp = pd.read_csv(path+filename).values


exp_name = "main_and_rp_linear"
data_arrays = [data_array_main, data_array_rp]
path_save = "results_"+exp_name+"/fig/"
filename_save = "plot_eigenvectors_igoshin_"+exp_name+"_q="+"_dp="+str(par.dp)+"_rho_w="+str(par.rho_w)
x_label = r"$S$"
y_label = r"Norms of eigenvectors"
colors = ['k', 'violet']
legend_names = ['main', 'rp']
# Plot
plo.plot_eigenvectors_1d_multiple(data_arrays=data_arrays, 
                                 path_save=path_save,
                                 filename=filename_save,
                                 legend_names=legend_names,
                                 colors=colors,
                                 x_label=x_label,
                                 y_label=y_label,
                                 xmax=False)











# %% TEST
delta_phi = []
K_2 = []
for rho_bar in par.rho_bar_array:
    delta_phi.append(mat.phi_r(rho_bar))
    K_2.append(mat.K_2(rho_bar))
plt.plot(delta_phi, K_2, linewidth=1)


# %%
from tqdm import tqdm
values = par.combined_array_4
grid = par.xi_grid_4
eig = Eigeinvalues(values, grid, par)
a = []
S = 3
K_1 = 0.1

for xi in tqdm(par.xi_array):
    a.append(eig.compute_eigeinvalues_rp_xi_S_K_1(np.array([xi,S,K_1])))

print(np.max(np.array(a)))

# %%
S = 0
K_1 = par.q_value_constant_value_constant / np.pi
g, __, __, __ = mat.main_matrix(0.2,S,K_1)
for xi in tqdm(par.xi_array):
    np.linalg.eig(g)



# %%
import numpy as np
from tqdm import tqdm
from scipy.sparse.linalg import eigs
from scipy.linalg import eigvals
from parameters import Parameters
from igoshin_matrices import IgoshinMatrix
par = Parameters()
mat = IgoshinMatrix()

xi = 0.2
S = 1
K_1 = 0.5

g, __, __, __ = mat.main_matrix(xi, S, K_1)
print("g=\n", g.real)

l, __, __, __, __ = mat.rp_matrix_L(xi, S, K_1)
print("l=\n", l.real)

b, __, __ = mat.rp_matrix_B(K_1)
print("b=\n", b.real)


e = eigvals(l, b)

print("Max eigenvalue", np.max(e.real))
# eig = []
# for xi in tqdm(par.xi_array):
#     g, __, __, __ = main_matrix(xi, S, K_1)
#     e, __ = np.linalg.eig(g)
#     eig.append(np.max(e.real))

# %%
# print("sum of matrix columns\n", np.sum(g.real, axis=0))
print("eigeinvalue max:", np.max(np.array(eig)))
print('matrix:\n', np.round(g.real, 2))
print('matrix sum columns:\n', np.sum(np.sum(g.real, axis=0)))
# %%
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
xi = 1
S = 2
K_1 = 1
dp = par.dp

g, g_plus, g_minus, g_a = mat.main_matrix(xi, S, K_1, dp)
l, l_plus, l_minus, l_a_plus, l_a_minus = mat.rp_matrix_L(xi, S, K_1, dp)
b, b_id, b_a = mat.rp_matrix_B(K_1, dp)
print('C: ', mat.K_1_linear(S))
print('g_plus:\n', g_plus.real)
print('g_minus:\n', g_minus.real)
print('g_plus imag:\n', g_plus.imag)
print('g_minus imag:\n', g_minus.imag)
print('g_a:\n', g_a.real)
print('g_a imag:\n', g_a.imag)
print('sum: ', np.sum(g.real, axis=0))