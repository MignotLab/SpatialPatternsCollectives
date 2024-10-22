# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import parameters
import plots
import tools
par = parameters.Parameters()
tool = tools.Tools()
# plo = plots.Plots(fontsize=30)
# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] = 'cmr10'  # "Computer Modern Roman"
# %%
df = pd.read_csv(par.path_reversals+'data_rev'+par.end_name_file+'.csv')
# %%
cond_frame = df.loc[:, 'frame'] == 0
len(df.loc[cond_frame])
# %%
path_simu_linear = "W:/jb/python/myxococcus_xanthus_simu2d/results/2023_08_24_big_simu_and_non_reversing/sample5/"
path_simu_sig = "W:/jb/python/myxococcus_xanthus_simu2d/results/2023_08_24_big_simu_and_non_reversing/sample6/"
filename = "coords__24660_bacts__tbf=120_secondes__space_size=390.csv"
df_simu_linear = pd.read_csv(path_simu_linear+filename)
df_simu_sig = pd.read_csv(path_simu_sig+filename)

# %%
space_size = 390

indices_time = np.unique(df_simu_linear.loc[:, 'frame'])
density_simu_linear = np.zeros((2, len(indices_time)))
density_simu_sig = np.zeros((2, len(indices_time)))

for count, frame in enumerate(indices_time):
    cond_time_linear = df_simu_linear.loc[:, 'frame'] == frame
    cond_time_sig = df_simu_sig.loc[:, 'frame'] == frame

    density_simu_linear[0, count] = np.sum(df_simu_linear.loc[cond_time_linear, 'x4'] < space_size / 2)
    density_simu_linear[1, count] = np.sum(df_simu_linear.loc[cond_time_linear, 'x4'] > space_size / 2)
    density_simu_sig[0, count] = np.sum(df_simu_sig.loc[cond_time_sig, 'x4'] < space_size / 2)
    density_simu_sig[1, count] = np.sum(df_simu_sig.loc[cond_time_sig, 'x4'] > space_size / 2)
# %%
color_rippling = tool.get_rgba_color(color_name='royalblue', alpha=0.4)
color_swarming = tool.get_rgba_color(color_name='limegreen', alpha=0.4)
fontsize = 30
tbf = 2 # min
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(indices_time*tbf, density_simu_linear[0, :], color=color_rippling, label='rippling', linewidth=3)
ax2 = ax.twinx()
ax2.plot(indices_time*tbf, density_simu_linear[1, :], color=color_swarming, label='swarming', linewidth=3)
# ax.set_title('linear', fontsize=fontsize)
# ax.set_ylim(15000, 16400)
# ax2.set_xlim(9000, 11000)
ax.set_xlabel("Time (min)", fontsize=fontsize)
ax.set_ylabel("/# Bacteria in rippling", fontsize=fontsize)
ax2.set_ylabel("/# Bacteria in swarming", fontsize=fontsize)
ax.tick_params(labelsize=fontsize)
ax2.tick_params(labelsize=fontsize)
# ax.legend(loc='upper left', fontsize=fontsize/1.5)
# ax2.legend(loc='upper right', fontsize=fontsize/1.5)
# array_legend=[Line2D([0], [0], linestyle='-', color=color_rippling, lw=3, label='rippling'),
#               Line2D([0], [0], linestyle='-', color=color_swarming, lw=3, label='swarming')]
# ax.legend(handles=array_legend,
#                   loc='center left', bbox_to_anchor=(1.2, 0.5),
#                   fontsize=fontsize)

color_rippling = tool.get_rgba_color(color_name='royalblue', alpha=0.4)
color_swarming = tool.get_rgba_color(color_name='limegreen', alpha=0.4)
fontsize = 30
tbf = 2 # min
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(indices_time*tbf, density_simu_sig[0, :], color=color_rippling, label='rippling', linewidth=3)
ax2 = ax.twinx()
ax2.plot(indices_time*tbf, density_simu_sig[1, :], color=color_swarming, label='swarming', linewidth=3)
# ax.set_title('linear', fontsize=fontsize)
# ax.set_ylim(15000, 16400)
# ax2.set_xlim(9000, 11000)
ax.set_xlabel("Time (min)", fontsize=fontsize)
ax.set_ylabel("/# bacteria rippling", fontsize=fontsize)
ax2.set_ylabel("/# bacteria swarming", fontsize=fontsize)
ax.tick_params(labelsize=fontsize)
ax2.tick_params(labelsize=fontsize)
# ax.legend(loc='upper left', fontsize=fontsize/1.5)
# ax2.legend(loc='upper right', fontsize=fontsize/1.5)
# array_legend=[Line2D([0], [0], linestyle='-', color=color_rippling, lw=3, label='rippling'),
#               Line2D([0], [0], linestyle='-', color=color_swarming, lw=3, label='swarming')]
# ax.legend(handles=array_legend,
#                   loc='center left', bbox_to_anchor=(1.2, 0.5),
#                   fontsize=fontsize)

# fig, ax = plt.subplots(figsize=(8,6))
# plt.plot(indices_time*tbf, density_simu_linear[0, :])
# plt.plot(indices_time*tbf, density_simu_sig[0, :])

# fig, ax = plt.subplots(figsize=(8,6))
# plt.plot(indices_time*tbf, density_simu_linear[1, :])
# plt.plot(indices_time*tbf, density_simu_sig[1, :])
# %%
cond_time_1 = density_simu_linear.loc[:, 'frame'] == 74
cond_time_2 = density_simu_sig.loc[:, 'frame'] == 74

plt.figure(figsize=(20,20))
x = density_simu_linear.loc[cond_time_1, 'x4']
y = density_simu_linear.loc[cond_time_1, 'y4']
plt.scatter(x, y)

plt.figure(figsize=(20,20))
x = density_simu_sig.loc[cond_time_2, 'x4']
y = density_simu_sig.loc[cond_time_2, 'y4']
plt.scatter(x, y)

# %%
# Polarity
path_folder = "W:/jb/python/notebook/Myxo/tracking/save_traj/2021_12_09_cover_area=0.6/"
filename = "df2_v14_max_dist=23_t=1178.csv"
df = pd.read_csv(path_folder+filename)
df.columns
# %%
bins = 90
# Polarity
cond_low_polarity = (df.loc[:, 'polarity_mean'] < 0)
cond_high_polarity = (df.loc[:, 'polarity_mean'] > 0)

tbr_low_polarity = df.loc[cond_low_polarity, 'tbr'].values * 2 / 60
tbr_low_polarity = tbr_low_polarity[tbr_low_polarity > 0]
tbr_high_polarity = df.loc[cond_high_polarity, 'tbr'].values * 2 / 60
tbr_high_polarity = tbr_high_polarity[tbr_high_polarity > 0]

plt.figure()
histogram_tbr_low_polarity = plt.hist(tbr_low_polarity, bins=bins, alpha=0.25, range=(0,30))
plt.figure()
histogram_tbr_high_polarity = plt.hist(tbr_high_polarity, bins=bins, alpha=0.25, range=(0,30))

# Neighbours
cond_zero_neighbours = df.loc[:, 'n_neighbors'] == 0
cond_low_neighbours = df.loc[:, 'n_neighbors'] <= 3
cond_high_neighbours = df.loc[:, 'n_neighbors'] > 3

tbr_zero_neighbours = df.loc[cond_zero_neighbours, 'tbr'].values * 2 / 60
tbr_zero_neighbours = tbr_zero_neighbours[tbr_zero_neighbours > 0]
tbr_low_neighbours = df.loc[cond_low_neighbours, 'tbr'].values * 2 / 60
tbr_low_neighbours = tbr_low_neighbours[tbr_low_neighbours > 0]
tbr_high_neighbours = df.loc[cond_high_neighbours, 'tbr'].values * 2 / 60
tbr_high_neighbours = tbr_high_neighbours[tbr_high_neighbours > 0]

# plt.figure()
# histogram_tbr_zero_neighbours = plt.hist(tbr_zero_neighbours, bins=bins, alpha=0.25)
# plt.xlim(0,30)

plt.figure()
histogram_tbr_low_neighbours = plt.hist(tbr_low_neighbours, bins=bins, alpha=0.25, range=(0,30))
plt.figure()
histogram_tbr_high_neighbours = plt.hist(tbr_high_neighbours, bins=bins, alpha=0.25, range=(0,30))

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'cmr10'  # "Computer Modern Roman"
path_folder = "W:/jb/python/notebook/Myxo/tracking/save_traj/2021_12_09_cover_area=0.6/" # Put your own path
path_save = 'save_plots/' # Save in a specific folder
filename_list = ['tbr_2021_12_09_low_polarity', 'tbr_2021_12_09_high_polarity', 'tbr_2021_12_09_low_neighbours', 'tbr_2021_12_09_high_neighbours']
for filename in filename_list:
    tbr = pd.read_csv(path_folder+filename+".csv").values.flatten() # Read the file
    mean_tbr = np.mean(tbr)
    fontsize = 30 # Taille de la police
    bins = 90 # Nombre de bins
    fig, ax = plt.subplots(figsize=(8,6)) # You can change the size of the figure here
    ax.hist(tbr, bins=bins, alpha=0.35, range=(0,30), histtype='bar', ec='black', color="royalblue", label="Mean TBR: "+str(round(mean_tbr,2))+" minutes") # You can change the colors of the histogram here with alpha and color
    ax.set_xlabel("Time between reversals (min)", fontsize=fontsize) # Title of the x-axis
    ax.set_ylabel("Count", fontsize=fontsize)# Title of the y-axis
    ax.tick_params(labelsize=fontsize) # Put the size of the police for the axis numbers
    ax.legend(fontsize=fontsize*0.8)
    fig.savefig(path_save+filename+".png", bbox_inches='tight', dpi=100)
    plt.close()
# %%
# Save
file_name_save_low_polarity = path_folder+'tbr_2021_12_09_low_polarity.csv'
file_name_save_high_polarity = path_folder+'tbr_2021_12_09_high_polarity.csv'
file_name_save_low_neighbours = path_folder+'tbr_2021_12_09_low_neighbours.csv'
file_name_save_high_neighbours = path_folder+'tbr_2021_12_09_high_neighbours.csv'
np.savetxt(file_name_save_low_polarity, tbr_low_polarity, delimiter=',', header='tbr_low_polarity', comments='')
np.savetxt(file_name_save_high_polarity, tbr_high_polarity, delimiter=',', header='tbr_high_polarity', comments='')
np.savetxt(file_name_save_low_neighbours, tbr_low_neighbours, delimiter=',', header='tbr_low_neighbours', comments='')
np.savetxt(file_name_save_high_neighbours, tbr_high_neighbours, delimiter=',', header='tbr_high_neighbours', comments='')

# %%
import csv

columns = ['x', 'y']
file_name_save_low_polarity = path_folder+'tbr_2021_12_09_low_polarity.csv'
file_name_save_high_polarity = path_folder+'tbr_2021_12_09_high_polarity.csv'
file_name_save_low_neighbours = path_folder+'tbr_2021_12_09_low_neighbours.csv'
file_name_save_high_neighbours = path_folder+'tbr_2021_12_09_high_neighbours.csv'

tool.initialize_directory_or_file(file_name_save_low_polarity, columns)
with open(file_name_save_low_polarity, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    width = histogram_tbr_low_polarity[1][1:] - histogram_tbr_low_polarity[1][:-1]
    writer.writerows(np.column_stack((histogram_tbr_low_polarity[1][:-1]+width, histogram_tbr_low_polarity[0])))

tool.initialize_directory_or_file(file_name_save_high_polarity, columns)
with open(file_name_save_high_polarity, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    width = histogram_tbr_high_polarity[1][1:] - histogram_tbr_high_polarity[1][:-1]
    writer.writerows(np.column_stack((histogram_tbr_high_polarity[1][:-1]+width, histogram_tbr_high_polarity[0])))

tool.initialize_directory_or_file(file_name_save_low_neighbours, columns)
with open(file_name_save_low_neighbours, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    width = histogram_tbr_low_neighbours[1][1:] - histogram_tbr_low_neighbours[1][:-1]
    writer.writerows(np.column_stack((histogram_tbr_low_neighbours[1][:-1]+width, histogram_tbr_low_neighbours[0])))

tool.initialize_directory_or_file(file_name_save_high_neighbours, columns)
with open(file_name_save_high_neighbours, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    width = histogram_tbr_high_neighbours[1][1:] - histogram_tbr_high_neighbours[1][:-1]
    writer.writerows(np.column_stack((histogram_tbr_high_neighbours[1][:-1]+width, histogram_tbr_high_neighbours[0])))

# np.savetxt(file_name_save_low_polarity, tbr_low_polarity, delimiter=',', header='tbr_low_polarity', comments='')
# np.savetxt(file_name_save_high_polarity, tbr_high_polarity, delimiter=',', header='tbr_high_polarity', comments='')
# np.savetxt(file_name_save_low_polarity, tbr_low_neighbours, delimiter=',', header='tbr_low_polarity', comments='')
# np.savetxt(file_name_save_high_polarity, tbr_high_neighbours, delimiter=',', header='tbr_high_polarity', comments='')

# # Neighbours
# cond_low_polarity = df.loc[:, 'n_neighbors'] <= 3
# cond_high_polarity = df.loc[:, 'n_neighbors'] > 3
# tbr_neg = df.loc[cond_low_polarity, 'tbr'].values
# tbr_neg = tbr_neg[tbr_neg>0] * 2 / 60
# tbr_pos = df.loc[cond_high_polarity, 'tbr'].values
# tbr_pos = tbr_pos[tbr_pos>0] * 2 / 60
# %%
plt.hist(tbr_neg, bins=60)
plt.xlabel('Time between reversals (minutes)')
plt.ylabel('Count')
# %%
plt.hist(tbr_pos, bins=100)
# %%
import csv

file_name_save_low = path_folder+'2021_12_09_cover_area=0.6_tbr_low_neighbours.csv'
file_name_save_high = path_folder+'2021_12_09_cover_area=0.6_tbr_high_neighbours.csv'
tool.initialize_directory_or_file(file_name_save_low, 'tbr_low_neighbours')
tool.initialize_directory_or_file(file_name_save_high, 'tbr_high_neighbours')

np.savetxt(file_name_save_low, tbr_neg, delimiter=',', header='tbr_low_neighbours', comments='')
np.savetxt(file_name_save_high, tbr_pos, delimiter=',', header='tbr_high_neighbours', comments='')

# %%
path = "W:/jb/python/myxococcus_xanthus_simu2d/results/2023_08_24_big_simu_and_non_reversing/sample2/"
file_name = "coords__3330_bacts__tbf=1.8_secondes__space_size=195.csv"
df = pd.read_csv(path+file_name)

file_name_save = "coords__3330_bacts__tbf=3_minutes__space_size=195.csv"
cond_time = df.loc[:, 'frame'].values % 100 == 0
time = df.loc[cond_time, 'frame'].values / 60 * 1.8
df_new = df.loc[cond_time].copy()
df_new.loc[:, 'time_minutes'] = time
df_new.loc[:, 'frame'] = df_new.loc[:, 'frame'].values / 100
df_new.to_csv(path+file_name_save, header=True, index=False)
# %%
path = "W:/jb/python/myxococcus_xanthus_simu2d/results/2023_08_24_big_simu_and_non_reversing/sample3/"
file_name = "coords__9000_bacts__tbf=1.8_secondes__space_size=195.csv"
df = pd.read_csv(path+file_name)

file_name_save = "coords__9000_bacts__tbf=3_minutes__space_size=195.csv"
cond_time = df.loc[:, 'frame'].values % 100 == 0
time = df.loc[cond_time, 'frame'].values / 60 * 1.8
df_new = df.loc[cond_time].copy()
df_new.loc[:, 'time_minutes'] = time
df_new.loc[:, 'frame'] = df_new.loc[:, 'frame'].values / 100
df_new.to_csv(path+file_name_save, header=True, index=False)
# %%
path = "W:/jb/python/myxococcus_xanthus_simu2d/results/2023_08_24_big_simu_and_non_reversing/sample4/"
file_name = "coords__3330_bacts__tbf=1.8_secondes__space_size=195.csv"
df = pd.read_csv(path+file_name)

file_name_save = "coords__3330_bacts__tbf=3_minutes__space_size=195.csv"
cond_time = df.loc[:, 'frame'].values % 100 == 0
time = df.loc[cond_time, 'frame'].values / 60 * 1.8
df_new = df.loc[cond_time].copy()
df_new.loc[:, 'time_minutes'] = time
df_new.loc[:, 'frame'] = df_new.loc[:, 'frame'].values / 100
df_new.to_csv(path+file_name_save, header=True, index=False)














# %%
########################### PLOT SIMULATION ###########################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
import gc
import os
import tools
tool = tools.Tools()
# %%
path_folder = "W:/jb/python/myxococcus_xanthus_simu2d/results/2023_08_24_big_simu_and_non_reversing/sample2/"
filename = "coords__3330_bacts__tbf=1.8_secondes__space_size=195.csv"
df = pd.read_csv(path_folder+filename)
# %%
column_x = tool.gen_string_numbered(n=10, str_name='x')
column_y = tool.gen_string_numbered(n=10, str_name='y')
frames = df.loc[:, 'frame'].unique()
# frames = [4000]
point_size = 0.5*25 / 195 * 32 ** 2
path_save = 'save_plots/im_seq_swarming_linear/'

for frame in tqdm(frames[::10]):
    cond_frame = df.loc[:, 'frame'] == frame

    fig = plt.figure(figsize=(32,32))
    ax = fig.add_subplot(111)
    # angle_color = np.tile(np.arctan2(self.dir.nodes_direction[1,0,:], self.dir.nodes_direction[0,0,:]),(self.par.n_nodes,1))
    plt.scatter(df.loc[cond_frame, column_x].values.flatten(), df.loc[cond_frame, column_y].values.flatten(), s=point_size, linewidths=1, c='k', edgecolor='lightgrey', alpha=0.6)#, cmap="Pastel2")
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, 195)
    ax.set_ylim(0, 195)
    ax.axis('off')
    fig.subplots_adjust(left=0,bottom=0,right=1,top=1)
    fig.subplots_adjust(wspace=0,hspace=0)
    # fig.savefig(path, bbox_inches='tight')
    fig.savefig(path_save+str(int(frame))+'.png', dpi=100)
    plt.close()
    gc.collect()

# %%
df.sort_values(by=['id', 'frame'], ignore_index=True, inplace=True)

# %%
track_id = df.loc[:, 'id'].values # Read the file
reversal = df.loc[:, 'reversal'].values
frames = df.loc[:, 'frame'].values
cond_rev = reversal == 1
tbr = frames[cond_rev]
tbr = (tbr[1:] - tbr[:-1]) * 1.8 / 60
cond_change_traj = track_id[cond_rev][1:] == track_id[cond_rev][:-1]
tbr = tbr[cond_change_traj]
# cond_change_traj = np.concatenate((cond_change_traj, np.array([False])))

# %%
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'cmr10'  # "Computer Modern Roman"
path_folder = "W:/jb/python/notebook/Myxo/tracking/save_traj/2021_12_09_cover_area=0.6/" # Put your own path
path_save = 'save_plots/' # Save in a specific folder
filename = 'tbr_swarming_simu.png'

mean_tbr = np.mean(tbr)
fontsize = 30 # Taille de la police
bins = 45 # Nombre de bins
fig, ax = plt.subplots(figsize=(8,6)) # You can change the size of the figure here
ax.hist(tbr, bins=bins, alpha=0.35, range=(0,15), density=True, histtype='bar', ec='black', color="limegreen", label="Mean TBR: "+str(round(mean_tbr,2))+" minutes") # You can change the colors of the histogram here with alpha and color
ax.set_xlabel("Time between reversals (min)", fontsize=fontsize) # Title of the x-axis
ax.set_ylabel("Density", fontsize=fontsize)# Title of the y-axis
ax.tick_params(labelsize=fontsize) # Put the size of the police for the axis numbers
ax.legend(fontsize=fontsize*0.6)
fig.savefig(path_save+filename, bbox_inches='tight', dpi=100)
plt.close()

# %%
import pickle
# Ouverture du fichier en mode binaire ('wb' pour write binary)
with open('save_python_object/tbr_simu_swarming.pkl', 'wb') as fichier:
    # Utilisation de pickle.dump() pour enregistrer le tableau dans le fichier
    pickle.dump(tbr, fichier)

# Ouverture du fichier en mode binaire ('rb' pour read binary)
with open('save_python_object/tbr_simu_swarming.pkl', 'rb') as fichier:
    # Utilisation de pickle.load() pour charger le tableau depuis le fichier
    tableau_charge = pickle.load(fichier)

# %%
# Load csv
import pandas as pd

path_folder = '/media/jb/DATA/a_thesis/data_for_paper/film_to_extract_polarity/high_density/'
file_name = 'data_dir_min_size_smoothed_um=1_um.csv'
df = pd.read_csv(path_folder+file_name)

# %%
cond_frame = df.loc[:, "frame"] < 3
df_new = df.loc[cond_frame, :]
df_new.to_csv(path_folder+'data_dir_min_size_smoothed_um=1_um_3_frames.csv', index=False)
# %%
import numpy as np

# Your 2D array
array_2d = np.array([[1, 2, 3, 1, 2, 4, 5, 6, 6],
                    [2, 3, 4, 5, 5, 6, 7, 8, 9]])

# Function to get a boolean array for the first occurrences in a given row
def first_occurrence(row):
    """
    Get a boolean array indicating the first occurrences of elements in a given row.

    Parameters:
    - row (numpy.ndarray): One-dimensional array representing a row in the 2D array.

    Returns:
    - numpy.ndarray: Boolean array of the same length as the input row, 
      where True indicates the first occurrence of each unique element.
    """
    # Get unique elements and their indices of the first occurrences
    unique_elements, first_occurrence_indices = np.unique(row, return_index=True)
    
    # Create a boolean array with False, then set True at indices of the first occurrences
    result = np.full_like(row, fill_value=False, dtype=bool)
    result[first_occurrence_indices] = True
    
    return result

# Apply the function to each row of the 2D array
result = np.apply_along_axis(first_occurrence, axis=1, arr=array_2d)

print(result)























# %%
# Tester la polarité exemple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import parameters
import tools
par = parameters.Parameters()
tool = tools.Tools()

# Windows
name_folder = "W:/jb/movie_analyse/100X/2023_01_27_DZ2_100X_scale=0.06um-px_tbf=2s/001_high_density/tracking/data/reversal_signal/"
# Linux
# name_folder = "/media/jb/DATA/a_thesis/data_for_paper/film_to_extract_polarity/high_density/reversal_signal/"
file_name = 'data_rev_sig_min_size_smoothed_um=0.5_um.csv'
df = pd.read_csv(name_folder+file_name)

# %%
cond_frame = df.loc[:, "frame"] == 0
x_start = df.loc[cond_frame, 'x5'].values
y_start = df.loc[cond_frame, 'y5'].values
angles = df.loc[cond_frame, "mean_angle"].values
# Igoshin
polarity = df.loc[cond_frame, 'mean_polarity_igoshin'].values
nb_neighbors = df.loc[cond_frame, 'n_neighbours_igoshin'].values
# Mean polarity
# polarity = df.loc[cond_frame, 'mean_polarity'].values
# nb_neighbors = df.loc[cond_frame, 'n_neighbours'].values

str_columns_x, str_columns_y = tool.gen_coord_str(n=par.n_nodes, xy=False)
x_nodes = df.loc[cond_frame, str_columns_x].values.flatten()
y_nodes = df.loc[cond_frame, str_columns_y].values.flatten()
x_0 = df.loc[cond_frame, 'x_main_pole'].values
y_0 = df.loc[cond_frame, 'y_main_pole'].values
polarity_repeat = np.repeat(polarity, par.n_nodes)
# Normalisation des données entre 0 et 1
norm = Normalize(vmin=-1, vmax=1)
norm_data = norm(polarity)

# Création de la colormap personnalisée
cmap = LinearSegmentedColormap.from_list('custom_cmap', [(0, 'blue'), (0.5, 'white'), (1, 'red')])

import matplotlib.pyplot as plt
import numpy as np

# Exemple de données (remplacez cela par vos propres données)
# x_start = np.array([1, 2, 3, 4])
# y_start = np.array([1, 2, 3, 4])
# angles = np.array([45, 30, 60, 90])
# polarity = np.array([1, -1, 1, -1])

# Longueur des flèches
longueur = 20

# Calcul des coordonnées des points d'arrêt des flèches
x_end = x_start + longueur * np.cos(angles)
y_end = y_start + longueur * np.sin(angles)

# plt.style.use('dark_background')
# Création de la figure POLARITY
fig, ax = plt.subplots(figsize=(32,32), dpi=150)
# Tracé des flèches avec la fonction quiver
# Tracé des points de départ avec la fonction scatter
# ax.scatter(x_start, y_start, c=polarity, s=50, cmap=cmap)
ax.scatter(x_nodes, y_nodes, c=polarity_repeat, s=10, cmap=cmap)
ax.quiver(x_start, y_start, x_end - x_start, y_end - y_start, angles='xy', scale_units='xy', scale=1, color='k', width=0.0005)
ax.scatter(x_0, y_0, c='violet', s=10, cmap=cmap)
# Ajout d'une barre de couleur pour représenter la colormap
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, orientation='vertical')
cbar.set_label('Color Map')
# Affichage du graphique
plt.show()

# Création de la figure NB_NEIGHBORS
bacteria_nb_neighbors_plots = 0
cond_neighhors = np.repeat(nb_neighbors == bacteria_nb_neighbors_plots, par.n_nodes)
fig, ax = plt.subplots(figsize=(32,32), dpi=150)
# Tracé des flèches avec la fonction quiver
# Tracé des points de départ avec la fonction scatter
# ax.scatter(x_start, y_start, c=polarity, s=50, cmap=cmap)
ax.scatter(x_nodes, y_nodes, c='lightblue', s=10, cmap=cmap)
ax.scatter(x_nodes[cond_neighhors], y_nodes[cond_neighhors], c='orange', s=10, cmap=cmap)
ax.quiver(x_start, y_start, x_end - x_start, y_end - y_start, angles='xy', scale_units='xy', scale=1, color='k', width=0.0005)
ax.scatter(x_0, y_0, c='violet', s=10, cmap=cmap)
# Ajout d'une barre de couleur pour représenter la colormap
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, orientation='vertical')
cbar.set_label('Color Map')
# Affichage du graphique
plt.show()








































# %%
import numpy as np
import reversal_signal
import parameters
par = parameters.Parameters()
sig = reversal_signal.ReversalSignal()

n_bact = 3
kn = 5
n_nodes = 2
array_same_bact = np.tile((np.ones((kn, n_bact)) * np.arange(n_bact)).T.astype(int), (n_nodes,1))
print(array_same_bact)
id_bact_reshape = array_same_bact.reshape((n_bact, int(n_nodes * kn)), order='F')
print(id_bact_reshape)

np.apply_along_axis(sig.first_occurrence, axis=1, arr=id_bact_reshape)