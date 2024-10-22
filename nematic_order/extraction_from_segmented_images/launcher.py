# %%
import parameters
import multiprocessing
import main

main_path = 'path'

params_list = [

    ## SAMPLE 1 rippling simulation 1
    {'path_folder': main_path + "fig4/nematic_order_simu_rippling/",
     'file_name': "simu_rippling_1__9000_bacts__t=500_minutes__tbf=600_secondes.csv",
     'scale': 1, # in µm/px
     'tbf': 10, # minutes
     'first_frame': 5,
     'n_nodes': 10,
     'space_size': 65*3,
     "dr": 5,
     "interval_frames": 1,
     "x_column_last": "x9",
     "y_column_last": "y9"},

      ## SAMPLE 2 rippling simulation 2
    {'path_folder': main_path + "fig4/nematic_order_simu_rippling/",
     'file_name': "simu_rippling_2__9000_bacts__t=500_minutes__tbf=600_secondes.csv",
     'scale': 1, # in µm/px
     'tbf': 10, # minutes
     'first_frame': 5,
     'n_nodes': 10,
     'space_size': 65*3,
     "dr": 5,
     "interval_frames": 1,
     "x_column_last": "x9",
     "y_column_last": "y9"},

      ## SAMPLE 3 rippling simulation 3
    {'path_folder': main_path + "fig4/nematic_order_simu_rippling/",
     'file_name': "simu_rippling_3__9000_bacts__t=500_minutes__tbf=600_secondes.csv",
     'scale': 1, # in µm/px
     'tbf': 10, # minutes
     'first_frame': 5,
     'n_nodes': 10,
     'space_size': 65*3,
     "dr": 5,
     "interval_frames": 1,
     "x_column_last": "x9",
     "y_column_last": "y9"},

    ## SAMPLE 4 swarming simulation 1
    {'path_folder': main_path + "fig5/nematic_order_simu_swarming/",
     'file_name': "simu_swarming_1__3330_bacts__t=500_minutes__tbf=600_secondes.csv",
     'scale': 1, # in µm/px
     'tbf': 10, # minutes
     'first_frame': 5,
     'n_nodes': 10,
     'space_size': 65*3,
     "dr": 5,
     "interval_frames": 1,
     "x_column_last": "x9",
     "y_column_last": "y9"},

      ## SAMPLE 5 swarming simulation 2
    {'path_folder': main_path + "fig5/nematic_order_simu_swarming/",
     'file_name': "simu_swarming_2__3330_bacts__t=500_minutes__tbf=600_secondes.csv",
     'scale': 1, # in µm/px
     'tbf': 10, # minutes
     'first_frame': 5,
     'n_nodes': 10,
     'space_size': 65*3,
     "dr": 5,
     "interval_frames": 1,
     "x_column_last": "x9",
     "y_column_last": "y9"},

      ## SAMPLE 6 swarming simulation 3
    {'path_folder': main_path + "fig5/nematic_order_simu_swarming/",
     'file_name': "simu_swarming_3__3330_bacts__t=500_minutes__tbf=600_secondes.csv",
     'scale': 1, # in µm/px
     'tbf': 10, # minutes
     'first_frame': 5,
     'n_nodes': 10,
     'space_size': 65*3,
     "dr": 5,
     "interval_frames": 1,
     "x_column_last": "x9",
     "y_column_last": "y9"},

    
     ## SAMPLE 7 rippling experiment 1
    {'path_folder': main_path + "fig1/",
     'file_name': "file_tracking_rippling_movie_1.csv",
     'scale': 1,
     'tbf': 2/60, # minutes
     'n_nodes': 11,
     'space_size': 3200*0.0646028, # µm
     "dr": 5, # µm
     "interval_frames": 120},

    ## SAMPLE 8 rippling experiment 2
    {'path_folder': main_path + "fig1/",
    'file_name': "file_tracking_rippling_movie_2.csv",
    'scale': 1,
    'tbf': 2/60, # minutes
    'n_nodes': 11,
    'space_size': 3200*0.0646028, # µm
    "dr": 5, # µm
    "interval_frames": 120},

    ## SAMPLE 9 rippling experiment 3
     {'path_folder': main_path + "fig1/",
     'file_name': "file_tracking_rippling_movie_3.csv",
     'scale': 1,
     'tbf': 2/60, # minutes
     'n_nodes': 11,
     'space_size': 3200*0.0646028, # µm
     "dr": 5, # µm
     "interval_frames": 120},

    ## SAMPLE 10 swarming experiment 1
    {'path_folder': main_path + "fig1/",
     'file_name': "file_tracking_swarming_movie_1.csv",
     'scale': 1,
     'tbf': 2/60, # minutes
     'n_nodes': 11,
     'space_size': 3200*0.0646028, # µm
     "dr": 5, # µm
     "interval_frames": 120},

     ## SAMPLE 11 swarming experiment 2
    {'path_folder': main_path + "fig1/",
     'file_name': "file_tracking_swarming_movie_2.csv",
     'scale': 1,
     'tbf': 2/60, # minutes
     'n_nodes': 11,
     'space_size': 3200*0.0646028, # µm
     "dr": 5, # µm
     "interval_frames": 120},

     ## SAMPLE 12 swarming experiment 3
    {'path_folder': main_path + "fig1/",
     'file_name': "file_tracking_swarming_movie_3.csv",
     'scale': 1,
     'tbf': 2/60, # minutes
     'n_nodes': 11,
     'space_size': 3200*0.0646028, # µm
     "dr": 5, # µm
     "interval_frames": 120},
    
]

sample_name = ['nematic_order_simu_rippling_1.csv', 'nematic_order_simu_rippling_2.csv', 'nematic_order_simu_rippling_3.csv', 
               'nematic_order_simu_swarming_1.csv', 'nematic_order_simu_swarming_2.csv', 'nematic_order_simu_swarming_3.csv',
               'nematic_order_exp_rippling_1.csv', 'nematic_order_exp_rippling_2.csv', 'nematic_order_exp_rippling_3.csv',
               'nematic_order_exp_swarming_1.csv', 'nematic_order_exp_swarming_2.csv', 'nematic_order_exp_swarming_3.csv']

# If the previous line doesn't work please split the computation into three 
# and uncomment one by one the line below

# sample_name = ['nematic_order_simu_rippling_1.csv', 'nematic_order_simu_rippling_2.csv', 'nematic_order_simu_rippling_3.csv', 
#                'nematic_order_simu_swarming_1.csv', 'nematic_order_simu_swarming_2.csv', 'nematic_order_simu_swarming_3.csv']

# sample_name = ['nematic_order_exp_rippling_1.csv', 'nematic_order_exp_rippling_2.csv', 'nematic_order_exp_rippling_3.csv']

# sample_name = ['nematic_order_exp_swarming_1.csv', 'nematic_order_exp_swarming_2.csv', 'nematic_order_exp_swarming_3.csv']

# Fonction pour chaque simulation
def simulate(params, sample):
    par = parameters.Parameters()
    for key, value in params.items():
        setattr(par, key, value)
    ma = main.Main(inst_par=par, sample=sample)
    ma.start()


if __name__ == '__main__':
    # Création et lancement des processus pour chaque simulation
    processes = []
    for i, params in enumerate(params_list):
        sample = sample_name[i]
        process = multiprocessing.Process(target=simulate, args=(params, sample))
        processes.append(process)
        process.start()

    # Attente de la fin de tous les processus
    for process in processes:
        process.join()





# %%
# Plot the results
import pandas as pd
import parameters
import plot
import tools

par = parameters.Parameters()
plo = plot.Plot(par)
tool = tools.Tools()

color_rippling = tool.get_rgba_color(color_name='royalblue', alpha=0.4)
color_swarming = tool.get_rgba_color(color_name='limegreen', alpha=0.4)

df_rippling_1 = pd.read_csv('results/dataset/2023_03_09_rippling_no_multi_layer.csv')
df_rippling_2 = pd.read_csv('results/dataset/2022_02_25_rippling_multi_layer_000.csv')
df_rippling_3 = pd.read_csv('results/dataset/2022_02_25_rippling_multi_layer_001.csv')
liste_dataframes = [df_rippling_1, df_rippling_2, df_rippling_3]
df_rippling = pd.concat(liste_dataframes, ignore_index=True)

df_swarming_1 = pd.read_csv('results/dataset/2023_01_27_swarming_high_density_no_static.csv')
df_swarming_2 = pd.read_csv('results/dataset/2023_07_01_DZ2_swarming_static.csv')
df_swarming_3 = pd.read_csv('results/dataset/2023_07_01_SgmX_swarming_static.csv')
liste_dataframes = [df_swarming_1, df_swarming_2, df_swarming_3]
df_swarming = pd.concat(liste_dataframes, ignore_index=True)

df_rippling_simu_linear = pd.read_csv('results/dataset/simu_rippling_linear.csv')
df_swarming_simu_linear = pd.read_csv('results/dataset/simu_swarming_linear.csv')
df_rippling_simu_sigmoid = pd.read_csv('results/dataset/simu_rippling_sigmoid.csv')
df_swarming_simu_sigmoid = pd.read_csv('results/dataset/simu_swarming_sigmoid.csv')

legend = False
# Rippling/swarming experiment
list_df = [df_rippling, df_swarming]
list_names = ['Rippling', 'Swarming']
list_colors = [color_rippling, color_swarming]
list_linestyles = ['-', '-']
list_markers = ['o', 's']
name_save = 'rippling_swarming_experiment'
plo.plot_nematic_over_distance(list_df, list_names, list_colors, list_linestyles, list_markers, name_save, legend)
# Rippling experiment rippling simulation linear
list_df = [df_rippling, df_rippling_simu_linear]
list_colors = [color_rippling, 'k']
list_linestyles = ['-', '-']
list_names = ['Experimental', 'Simulation']
name_save = 'rippling_experiment_rippling_simu_linear'
plo.plot_nematic_over_distance(list_df, list_names, list_colors, list_linestyles, list_markers, name_save, legend)
# Rippling experiment rippling simulation sigmoid
list_df = [df_rippling, df_rippling_simu_sigmoid]
list_colors = [color_rippling, 'k']
list_linestyles = ['-', '-']
list_names = ['Experimental', 'Simulation']
name_save = 'rippling_experiment_rippling_simu_sigmoid'
plo.plot_nematic_over_distance(list_df, list_names, list_colors, list_linestyles, list_markers, name_save, legend)
# Swarming experiment rippling simulation linear
list_df = [df_swarming, df_swarming_simu_linear]
list_colors = [color_swarming, 'k']
list_linestyles = ['-', '-']
list_names = ['Experimental', 'Simulation']
name_save = 'swarming_experiment_rippling_simu_linear'
plo.plot_nematic_over_distance(list_df, list_names, list_colors, list_linestyles, list_markers, name_save, legend)
# Swarming experiment rippling simulation sigmoid
list_df = [df_swarming, df_swarming_simu_sigmoid]
list_colors = [color_swarming, 'k']
list_linestyles = ['-', '-']
list_names = ['Experimental', 'Simulation']
name_save = 'swarming_experiment_rippling_simu_sigmoid'
plo.plot_nematic_over_distance(list_df, list_names, list_colors, list_linestyles, list_markers, name_save, legend)





# Rippling and swarming files
# f1 = pd.read_csv('results/dataset/sample1_correction_11_nodes_trackmate_spots_swarming_static_100X_tbf=2s_006_um_px_.csv')
# f2 = pd.read_csv('results/dataset/sample2_correction_11_nodes_trackmate_kalman_filter_gap=4_spots_swarming_static_100X_tbf=2s_006_um_px_1_900_frames.csv')
# f3 = pd.read_csv('results/dataset/sample3_correction_11_nodes_trackmate_spots_2023_01_27_DZ2_100X_scale=0.06um-px_tbf=2s_kalman_frame_gap_4frames.csv')
# f4 = pd.read_csv('results/dataset/sample1_correction_11_nodes_trackmate_spots_tbf=2s_scale=006_um_kalman_frame_gap_3frames_0_900.csv')
# list_df = [f4, f1, f2, f3]
# list_names = ['Rippling', 'Swarming', 'Swarming', 'Swarming']
# list_colors = [color_rippling, color_swarming, color_swarming, color_swarming]
# list_linestyles = ['-', '-', '-', '-']
# list_markers = ['o', 's', 's', 's']

# Simulation files
# f1 = 'results/dataset/sample1_coords_1000_bacts_tbf=2_secondes.csv'
# f2 = 'results/dataset/sample2_coords_300_bacts_tbf=2_secondes.csv'
# list_paths = [f1, f2]
# list_names = ['Rippling', 'Swarming']
# list_colors = ['blue', 'red']
# list_linestyles = ['-', '-']
# list_markers = ['', '']

# plo.plot_nematic_over_distance(list_df, list_names, list_colors, list_linestyles, list_markers)

# %%
# import tools
# import matplotlib.pyplot as plt
# import pandas as pd
# from scipy.stats import binned_statistic_2d
# import numpy as np
# import nematic_order
# import parameters
# par = parameters.Parameters()
# nem = nematic_order.NematicOrder(par)
# tool = tools.Tools()


# path_folder = "W:/jb/python/myxococcus_xanthus_simu2d/sample1/"
# file_name = "coords_1000_bacts_tbf=2_secondes.csv"
# df = pd.read_csv(path_folder+file_name)
# cond_frame = df.loc[:, 'frame'] == 3000
# df_tmp = df.loc[cond_frame, :]

# # %%
# columns_x = tool.gen_string_numbered(n=10, str_name='x')
# columns_y = tool.gen_string_numbered(n=10, str_name='y')
# x = df_tmp.loc[:, columns_x].values
# y = df_tmp.loc[:, columns_y].values
# angles = np.arctan2(y[:, 1:]-y[:, :-1], x[:, 1:]-x[:, :-1]).flatten()
# angles[angles < 0] += np.pi
# x = x[:, :-1].flatten()
# y = y[:, :-1].flatten()
# orientation_map, __, __, __ = binned_statistic_2d(x=x, y=y, values=angles, statistic=nem.mean_angle, bins=12, range=[[0, 65], [0, 65]])

# plt.figure(figsize=(10,10))
# plt.imshow(orientation_map, cmap='bwr')
# plt.colorbar()