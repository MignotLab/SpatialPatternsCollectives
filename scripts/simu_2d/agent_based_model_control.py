# %%
import sys, os
import multiprocessing

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from simulation.agent_based_model_myxo.main import Main
from simulation.agent_based_model_myxo.parameters import Parameters

"""
'reversal_type could be: (refractory_period_type, reversal_rate_type), "threshold_frustration", "periodic" or "off"; default is ("linear", "bilinear")

'reversal_rate_type could be: "bilinear", "bilinear_smooth", "linear", "sigmoidal", "exponential" or "constant"; default is "bilinear"

'refractory_period_type could be: "linear", "sigmoidal" or "constant"; default is "linear"

# EXAMPLE SWARMING
{   'generation_type':"square_random_orientation", 'n_bact':300, 'space_size':65,
    
    'repulsion_type':"repulsion", 'k_r':9e4,
    'attraction_type':"attraction_body", 'k_a':1e3,
    'alignment_type':"no_alignment",
    'eps_follower_type':"igoshin_eps_road_follower", 'plot_eps_grid':True,
    'reversal_type':("linear", "bilinear"),  's1':0.11, 's2':0.15, 'r_min':0, 'r_max':2, 'rp_min':0.5, 'rp_max':5,
    'alpha_sigmoid_rr':200, 'alpha_sigmoid_rp':100, 'alpha_bilinear_rr':10
    
},

# EXAMPLE RIPPLING
{   'generation_type':"square_alignment", 'n_bact':1000, 'space_size':65,
    
    'repulsion_type':"repulsion", 'k_r':9e4,
    'attraction_type':"attraction_body", 'k_a':1e3,
    'alignment_type':"global_alignment",
    'eps_follower_type':"no_eps", 'plot_eps_grid':True,
    'reversal_type':("linear", "bilinear"),  's1':0.11, 's2':0.15, 'r_min':0, 'r_max':2, 'rp_min':0.5, 'rp_max':5,
    'alpha_sigmoid_rr':200, 'alpha_sigmoid_rp':100, 'alpha_bilinear_rr':10
    
},

"""
# Number of cell in rippling 100X field of the paper : 6294
# Number of cell in swarming 100X field of the paper : 3305

T = 100 # minutes
# Liste des ensembles de paramètres pour chaque simulation
params_list = [
    # ## RIPPLING CONTROL 1
    # {'generation_type':"square_alignment", 'n_bact':1000, 'space_size':65, 'save_frequency_csv':2, 'param_point_size':0.4,
    #  'alignment_type':"global_alignment",
    #  'eps_follower_type':"no_eps",
    # },

    # ## RIPPLING CONTROL 2
    # {'generation_type':"square_alignment", 'n_bact':1000, 'space_size':65, 'save_frequency_csv':2, 'param_point_size':0.4,
    #  'alignment_type':"global_alignment",
    #  'eps_follower_type':"no_eps",
    # },

    # ## RIPPLING CONTROL 3
    # {'generation_type':"square_alignment", 'n_bact':1000, 'space_size':65, 'save_frequency_csv':2, 'param_point_size':0.4,
    #  'alignment_type':"global_alignment",
    #  'eps_follower_type':"no_eps",
    # },

    # ## SWARMING CONTROL 4
    # {'generation_type':"square_random_orientation", 'n_bact':370, 'space_size':65, 'save_frequency_csv':2, 'param_point_size':0.4,
    #  'alignment_type':"no_alignment",
    #  'eps_follower_type':"igoshin_eps_road_follower",
    #  'plot_ecm_grid':True,
    # },

    # ## SWARMING CONTROL 5
    # {'generation_type':"square_random_orientation", 'n_bact':370, 'space_size':65, 'save_frequency_csv':2, 'param_point_size':0.4,
    #  'alignment_type':"no_alignment",
    #  'eps_follower_type':"igoshin_eps_road_follower",
    #  'plot_ecm_grid':True,
    # },

    # ## SWARMING CONTROL 6
    # {'generation_type':"square_random_orientation", 'n_bact':370, 'space_size':65, 'save_frequency_csv':2, 'param_point_size':0.4,
    #  'alignment_type':"no_alignment",
    #  'eps_follower_type':"igoshin_eps_road_follower",
    #  'plot_ecm_grid':True,
    # },

    ## TEST ECM
    {'generation_type':"square_random_orientation", 'n_bact':50, 'space_size':50, 'save_frequency_image':1/6, 'param_point_size':0.4,
     'alignment_type':"no_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'plot_ecm_grid':True,
     'pili_length':1,
     'reversal_type':'off',
     'dpi_simu':30,
    },

    # Ajoutez d'autres ensembles de paramètres selon vos besoins
]


# Fonction pour chaque simulation
def simulate(params, sample):
    par = Parameters()
    for key, value in params.items():
        setattr(par, key, value)
    ma = Main(inst_par=par, sample=sample, T=T)
    ma.start()

if __name__ == '__main__':
    # Forcer l'utilisation de 'spawn' sur Linux pour que les simulations avec les même paramètre
    # ai un seed différent les une des autres
    multiprocessing.set_start_method('spawn')
    # Création et lancement des processus pour chaque simulation
    processes = []
    for i, params in enumerate(params_list):
        sample = 'output/agent_based_model_control/sample' + str(i+1)
        process = multiprocessing.Process(target=simulate, args=(params, sample))
        processes.append(process)
        process.start()

    # Attente de la fin de tous les processus
    for process in processes:
        process.join()

# %%
# import numpy as np
# import matplotlib.pyplot as plt

# # Définition d'une grille 3x3
# grid_size = 3
# num_bins = grid_size * grid_size  # 9 bins

# # Coordonnées des centres des bins
# x_centers = np.arange(0.5, grid_size, 1)
# y_centers = np.arange(0.5, grid_size, 1)

# # Génération des données avec positions explicites
# x_data = []
# y_data = []
# colors = []

# count = 1
# for i, x in enumerate(x_centers):
#     for j, y in enumerate(y_centers):
#         x_data.extend([x] * count)  # Centre X répété pour le nombre de points requis
#         y_data.extend([y] * count)  # Centre Y répété pour le nombre de points requis
#         colors.extend([count] * count)  # Couleur proportionnelle au nombre de points
#         count += 1

# x_data = np.array(x_data)
# y_data = np.array(y_data)
# colors = np.array(colors)

# # Création de l'histogramme 2D
# hist, x_edges, y_edges = np.histogram2d(x_data, y_data, bins=[grid_size, grid_size], range=[[0, grid_size], [0, grid_size]])
# # hist = hist.T[::-1]



# # HISTOGRAMME 2D
# plt.figure(figsize=(5, 5))
# img = plt.imshow(hist, cmap='Reds', extent=[0, grid_size, 0, grid_size])

# # Ajout des valeurs exactes sur la colorbar
# cbar = plt.colorbar(img, ticks=np.arange(1, num_bins + 1))
# cbar.set_label("Nombre de points par bin")

# # Ajout des labels et du titre
# plt.xlabel("Bin X")
# plt.ylabel("Bin Y")
# plt.title("Histogramme 2D (3x3) avec nombre croissant de points par bin")

# plt.show()




# # SCATTER
# colors = np.array(colors)

# # Création du scatter plot
# plt.figure(figsize=(5, 5))
# scatter = plt.scatter(x_data, y_data, c=colors, cmap='Reds',s=400)

# # Ajout d'une colorbar avec des valeurs explicites
# cbar = plt.colorbar(scatter, ticks=np.arange(1, num_bins + 1))
# cbar.set_label("Nombre de points par bin")

# # Ajout de la grille pour délimiter les bins
# plt.xticks(np.arange(0, grid_size + 1, 1))
# plt.yticks(np.arange(0, grid_size + 1, 1))
# plt.grid(True, linestyle='--', color='gray', alpha=0.5)

# # Ajout des labels et du titre
# plt.xlabel("Bin X")
# plt.ylabel("Bin Y")
# plt.title("Scatter plot (3x3) avec accumulation des points")

# plt.show()

# grad_y, grad_x = np.gradient(hist**2)
# # HISTOGRAMME 2D
# plt.figure(figsize=(5, 5))
# img = plt.imshow(hist**2, cmap='Reds', extent=[0, grid_size, 0, grid_size])

# # Ajout des valeurs exactes sur la colorbar
# cbar = plt.colorbar(img, ticks=np.arange(1, num_bins**2 + 1, 10))
# cbar.set_label("Nombre de points par bin")

# # Ajout des labels et du titre
# plt.xlabel("Bin X")
# plt.ylabel("Bin Y")
# plt.title("Histogramme 2D (3x3) avec nombre croissant de points par bin")

# plt.show()
# print(grad_x)
# print(print(grad_y))

# g_x = np.tile(np.arange(3), (3, 1))  # X coordinates in the EPS grid
# g_y = g_x.T  # Y coordinates in the EPS grid (transpose of `g_x`)
# print('g_x', g_x)
# print('g_y', g_y)

# x = 1
# y = 2
# print(hist[y, x])
# print(hist.T[x, y])
