# %%
import numpy as np
# Create a random number generator with a specific seed
rng = np.random.default_rng(seed=46)


def extract_local_squared(matrix, centers, radius):
    """
    Extract and sum values of local squared regions from a matrix.

    Parameters:
    -----------
    matrix (ndarray): The matrix from which local regions will be extracted.
    centers (ndarray): The center coordinates for each region in the matrix.
    radius (int): The size of the local square in pixel

    Returns:
    --------
    ndarray: The local squared regions centered around the given coordinates.
    """
    v = np.lib.stride_tricks.sliding_window_view(matrix, (radius * 2 + 1, radius * 2 + 1))
    results = v[centers[1, :], centers[0, :], :, :]
    return results

# # Exemple de matrice 2D
# matrix = np.array([[1, 2, 3],
#                    [4, 5, 6],
#                    [7, 8, 9]])

# # Inverser sur l'axe des y (haut devient bas)
# matrix[:, :].T.T

# random_matrix = np.random.randint(low=0, high=10, size=(9, 9))
random_matrix = rng.integers(low=0, high=10, size=(9, 9))

# Afficher la matrice
print(random_matrix)

bact_centers = np.array([[0, 4, 3, 4], 
                         [0, 5, 5, 3]])
bact_directions = np.array([[-1, 1, -1, 1],
                            [0, 1, 2, 1]])

ecm_prey_angle = np.zeros(len(bact_centers[0]))
print("bact_directions\n", bact_directions)
radius = 1
local_bin_values = extract_local_squared(random_matrix, bact_centers, radius)
print('local_bin_values\n', local_bin_values)
local_bin_value_flattened = local_bin_values.reshape(bact_centers.shape[1], (radius + 2)**2)
# print('local_bin_value_flattened\n', local_bin_value_flattened)


import numpy as np

def generate_directions_from_matrix_centers(n):
    """
    Generates an array of directions in radians and their associated unit vectors
    for an n x n matrix. The directions represent the angle from the center of 
    the matrix to each element in the matrix, flattened.

    Parameters:
    - n (int): Size of the matrix (must be odd to have a center).

    Returns:
    - tuple: A tuple containing:
        - list of angles in radians (flattened).
        - list of unit vectors [(ux, uy)] (flattened).
    """
    if n % 2 == 0:
        raise ValueError("Matrix size must be odd to have a center.")

    # Center coordinates of the matrix
    center = (n // 2, n // 2)

    # Initialize lists for angles and vectors
    angles = []
    unit_vectors = []

    # Generate the directions
    for i in range(n):
        for j in range(n):
            # Calculate the angle from the center to the current cell
            dy = center[0] - i  # Vertical difference
            dx = j - center[1]  # Horizontal difference
            angle = np.arctan2(dy, dx)  # Angle in radians
            angles.append(angle)

            # Calculate the unit vector (ux, uy) corresponding to the angle
            norm = np.sqrt(dx**2 + dy**2)  # Distance from center
            if norm != 0:
                ux, uy = dx / norm, dy / norm  # Normalize
            else:
                ux, uy = 0, 0  # Center has no direction
            unit_vectors.append((ux, uy))

    return np.array(angles), np.array(unit_vectors).T

# Example usage for a 3x3 matrix
size_kernel = 3
local_angles, local_directions = generate_directions_from_matrix_centers(size_kernel)
print("Angles (radians):", local_angles)
print("Unit vectors:", local_directions)
print("\n")


# %%
scalar_product = np.sum(local_directions[:, np.newaxis, :] * bact_directions[:, :, np.newaxis], axis=0)
cond_angle_view = scalar_product >= 0
# Set the central bin to False
cond_angle_view[:, int(size_kernel**2 / 2)] = False
# print('cond_angle_view\n', cond_angle_view)

min_ecm_prey_factor = 0.8
# Set the local bin that are not in the angle view to -1 to be sure to don't take them into account
local_bin_value_flattened[~cond_angle_view] = -1
cond_maxs_local_bin_value = local_bin_value_flattened > min_ecm_prey_factor * np.max(local_bin_value_flattened, axis=1)[:, np.newaxis]
print('cond_maxs_local_bin_value\n', cond_maxs_local_bin_value)


number_of_close_ecm_prey_bins_lvl = np.sum(cond_maxs_local_bin_value, axis=1)
# print('number_of_close_ecm_prey_bins_lvl ', number_of_close_ecm_prey_bins_lvl)
bact_angle = np.arctan2(bact_directions[1], bact_directions[0])
angle_diff_bins_cells = np.abs(local_angles[np.newaxis, :] - bact_angle[:, np.newaxis])
# Set a big angle for the bins that are not selctioned by cond_maxs_local_bin_value
angle_diff_bins_cells[~cond_maxs_local_bin_value] = np.pi

cond_maxs_local_bin_value_unique = np.zeros(cond_maxs_local_bin_value.shape, dtype=bool)
cond_maxs_local_bin_value_unique[number_of_close_ecm_prey_bins_lvl == 1, :] = cond_maxs_local_bin_value[number_of_close_ecm_prey_bins_lvl == 1, :]

# Adjust direction based on EPS gradient.
# In the case several section have the same value, choose the one closer than the actual cell direction.
# In the case two sections have the same lvl either the same distance compare to the actual cell direction,
# choose a random section
for i in range(2, int(size_kernel**2 / 2 + 2)):
    cond_i_bin_value_same_lvl = number_of_close_ecm_prey_bins_lvl == i
    if np.sum(cond_i_bin_value_same_lvl) > 0:
        tmp = angle_diff_bins_cells[cond_i_bin_value_same_lvl, :]
        idx_sector = np.argmin(tmp * 
                                (1 + np.random.uniform(low=-0.001, high=0.001, size=tmp.shape)), axis=1)
        tmp = cond_maxs_local_bin_value_unique[cond_i_bin_value_same_lvl, :].copy()
        tmp[np.arange(idx_sector.size), idx_sector] = True
        cond_maxs_local_bin_value_unique[cond_i_bin_value_same_lvl, :] = tmp

print('check all; should be 1', np.sum(cond_maxs_local_bin_value_unique, axis=1))

# Get the indices of the True values
rows, cols = np.where(cond_maxs_local_bin_value_unique)
# Extract the corresponding values from `local_angles`
ecm_prey_angle[:] = local_angles[cols]

print('ecm_prey_angle\n', ecm_prey_angle)
# a = np.arange(100)
# print(a)
# print(np.clip(a,0,1))

# %%
cond_i_bin_value_same_lvl = number_of_close_ecm_prey_bins_lvl == 2
tmp = angle_diff_bins_cells[cond_i_bin_value_same_lvl, :]
print(tmp)
print(tmp * (1 + np.random.uniform(low=-0.001, high=0.001, size=tmp.shape))[:, np.newaxis])
# import numpy as np
# import matplotlib.pyplot as plt

# import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from simulation.agent_based_model.parameters import Parameters
# from simulation.agent_based_model.bacteria_generation import GenerateBacteria
# par = Parameters()

# params_list = [
#     ## Myxo-coli control 1
#     {'generation_type':"square_random_orientation", 'n_bact':100, 'n_bact_prey':0, 'n_nodes':11, 'space_size':200, 
#      'save_frequency_csv':5/60, 'save_frequency_image':0.2, 'dpi':10, 'param_point_size':0.1, 'plot_ecm_grid':True,
#      'alignment_type':"no_alignment",
#      'dt':5/60/20,
#     #  'eps_follower_type':"igoshin_eps_road_follower_old",
#      'eps_follower_type':"follow_gradient",
#     #  'prey_follower_type':"follow_gradient",
#      'prey_follower_type':"no_prey_ecm",
#      'sigma_blur_eps_map':100,
#      'epsilon_eps':4,
#      'epsilon_prey':6,
#      'radius_prey_ecm_effect':(par.n_nodes * par.d_n_prey + 2 * par.d_n_prey) * 3,
#      'reversal_type':"off",
#      'random_movement':False,
#      'sigma_random':0.03,
#      'max_eps_value':5,
#      'deposit_amount':2,
#      'sigma_blur_eps_map':0,
#     },
# ]


# for key, value in params_list[0].items():
#     setattr(par, key, value)
# gen = GenerateBacteria(par)
# gen.generate_bacteria()

# # Paramètres
# np.random.seed(42)  # Pour des résultats reproductibles
# num_points = 20
# grid_size = 700     # Taille de la grille en nombre de bins
# edges_width = 10    # Largeur des bords ajoutée autour des données
# l = 200             # Taille de la zone de simulation
# point_size = 50
# eps_grid = np.zeros((grid_size, grid_size))

# # Générer des positions aléatoires de bactéries
# x_coords = gen.data[0, :, :]
# y_coords = gen.data[1, :, :]

# # Générer une grid avec np.histogram2d
# grid, x_edges, y_edges = np.histogram2d(
#     gen.data[0, -1, :],
#     gen.data[1, -1, :],
#     bins=grid_size,
#     range=[[-edges_width, l + edges_width], [-edges_width, l + edges_width]]
# )

# fig, ax = plt.subplots(figsize=(32, 32))
# # Afficher la grid avec plt.imshow
# eps_grid += 2 * grid
# ax.imshow(
#     eps_grid.T,  # Transposée pour aligner les axes x et y
#     extent=(-edges_width, l + edges_width, -edges_width, l + edges_width),
#     origin='lower',  # Pour que l'origine soit en bas à gauche
#     cmap='Reds'
# )

# # Ajouter des points avec plt.scatter
# ax.scatter(
#     x_coords,
#     y_coords,
#     s=point_size,
#     c='blue',
#     edgecolor='w',
#     alpha=0.5,
#     label='Bacteria positions'
# )

# ax.set_aspect('equal', adjustable='box')
# # plt.xlim(-edges_width, l+edges_width)
# # plt.ylim(-edges_width, l+edges_width)
# ax.axis('off')  # Remove axis lines and labels for a cleaner plot.
# # Adjust subplot layout for tight bounds.
# fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
# fig.subplots_adjust(wspace=0, hspace=0)

# # Afficher le plot
# plt.show()




# %%
import numpy as np
import scipy.signal
from scipy.ndimage import convolve
import matplotlib.pyplot as plt


def generate_chain_positions(start, direction, num_spheres, spacing):
    """
    Génère des positions pour une chaîne de sphères alignées dans une direction donnée.

    Parameters:
    - start : tuple (x, y) pour la position de départ.
    - direction : tuple (dx, dy) pour la direction de la chaîne (doit être normalisé).
    - num_spheres : nombre de sphères dans la chaîne.
    - spacing : espacement entre les centres des sphères.

    Returns:
    - positions : tableau de shape (num_spheres, 2) avec les positions (x, y) des sphères.
    """
    positions = np.zeros((num_spheres, 2), dtype=int)
    positions[0] = start
    for i in range(1, num_spheres):
        new_position = (int(start[0] + i * direction[0] * spacing), int(start[1] + i * direction[1] * spacing))
        positions[i] = new_position
    return positions


def create_grid_with_sphere_chains(chains, grid_size, radius, dtype):
    """
    Create a grid with chains of spheres placed at specified positions.

    Parameters:
    - chains (numpy.ndarray): Array of shape (2, num_nodes, num_chains) representing (x, y) coordinates 
                              of spheres in each chain.
    - grid_size (tuple): Tuple (height, width) representing the dimensions of the grid.
    - radius (int): Radius of the spheres in pixels.
    - dtype (data-type): Data type for the grid (e.g., np.uint8, np.float32).

    Returns:
    - grid (numpy.ndarray): A 2D array of size `grid_size` with spheres drawn at each position in each chain.
    """
    # Initialize an empty grid
    height, width = grid_size
    grid = np.zeros((height, width), dtype=dtype)

    # Create a circular mask once based on the radius
    y_mask, x_mask = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    mask = (x_mask**2 + y_mask**2) <= radius**2

    # Get all (x, y) coordinates from the chains array
    x_coords = chains[0].flatten()
    y_coords = chains[1].flatten()

    # Loop through each (x, y) position and place the mask on the grid
    for x, y in zip(x_coords, y_coords):
        x_start, x_end = max(x - radius, 0), min(x + radius + 1, width)
        y_start, y_end = max(y - radius, 0), min(y + radius + 1, height)

        # Calculate the appropriate slice of the mask
        mask_x_start = max(radius - x, 0)
        mask_x_end = mask_x_start + (x_end - x_start)
        mask_y_start = max(radius - y, 0)
        mask_y_end = mask_y_start + (y_end - y_start)

        # Apply the mask to the grid
        grid[y_start:y_end, x_start:x_end][mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]] = 1

    return grid


def generate_random_start_positions(num_chains, grid_size):
    """
    Génère des positions de départ aléatoires pour chaque chaîne.

    Parameters:
    - num_chains : nombre de chaînes à générer.
    - grid_size : taille de la grille (height, width).

    Returns:
    - start_positions : liste des positions de départ pour chaque chaîne.
    """
    height, width = grid_size
    start_positions = []
    for _ in range(num_chains):
        # Générer des positions aléatoires dans les limites de la grille
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        start_positions.append((x, y))
    return start_positions


def generate_random_directions(num_chains):
    """
    Génère des directions aléatoires pour chaque chaîne.

    Parameters:
    - num_chains : nombre de chaînes à générer.

    Returns:
    - directions : liste des directions pour chaque chaîne.
    """
    directions = []
    for _ in range(num_chains):
        # Générer une direction aléatoire dans le plan
        angle = np.random.uniform(0, 2 * np.pi)
        dx = np.cos(angle)
        dy = np.sin(angle)
        directions.append((dx, dy))
    return directions


def create_exponential_decay_kernel(size, sharpness):
    """
    Create a kernel with rapid decay as the distance from the center increases.

    Parameters:
    - size (int): Size of the kernel (must be an odd number).
    - sharpness (float): Controls the rate of decay (higher value = faster decay).

    Returns:
    - kernel (numpy.ndarray): A (size, size) array with values that decay exponentially from the center.
    """
    # Create a grid of coordinates for x and y centered around 0
    ax = np.linspace(-(size // 2), size // 2, size)
    x, y = np.meshgrid(ax, ax)
    
    # Calculate the Euclidean distance from the center
    distance = np.sqrt(x**2 + y**2)
    
    # Apply an exponential decay function with the specified sharpness
    kernel = np.exp(-sharpness * distance)

    # Normalize the kernel so that the sum of all elements equals 1
    return kernel / np.sum(kernel)


# Paramètres de génération des chaînes
num_chains = 2  # Nombre de chaînes à générer
grid_size = (700, 700)
num_nodes = 11
spacing = 2
radius = 30
dtype = np.float32
node_thickness = 5

# Générer des positions de départ aléatoires
start_positions = generate_random_start_positions(num_chains, grid_size)

# Générer des directions aléatoires pour chaque chaîne
directions = generate_random_directions(num_chains)

# Créer un tableau de forme (2, num_nodes, num_chains) pour stocker les positions des sphères
chains = np.zeros((2, num_nodes, num_chains), dtype=int)

# Générer les positions des chaînes et les stocker dans `chains`
for i, start in enumerate(start_positions):
    chains[:, :, i] = generate_chain_positions(start=start, direction=directions[i], num_spheres=num_nodes, spacing=spacing).T


# Créer une grille avec des chaînes de sphères
grid = create_grid_with_sphere_chains(chains, grid_size, radius, dtype)

# Créer un noyau avec une décroissance rapide
kernel_size = radius * 2 + 1  # Taille du noyau
sharpness = 0.001  # Facteur de contrôle de la rapidité de la décroissance
kernel = create_exponential_decay_kernel(size=kernel_size, sharpness=sharpness)

# Appliquer la convolution pour ajouter la diffusion
grid_blur = convolve(grid, kernel, mode='reflect', cval=0)

# Normaliser le résultat
grid_blur /= np.max(grid_blur)


# Affichage du résultat
plt.figure(figsize=(10, 10))
plt.imshow(grid, cmap='viridis')
plt.plot(chains[0], chains[1], color='k')
plt.title(f"Rapid Decay Applied to {num_chains} Random Chains")
plt.colorbar()
plt.show()
plt.figure(figsize=(10, 10))
plt.imshow(grid_blur, cmap='viridis')
plt.plot(chains[0], chains[1], color='k')
plt.title(f"Rapid Decay Applied to {num_chains} Random Chains")
plt.colorbar()
plt.show()

print('start_positions', start_positions)

# %%
import numpy as np
prob = 1 - np.exp(-0.2 * np.ones(11))  # Probability of stopping at prey contact
cond_stop = np.random.binomial(1, prob)
print(cond_stop)

# %%
import numpy as np

r_max = 3
dt = 0.005
a = 1 - np.exp(-r_max*dt)
b = (1 - np.exp(-3))*dt
print(a)
print(b)
print(a/b)

print(9e4 * 5/60/20)

# %%
import numpy as np
a = []
dt = 0.01
rate_stop_at_prey = 2
for i in range(int(1/dt)):
    cond_prey_neighbour = np.array([True, False])
    prob = 1 - np.exp(-rate_stop_at_prey * dt) * np.ones(len(cond_prey_neighbour))  # Probability of stopping at prey contact
    cond_stop = cond_prey_neighbour & np.random.binomial(1, prob).astype(bool)
    # print(np.random.binomial(1, prob).astype(bool))
    # print(cond_stop)
    a.append(np.sum(cond_stop))

print(np.sum(a))

# %%
import numpy as np

# Exemple de définition des variables
n_bact, n_nodes, kn = 5, 3, 4  # Exemple de valeurs
ind = np.random.randint(0, n_bact * n_nodes, (n_bact * n_nodes, kn))  # Array aléatoire
index_dead = np.random.choice(n_bact * n_nodes, size=5, replace=False)  # Exemples d'indices morts
print(ind)
print(index_dead)
# Création du masque booléen
bool_mask = np.isin(ind, index_dead)

# Résultat: bool_mask a la même shape que 'ind' et est True là où ind contient un élément de index_dead
print(bool_mask)  # Doit être (n_bact * n_nodes, kn)

# %%
index_dead = np.where(np.array([False, False, False, False, False]))[0]
np.isin(ind, index_dead)

# %%
a = 3
print(a)
b = 5
print(5)
b = a
print(b)
b = 10
print(b)
print(a)

# %%
import numpy as np
import matplotlib.pyplot as plt

x = np.array([[40.43583383,  1.88818645, 36.7077381 , 45.49858528, 21.02437569,
        34.03514934, 42.47574861, 40.15405726, 23.41418909, 41.17043448],
       [40.67679435,  1.88818645, 36.20837157, 45.85397395, 20.52411507,
        34.15569445, 42.34996106, 40.08339437, 23.45491182, 41.17636607],
       [41.04659473,  1.88818645, 35.70837862, 46.21100831, 20.0224145 ,
        34.27623955, 42.22417351, 40.01273147, 23.49563454, 41.18229767],
       [41.47402133,  1.88818645, 35.20786956, 46.56933848, 19.51956579,
        34.39678466, 42.09838596, 39.94206858, 23.53635727, 41.18822927],
       [41.91769355,  1.88818645, 34.70699749, 46.92850524, 19.01626749,
        34.51732976, 41.97259841, 39.87140568, 23.57708   , 41.19416086],
       [42.35596159,  1.88818645, 34.20594514, 47.28800075, 18.5133142 ,
        34.63787487, 41.84681086, 39.80074278, 23.61780273, 41.20009246],
       [42.78052523,  1.88818645, 33.70490954, 47.64732625, 18.01152039,
        34.75841997, 41.72102331, 39.73007989, 23.65852545, 41.20602405],
       [43.19259644,  1.88818645, 33.20408568, 48.00603697, 17.51164224,
        34.87896508, 41.59523576, 39.65941699, 23.69924818, 41.21195565],
       [43.59688733,  1.88818645, 32.7036505 , 48.36376987, 17.01432407,
        34.99951018, 41.46944821, 39.5887541 , 23.73997091, 41.21788724],
       [43.99716058,  1.88818646, 32.20374859, 48.72025537, 16.52006977,
        35.12005529, 41.34366066, 39.5180912 , 23.78069363, 41.22381884],
       [44.39556064,  1.88818646, 31.70448126, 49.07531792, 16.02918757,
        35.24060039, 41.21787311, 39.44742831, 23.82141636, 41.22975043]])

y = np.array([[46.12526322, 36.39712472,  9.43800747, 30.87132901,  6.92436778,
        10.3559105 , 46.91683374, 47.02734176,  5.63708466,  6.02775802],
       [45.68652522, 36.89972007,  9.47433756, 30.51594052,  6.97270006,
        10.44517907, 46.83511782, 46.89502867,  5.78145107,  5.87787535],
       [45.34834741, 37.40464275,  9.51073714, 30.15890637,  7.02956067,
        10.53444764, 46.75340189, 46.76271557,  5.92581748,  5.72799267],
       [45.08577056, 37.91139796,  9.54723611, 29.80057645,  7.09224752,
        10.62371621, 46.67168597, 46.63040248,  6.07018388,  5.57811   ],
       [44.85082814, 38.41933626,  9.58387694, 29.44140997,  7.16062065,
        10.71298478, 46.58997004, 46.49808938,  6.21455029,  5.42822732],
       [44.60548909, 38.92773947,  9.62070717, 29.08191478,  7.23473591,
        10.80225335, 46.50825412, 46.36577629,  6.3589167 ,  5.27834465],
       [44.33707141, 39.43590222,  9.65776461, 28.72258963,  7.31482087,
        10.89152192, 46.4265382 , 46.23346319,  6.50328311,  5.12846197],
       [44.05012856, 39.9431955 ,  9.69505871, 28.36387931,  7.40114053,
        10.98079049, 46.34482227, 46.1011501 ,  6.64764951,  4.9785793 ],
       [43.75290715, 40.4491059 ,  9.73255685, 28.00614685,  7.49390723,
        11.07005906, 46.26310635, 45.968837  ,  6.79201592,  4.82869662],
       [43.45112879, 40.9532522 ,  9.77018607, 27.64966183,  7.59323488,
        11.15932763, 46.18139043, 45.8365239 ,  6.93638233,  4.67881395],
       [43.14789451, 41.4553861 ,  9.80785492, 27.2945998 ,  7.69891287,
        11.2485962 , 46.0996745 , 45.70421081,  7.08074873,  4.52893127]])

plt.figure(figsize=(32,32))
plt.scatter(x, y, s=700, alpha=0.5)
plt.xlim(-5, 55)
plt.ylim(-5, 55)
plt.xticks(fontsize=100)
plt.yticks(fontsize=100)

cond_alive = np.array([[ True,  True,  True,  True,  True,  True,  True, False,  True,
         True],
       [ True,  True,  True,  True,  True,  True,  True, False,  True,
         True],
       [ True,  True,  True,  True,  True,  True,  True, False,  True,
         True],
       [ True,  True,  True,  True,  True,  True,  True, False,  True,
         True],
       [ True,  True,  True,  True,  True,  True,  True, False,  True,
         True],
       [ True,  True,  True,  True,  True,  True,  True, False,  True,
         True],
       [ True,  True,  True,  True,  True,  True,  True, False,  True,
         True],
       [ True,  True,  True,  True,  True,  True,  True, False,  True,
         True],
       [ True,  True,  True,  True,  True,  True,  True, False,  True,
         True],
       [ True,  True,  True,  True,  True,  True,  True, False,  True,
         True],
       [ True,  True,  True,  True,  True,  True,  True, False,  True,
         True]])

plt.figure(figsize=(32,32))
plt.scatter(x[cond_alive], y[cond_alive], s=700, alpha=0.5)
plt.xlim(-5, 55)
plt.ylim(-5, 55)
plt.xticks(fontsize=100)
plt.yticks(fontsize=100)

# %%
import numpy as np
np.random.seed(42)  # Fixe la graine

matrix = np.random.randint(0, 9, size=(14, 14))
print(matrix)

def extract_local_squared(matrix, centers, radius):
    """
    Extract and sum values of local squared regions from a matrix.

    Parameters:
    -----------
    matrix (ndarray): The matrix from which local regions will be extracted.
    centers (ndarray): The center coordinates for each region in the matrix.
    radius (int): The half size of the local square in pixel

    Returns:
    --------
    ndarray: The local squared regions centered around the given coordinates.
    """
    v = np.lib.stride_tricks.sliding_window_view(matrix, (radius * 2 - 1, radius * 2 - 1))
    results = v[centers[1, :], centers[0, :], :, :]

    return results

edge = 5
r = 3
centers_x = np.array([0, 1, 2, 3, 4]) + edge
centers_y = np.array([0, 3, 2, 1, 0]) + edge
centers = np.array((centers_x, centers_y))

center_id = 0
local_eps_grid_old = matrix[centers_x[center_id] - r : centers_x[center_id] + r - 1, centers_y[center_id] - r : centers_y[center_id] + r - 1].T
print(local_eps_grid_old)

local_eps_grid = extract_local_squared(matrix.T, centers - r, r)
print(local_eps_grid[center_id])