# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
from skimage import io
from nd2reader import ND2Reader
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
import os

class TypeError(Exception):
    def __init__(self):
        super().__init__('cond_area must be None or a 2D numpy array with the same shape as the orientation parameter')


_color_rippling = 'mediumturquoise'
_color_swarming = 'darkorange'
_alpha = 0.7
_size_height_figure = 7
_figsize = (_size_height_figure, _size_height_figure-1)
_dpi = 300
_fontsize = 30
_fontsize_ticks = _fontsize / 1.5


def graner_tool(weights, nb_orientation):
	"""
	Graner tool to compute mean orientation
	
	"""
	weights_tmp = np.concatenate((weights, weights))
	orientations_arr = np.arange(2 * nb_orientation)
	orientation_bin_midpoints = 2 * np.pi * (orientations_arr + 0.5) / (2 * nb_orientation)

	x = weights_tmp * np.cos(orientation_bin_midpoints)
	y = weights_tmp * np.sin(orientation_bin_midpoints)

	mat = np.zeros((2,2))
	mat[0, 0] = np.sum(x*x)
	mat[0, 1] = np.sum(x*y)
	mat[1, 0] = np.sum(x*y)
	mat[1, 1] = np.sum(y*y)

	e, v = np.linalg.eig(mat)

	index = np.argmax(e)
	diff_e = np.abs(e[0] - e[1])

	return e[index], v[index], diff_e

def extract_nematic_orientation(image, nb_orientation, bin_size, cells_per_block, channel_axis=None, plot=False):
	"""
	Extract the nematic orientations of an image using skimage.feature.hog
	 
	"""
	# print('COMPUTE HOG IMAGE FROM THE INPUT IMAGE')
	fd, hog_image = hog(image,
						orientations=nb_orientation,
						pixels_per_cell=(bin_size[0], bin_size[1]),
						cells_per_block=cells_per_block,
						visualize=True,
						transform_sqrt=True,
						feature_vector=False,
						channel_axis=channel_axis
						)
	
	hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 3200))
	if plot:
		print('PLOT HOG IMAGE')
		# Vizualize hog_image_rescaled 
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(32, 16), sharex=True, sharey=True)
		ax1.axis('off')
		ax1.imshow(image, cmap=plt.cm.gray)
		ax1.set_title('Input image')
		# Rescale histogram for better display
		ax2.axis('off')
		im = ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray, vmin=0, vmax=np.max(hog_image_rescaled)/nb_orientation, aspect='auto')
		ax2.set_title('Histogram of Oriented Gradients')
		# Division de l'axe pour ajouter la colorbar
		divider = make_axes_locatable(ax2)
		cax = divider.append_axes("right", size="5%", pad=0.1)
		cbar = plt.colorbar(im, cax=cax)
		plt.show()


	# print('COMPUTE ORIENTATION FROM THE HOG IMAGE')
	# Dimensions de fd
	n_cells_y, n_cells_x, n_blocks_y, n_blocks_x, n_orientations = fd.shape
	# Initialiser un tableau pour stocker les orientations moyennes
	weighted_sum = np.zeros((2, n_cells_y, n_cells_x))
	hist_all = np.zeros((n_cells_y, n_cells_x, nb_orientation))
	# Calculer l'orientation moyenne pour chaque cellule
	for i in range(n_cells_y):
		for j in range(n_cells_x):
			# Obtenir les histogrammes d'orientations pour la cellule (i, j)
			hist = fd[i, j, 0, 0, :]
			hist_all[i, j, :] = hist

			e_max, v_max, diff_e = graner_tool(weights=hist, nb_orientation=nb_orientation)
			if diff_e > 1e-1:
				weighted_sum[0, i, j] = e_max * v_max[0]
				weighted_sum[1, i, j] = e_max * v_max[1]
			else:
				weighted_sum[0, i, j] = 0
				weighted_sum[1, i, j] = 0
			
	return hog_image_rescaled, weighted_sum, hist_all

def draw_segments_in_bin(i, j, bin_size, ax, weighted_sum):
	# Calculer les coordonnées du bin
	x_start = i * bin_size[1]
	x_end = (i + 1) * bin_size[1]
	y_start = j * bin_size[0]
	y_end = (j + 1) * bin_size[0]
	# Calculer le centre du bin
	x_center = x_start + bin_size[1] / 2
	y_center = y_start + bin_size[0] / 2
	# Définir la longueur du segment
	segment_length = min(bin_size) / 2  # Longueur totale du segment
	# Calculer les coordonnées des extrémités du segment
	(segment_length / 2)
	x1 = x_center + (segment_length / 2) * weighted_sum[0, i, j]
	y1 = y_center + (segment_length / 2) * weighted_sum[1, i, j]
	x2 = x_center - (segment_length / 2) * weighted_sum[0, i, j]
	y2 = y_center - (segment_length / 2) * weighted_sum[1, i, j]
	norm = np.sqrt(weighted_sum[0, i, j]**2 + weighted_sum[1, i, j]**2)
	if norm > 1e-2:
		x1_norm = x_center + (segment_length / 2) * (weighted_sum[0, i, j] / norm)
		y1_norm = y_center + (segment_length / 2) * (weighted_sum[1, i, j] / norm)
		x2_norm = x_center - (segment_length / 2) * (weighted_sum[0, i, j] / norm)
		y2_norm = y_center - (segment_length / 2) * (weighted_sum[1, i, j] / norm)
	else:
		x2 = x1
		y2 = y1
		x1_norm = x1
		y1_norm = y1
		x2_norm = x2
		y2_norm = y2
	# Dessiner le segment
	ax.plot([y1_norm, y2_norm], [x1_norm, x2_norm], color='grey', linewidth=2)
	ax.plot([y1, y2], [x1, x2], color='white', linewidth=2)
	# Dessiner un rectangle pour visualiser le bin
	rect = plt.Rectangle((x_start, y_start), bin_size[1], bin_size[0], 
						 linewidth=0.2, edgecolor='r', facecolor='none')
	ax.add_patch(rect)

def condition_area(image, coords, nb_bins, bin_size, weighted_sum, exp_id,  plot=False):
	"""
	Condition area between two lines
	
	"""
	x1 = coords.loc[0, 'x0']
	y1 = coords.loc[0, 'y0']
	x2 = coords.loc[1, 'x0']
	y2 = coords.loc[1, 'y0']
	x3 = coords.loc[0, 'x1']
	y3 = coords.loc[0, 'y1']
	x4 = coords.loc[1, 'x1']
	y4 = coords.loc[1, 'y1']
	# Calcul des pentes et ordonnées à l'origine
	a1 = (y2 - y1) / (x2 - x1)
	b1 = y1 - a1 * x1
	a2 = (y4 - y3) / (x4 - x3)
	b2 = y3 - a2 * x3
	delta_x = (image.shape[1] / nb_bins[1]) / 2
	delta_y = (image.shape[0] / nb_bins[0]) / 2
	x_coords = np.linspace(delta_x, image.shape[1] - delta_x, nb_bins[1])
	y_coords = np.linspace(delta_y, image.shape[0] - delta_y, nb_bins[0])
	x_grid, y_grid = np.meshgrid(x_coords, y_coords, indexing='ij')
	if exp_id == 1:
		cond_bins_between_lines_1 = (y_grid < (a1 * x_grid + b1)) & (y_grid > (a2 * x_grid + b2))
		cond_bins_between_lines_2 = y_grid < (a2 * x_grid + b2)
	if exp_id == 2:
		cond_bins_between_lines_1 = (y_grid > (a1 * x_grid + b1)) & (y_grid < (a2 * x_grid + b2))
		cond_bins_between_lines_2 = y_grid > (a2 * x_grid + b2)
	if exp_id == 3:
		cond_bins_between_lines_1 = (y_grid > (a1 * x_grid + b1)) & (y_grid < (a2 * x_grid + b2))
		cond_bins_between_lines_2 = y_grid > (a2 * x_grid + b2)

	if plot:
		x_between_line, y_between_line = x_grid[cond_bins_between_lines_1].flatten(), y_grid[cond_bins_between_lines_1].flatten()
		# Tracer l'image
		fig, ax = plt.subplots(figsize=(_size_height_figure, _size_height_figure))
		plt.imshow(image, cmap=plt.cm.gray)
		# Calculer les limites des droites pour ne pas dépasser les bords de l'image
		x_min, x_max = 0, image.shape[1] - 1
		y_min, y_max = 0, image.shape[0] - 1
		# Tracer les droites dans les limites de l'image
		x_values = np.linspace(x_min, x_max, 100)
		y_values1 = a1 * x_values + b1
		y_values2 = a2 * x_values + b2
		# S'assurer que les points des droites sont dans les limites de l'image
		valid_idx1 = (y_values1 >= y_min) & (y_values1 <= y_max)
		valid_idx2 = (y_values2 >= y_min) & (y_values2 <= y_max)
		plt.plot(x_values[valid_idx1], y_values1[valid_idx1], label='Droite 1', color='blue')
		plt.plot(x_values[valid_idx2], y_values2[valid_idx2], label='Droite 2', color='red')
		# Mettre en évidence les bins entre les deux droites
		plt.scatter(x_between_line, y_between_line, color='green', s=20, label='Bins entre les droites (x0, y0)', alpha=0.5)
		
		# # Définir les paramètres des bins
		# rows, cols = image.shape
		# num_bins_x = cols // bin_size[1]
		# num_bins_y = rows // bin_size[0]
		# vmin = 0
		# tasks = [(i, j, bin_size, ax, weighted_sum) for i in range(num_bins_y) for j in range(num_bins_x)]
		# # Utiliser une boucle simple avec une barre de progression
		# for task in tqdm(tasks, desc="Processing Bins"):
		# 	draw_segments_in_bin(*task)

		plt.ylim(0, x_max)
		plt.ylim(y_max, 0)
		plt.axis('off')
		fig.subplots_adjust(left=0,bottom=0,right=1,top=1)
		fig.subplots_adjust(wspace=0,hspace=0)
		ax.legend(fontsize=_fontsize_ticks)
		plt.show()

		plt.figure()
		plt.imshow(cond_bins_between_lines_1.T)
		plt.show()

		plt.figure()
		plt.imshow(cond_bins_between_lines_2.T)
		plt.show()

	return cond_bins_between_lines_1.T, cond_bins_between_lines_2.T

def compute_distance_between_bins(size_x, size_y, xmax, ymax, cond_area=None):
	"""
	Compute the distance between each bin in a grid.

	Parameters
	----------
	size_x : int
		Number of columns in the grid.
	size_y : int
		Number of rows in the grid.
	scale : float
		Conversion factor for the distance between two bins in micrometers (µm).

	Returns
	-------
	distances : ndarray
		Array of shape (size_x * size_y, size_x, size_y) containing the distances 
		between each pair of bins in the grid.
	"""
	# Generate the grid coordinates
	x_coords = np.linspace(0, xmax, size_x)
	y_coords = np.linspace(0, ymax, size_y)
	
	xv, yv = np.meshgrid(x_coords, y_coords, indexing='ij')
	# Apply conditional area if provided
	if cond_area is not None:
		if not isinstance(cond_area, np.ndarray) or cond_area.shape != (size_x, size_y):
			raise TypeError()
		else:
			xv, yv = xv[cond_area], yv[cond_area]
	
	# Flatten the coordinate arrays
	xv_flat = xv.flatten()
	yv_flat = yv.flatten()
	
	# Create arrays to store all distances
	distances = np.tile(np.zeros(xv_flat.shape), (xv_flat.shape[0], 1))
	
	# Compute the distances using broadcasting
	for idx, (x1, y1) in enumerate(zip(xv_flat, yv_flat)):
		dx = x1 - xv_flat
		dy = y1 - yv_flat
		distances[idx] = np.sqrt(dx**2 + dy**2)
	
	return distances

def nematic_order(theta_i, theta_n):
    """
    Compute the nematic order for an object i with its neighbors n using the cos^2(delta) formula.

    Parameters
    ----------
    theta_i : float
        Orientation angle of the object i in radians.
    theta_n : ndarray of float
        Array of orientation angles of the neighboring objects in radians.

    Returns
    -------
    nematic_order : float
        The nematic order parameter for the object i with its neighbors n.
    
    Explanation
    -----------
    The nematic order parameter is a measure of the degree of alignment between the orientation
    of an object and its neighbors. It is computed using the formula:

        S = <2*cos(delta)^2 - 1>

    where:
        - S is the nematic order parameter.
        - delta = theta_i - theta_n is the angle difference between the object and its neighbors.
        - <...> denotes the average over all neighboring objects.

    The nematic order parameter ranges from 0 to 1:
        - S = 1 indicates perfect alignment (all neighbors are aligned with the object).
        - S = 0 indicates no alignment (random orientations).
    """
    # Compute the difference in angles
    delta_theta = theta_i - theta_n
    
    # Compute the nematic order parameter using cos(delta)^2
    nematic_order = np.nanmean(2 * np.cos(delta_theta)**2 - 1)
    
    return nematic_order

def compute_nematic_order_over_distance(orientation, distances, delta_r, scale, cond_area=None):
	"""
	Compute the nematic order over the distance in a binned map.

	Parameters
	----------
	orientation : ndarray of float
		Orientation map (angles in radians).
	distances : ndarray of int
		Flattened distance map of shape (orientation.shape[0] * orientation.shape[1], orientation.shape[0] * orientation.shape[1]).
	delta_r : float
		Width of the ring where the bin is considered for a specific distance r.
	cond_area : ndarray of bool, optional, default=None
		Boolean array indicating which bins to include in the analysis. If None, all bins are included.

	Returns
	-------
	nematic_values : ndarray
		Array containing the nematic order values for each bin at each specific distance.
		For each distance, the nematic order values of all bins are averaged.
	"""
	orientation_tmp = orientation[cond_area]
	orientation_flatten = orientation_tmp.flatten()
	maximum_r_distance = 3200 * scale / 2
	distances_r = np.arange(0, maximum_r_distance, delta_r)
	array_nematic_order = np.zeros((len(distances_r), distances.shape[0]))

	for count_dist, dist in enumerate(distances_r):
		cond_dist = (distances > dist) & (distances <= dist + delta_r)

		for count_bin in range(distances.shape[0]):
			array_nematic_order[count_dist, count_bin] = nematic_order(orientation_flatten[count_bin], orientation_flatten[cond_dist[count_bin]])

	return array_nematic_order, distances_r

def main(images, step_im, coords, scale, nb_orientation, bin_size, cells_per_block=(1, 1), plot=False):
	"""
	Run specific function to save in csv and plot the nematic alignment inside and outside
	
	"""
	delta_r = np.sqrt(2) * bin_size[0] * scale
	maximum_r_distance = 3200 * scale / 2
	distances_r = np.arange(0, maximum_r_distance, delta_r)

	nematic_over_distance_area_1_all = []
	nematic_over_distance_area_2_all = []

	for sample in range(len(images)):
		nematic_over_distance_area_1_tmp = []
		nematic_over_distance_area_2_tmp = []

		for i in tqdm(range(0, len(images[sample]), step_im)):
			_, weighted_sum, _ = extract_nematic_orientation(images[sample][i], nb_orientation, bin_size, cells_per_block)

			cond_zero = (weighted_sum[1] == 0) & (weighted_sum[0] == 0)
			nematic_map = np.arctan2(weighted_sum[1], weighted_sum[0])
			nematic_map[cond_zero] = np.nan
			cond_neg = nematic_map < 0
			nematic_map[cond_neg] += np.pi

			exp_id = sample + 1
			cond_bins_between_lines_1, cond_bins_between_lines_2 = condition_area(images[sample][i], coords[sample], nematic_map.shape, bin_size, weighted_sum, exp_id, plot)

			size_x, size_y = nematic_map.shape  # number of columns
			xmax, ymax = np.array(images[sample][i].shape) * scale # maximum coordinate value in µm
			distances_1 = compute_distance_between_bins(size_x, size_y, xmax, ymax, cond_bins_between_lines_1)
			distances_2 = compute_distance_between_bins(size_x, size_y, xmax, ymax, cond_bins_between_lines_2)

			nematic_over_distance_area_1, _ = compute_nematic_order_over_distance(nematic_map, distances_1, delta_r, scale, cond_bins_between_lines_1)
			nematic_over_distance_area_2, _ = compute_nematic_order_over_distance(nematic_map, distances_2, delta_r, scale, cond_bins_between_lines_2)

			mean_nematic_over_distance_area_1 = np.nanmean(nematic_over_distance_area_1, axis=1)
			mean_nematic_over_distance_area_2 = np.nanmean(nematic_over_distance_area_2, axis=1)

			nematic_over_distance_area_1_tmp.append(mean_nematic_over_distance_area_1)
			nematic_over_distance_area_2_tmp.append(mean_nematic_over_distance_area_2)
			# print(np.array(nematic_over_distance_area_2_tmp).shape)

		nematic_over_distance_area_1_all.append(np.mean(nematic_over_distance_area_1_tmp, axis=0))
		nematic_over_distance_area_2_all.append(np.mean(nematic_over_distance_area_2_tmp, axis=0))
		# print(np.array(nematic_over_distance_area_1_all).shape)

	mean_nematic_1 = np.mean(nematic_over_distance_area_1_all, axis=0)
	std_nematic_1 = np.std(nematic_over_distance_area_1_all, axis=0)
	mean_nematic_2 = np.mean(nematic_over_distance_area_2_all, axis=0)
	std_nematic_2 = np.std(nematic_over_distance_area_2_all, axis=0)

	return mean_nematic_1, std_nematic_1, mean_nematic_2, std_nematic_2, distances_r

def plot_nematic(output_folder, output_file_name, mean_nematic_1, std_nematic_1, mean_nematic_2, std_nematic_2, distances_r):
	"""
	Plot the previous computation
	
	"""
	fig, ax = plt.subplots(figsize=_figsize)

	ax.plot(distances_r, mean_nematic_1, color=_color_rippling, linewidth=2, label='Inside prey colony', alpha=_alpha)
	ax.plot(distances_r, mean_nematic_2, color=_color_swarming, linewidth=2, label='Outside prey colony', alpha=_alpha)
	ax.plot(distances_r, np.zeros(len(distances_r)), color='k', linewidth=0.5, linestyle=":")
	ax.fill_between(distances_r, 
					mean_nematic_1 - std_nematic_1, 
					mean_nematic_1 + std_nematic_1, 
					color=_color_rippling, 
					alpha=0.5*_alpha, 
					linewidth=0)
	ax.fill_between(distances_r, 
					mean_nematic_2 - std_nematic_2, 
					mean_nematic_2 + std_nematic_2, 
					color=_color_swarming, 
					alpha=0.5*_alpha, 
					linewidth=0)
	ax.set_xlabel(r"Distance ($\mu$m)",fontsize=_fontsize)
	ax.set_ylabel("Nematic order",fontsize=_fontsize)
	ax.legend(loc='best', handlelength=1, borderpad=0, frameon=False, fontsize=_fontsize_ticks)
	# plt.tick_par(axis="both", which="both", labelsize=fontsize)
	# plt.xlim(xmin, xmax)
	ax.tick_params(axis='both',
				which='major',
				labelsize=_fontsize_ticks)
	# ax.set_xticks(np.arange(5, 90, step=20))
	# ax.set_yticks(np.arange(0, 1.1, step=0.2))
	delta_r = np.sqrt(2) * bin_size[0] * scale
	ax.set_xlim(delta_r, np.max(distances_r))
	ax.set_ylim(-0.1, 1)

	isExist = os.path.exists(output_folder)
	if not isExist:
		# Create a new directory because it does not exist
		os.makedirs(output_folder)
		print("The new directory is created!")
	fig.savefig(output_folder+output_file_name+'.png', bbox_inches='tight', dpi=_dpi)
	fig.savefig(output_folder+output_file_name+'.svg', dpi=_dpi)

	plt.show()

# %%
# Put the folder where you saved the data from figS5
image_folder = ''

image_filename_1 = '2024_08_05_72h_100X_FrzE_coli_edge.nd2'
coords_filename_1 = '2024_08_05_72h_100X_FrzE_coli_edge_coords.csv'
image_filename_2 = '2024_08_04_48h_100X_FrzE_coli_alignement_in_prey_1.nd2'
coords_filename_2 = '2024_08_04_48h_100X_FrzE_coli_alignement_in_prey_1_coords.csv'
image_filename_3 = '2024_08_04_48h_100X_FrzE_coli_alignement_in_prey_2.nd2'
coords_filename_3 = '2024_08_04_48h_100X_FrzE_coli_alignement_in_prey_2_coords.csv'

image_path_1 = image_folder + image_filename_1
coords_path_1 = image_folder + coords_filename_1
image_path_2 = image_folder + image_filename_2
coords_path_2 = image_folder + coords_filename_2
image_path_3 = image_folder + image_filename_3
coords_path_3 = image_folder + coords_filename_3

image_1 = ND2Reader(image_path_1)
coords_1 = pd.read_csv(coords_path_1)
image_2 = ND2Reader(image_path_2)
coords_2 = pd.read_csv(coords_path_2)
image_3 = ND2Reader(image_path_3)
coords_3 = pd.read_csv(coords_path_3)

images = [image_1, image_2, image_3]
step_im = 50
coords = [coords_1, coords_2, coords_3]
scale = 0.0646028
nb_orientation = 12
bin_size = (50, 50) # Taille des bins (hauteur, largeur)

mean_nematic_1, std_nematic_1, mean_nematic_2, std_nematic_2, distances_r = main(images, step_im, coords, scale, nb_orientation, bin_size)

# %%
# Plot
output_folder = '/home/jb/Documents/DATA/a_thesis/codes/python/codes_for_plotting_figures_papers_2023/data_for_supp_main/plots/'
output_file_name = 'FrzE_alignement_edge_prey_colony'
plot_nematic(output_folder, output_file_name, mean_nematic_1, std_nematic_1, mean_nematic_2, std_nematic_2, distances_r)