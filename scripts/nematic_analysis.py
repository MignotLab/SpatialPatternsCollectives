# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
from tqdm import tqdm
import os
import sys
# Ajouter la racine du projet au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from analysis import nematics as no
import utils as tl

# Définir la police sans-serif pour les graphiques
plt.rcParams['font.sans-serif'] = 'Arial'
# Utiliser la police sans-serif pour les graphiques
plt.rcParams['font.family'] = 'sans-serif'


_alpha = 0.7
_size_height_figure = 7
_figsize = (_size_height_figure, _size_height_figure-1)
_dpi = 300
_fontsize = 30
_fontsize_ticks = _fontsize / 1.5

def main(images, output_path, step_im, scale, nb_orientation, bin_size, cells_per_block=(1, 1)):
    """
    Run the nematic alignment analysis over distance and SAVE per-sample curves (triplicates kept separate).

    Parameters
    ----------
    images : list[list[np.ndarray]]
        Nested list: images[sample][frame] -> 2D image array.
    output_path : str
        Base file path (without extension) used to save results.
        This function writes a single CSV:
          - if coords is None:  <output_path>_per_sample.csv  with columns distances_r, sample_1..sample_S
          - if coords is provided: <output_path>_per_sample.csv  with columns distances_r, area1_s1.., area2_s1..
    step_im : int
        Frame step when iterating images (subsampling).
    coords : list or None
        If provided, per-sample coordinates for splitting two areas via `condition_area`.
    scale : float
        Pixel-to-µm scaling factor.
    nb_orientation : int
        Number of orientation bins for the `extract_nematic_orientation` routine.
    bin_size : tuple[int, int]
        Size (in pixels) of the bin used for the orientation extraction.
    cells_per_block : tuple[int, int], optional
        Forwarded to `extract_nematic_orientation`.
    plot : bool, optional
        If True, `condition_area` may visualize/debug areas.

    Saves
    -----
    CSV file at <output_path>_per_sample.csv
        - If coords is None:
            columns = [distances_r, sample_1, sample_2, ...]
        - If coords is provided:
            columns = [distances_r, area1_s1, area1_s2, ..., area2_s1, area2_s2, ...]

    Notes
    -----
    - Expects external functions:
        extract_nematic_orientation(image, nb_orientation, bin_size, cells_per_block)
        condition_area(image, coords_for_sample, map_shape, bin_size, weighted_sum, exp_id, plot)
        compute_distance_between_bins(size_x, size_y, xmax, ymax)
        compute_nematic_order_over_distance(orientation, distances, delta_r, scale, max_distance)
    - Distance cap fixed to `3200*scale/2` for cross-dataset comparability.
    """
    # -- Ring params (identiques à ta pipeline) --------------------------------
    delta_r = bin_size[0] * scale
    maximum_r_distance = 3200 * scale / 2.0
    distances_r = np.arange(delta_r, maximum_r_distance, delta_r)

    n_samples = len(images)

    # --------- 1 zone : on garde les courbes moyennes par échantillon ---------
    per_sample_mean = []  # (S, n_r) après vstack

    for sample in range(n_samples):
        per_frame_means = []

        for i in tqdm(range(0, len(images[sample]), step_im)):
            # Orientation map
            _, weighted_sum, _ = no.extract_nematic_orientation(
                images[sample][i], nb_orientation, bin_size, cells_per_block, plot=True
            )
            cond_zero = (weighted_sum[1] == 0) & (weighted_sum[0] == 0)
            nematic_map = np.arctan2(weighted_sum[1], weighted_sum[0])
            nematic_map[cond_zero] = np.nan
            cond_neg = nematic_map < 0
            nematic_map[cond_neg] += np.pi

            # Distances (tous bins; la fn nematic est NaN-safe)
            size_x, size_y = nematic_map.shape
            xmax, ymax = np.array(images[sample][i].shape) * scale
            distances_single = no.compute_distance_between_bins(size_x, size_y, xmax, ymax, mask=None)

            nem_over_dist, _ = no.compute_nematic_order_over_distance(
                orientation=nematic_map,
                distances=distances_single,
                delta_r=delta_r,
                cond_area=None,
                max_distance=maximum_r_distance
            )

            per_frame_means.append(np.nanmean(nem_over_dist, axis=1))  # (n_r,)

        per_sample_mean.append(np.mean(per_frame_means, axis=0))       # (n_r,)

    per_sample_mean = np.vstack(per_sample_mean)  # (S, n_r)

    # --- Save wide CSV: distances_r + sample_i colonnes ---
    df = pd.DataFrame({"distances_r": distances_r})
    for s in range(per_sample_mean.shape[0]):
        df[f"sample_{s+1}"] = per_sample_mean[s]
    out_csv = f"{output_path}.csv"
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)


def plot_nematic(output_folder, output_file_name, means_nematic, stds_nematic, distances_r, colors, labels):
	"""
	Plot the previous computation
	
	"""
	fig, ax = plt.subplots(figsize=_figsize)

	ax.plot(distances_r, means_nematic[0], color=colors[0], linewidth=2, label=labels[0], alpha=_alpha)
	ax.plot(distances_r, means_nematic[1], color=colors[1], linewidth=2, label=labels[1], alpha=_alpha)
	ax.plot(distances_r, np.zeros(len(distances_r)), color='k', linewidth=0.5, linestyle=":")
	ax.fill_between(distances_r, 
					means_nematic[0] - stds_nematic[0], 
					means_nematic[0] + stds_nematic[0], 
					color=colors[0], 
					alpha=0.5*_alpha, 
					linewidth=0)
	ax.fill_between(distances_r, 
					means_nematic[1] - stds_nematic[1], 
					means_nematic[1] + stds_nematic[1], 
					color=colors[1], 
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
	ax.set_xlim(np.min(distances_r), np.max(distances_r))
	ax.set_ylim(-0.1, 1)

	current_max = np.max(distances_r)
	step = 20  # µm par exemple → ajuste selon ton échelle
	ticks = np.arange(np.min(distances_r), current_max, step)
	if np.min(distances_r) not in ticks:  # au cas où arrondi
		ticks = np.insert(ticks, 0, np.min(distances_r))
	ax.set_xticks(ticks)

	# save
	os.makedirs(output_folder, exist_ok=True)
	fig.savefig(output_folder + output_file_name + '.png', bbox_inches='tight', dpi=_dpi)
	fig.savefig(output_folder + output_file_name + '.svg', dpi=_dpi)

	plt.show()

# Load your images from in image_folder
image_folder = 'data/phase_contrast_images_100X/'
filenames = tl.list_files_in_directory(image_folder, ext='tif')
images = np.array([io.imread(filenames[i]) for i in range(len(filenames))])

# Compute the nematic order over distance and save a csv file in output_path
step_im = 1
scale = 0.0646028
nb_orientation = 12
bin_size = (5/scale, 5/scale) # Taille des bins (hauteur, largeur)
output_path = 'output/nematic_order_analysis/nematic_order_over_distance_raw_data_rippling'
main(images=[images],
	 step_im=step_im,
	 scale=scale,
	 nb_orientation=nb_orientation,
	 bin_size=bin_size,
	 cells_per_block=(1, 1),
	 output_path=output_path,
)


# Plot the results
df = pd.read_csv(output_path + '.csv')
distances = df['distances_r'].to_numpy()
mean_nematic = np.mean(df[[col for col in df.columns[1:]]].to_numpy(), axis=1)
std_nematic = np.std(df[[col for col in df.columns[1:]]].to_numpy(), axis=1)
plot_nematic(
	output_folder='output/nematic_order_analysis/plot/',
	output_file_name='figure_nematic_over_distance_test',
	means_nematic=[mean_nematic, mean_nematic],
	stds_nematic=[std_nematic, mean_nematic],
	distances_r=distances,
	colors=["k", "k"],
	labels=['test', 'test'],
)
