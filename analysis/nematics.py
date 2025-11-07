"""
Author: Jean-Baptiste Saulnier
Date: 2024-11-21

This module provides tools for analyzing nematic orientation in images.
It includes functionalities to extract nematic orientation, compute distances
between bins in a grid, and evaluate nematic order over distances. 
Additionally, it contains utility functions for visualization and processing.

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
from skimage.measure import label, regionprops
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import cKDTree
from tqdm import tqdm


class TypeError(Exception):
    """
    Custom exception for handling type errors in conditional area checks.

    Raises
    ------
    TypeError
        If the provided conditional area is not None or not a 2D numpy array
        with the same shape as the orientation parameter.
    """
    def __init__(self):
        super().__init__('cond_area must be None or a 2D numpy array with the same shape as the orientation parameter')


def graner_tool(weights, nb_orientation):
    """
    Compute the dominant orientation and alignment strength using the Graner tool.

    Parameters
    ----------
    weights : ndarray
        Histogram of orientations for a specific cell.
    nb_orientation : int
        Number of orientation bins.

    Returns
    -------
    e_max : float
        Largest eigenvalue representing alignment strength.
    v_max : ndarray
        Corresponding eigenvector (dominant orientation).
    diff_e : float
        Difference between the two eigenvalues.

    Example
    -------
    >>> weights = np.array([0.1, 0.3, 0.5, 0.1])
    >>> e_max, v_max, diff_e = graner_tool(weights, nb_orientation=4)
    """
    weights_tmp = np.concatenate((weights, weights))
    orientations_arr = np.arange(2 * nb_orientation)
    orientation_bin_midpoints = 2 * np.pi * (orientations_arr + 0.5) / (2 * nb_orientation)

    x = weights_tmp * np.cos(orientation_bin_midpoints)
    y = weights_tmp * np.sin(orientation_bin_midpoints)

    mat = np.zeros((2, 2))
    mat[0, 0] = np.sum(x * x)
    mat[0, 1] = np.sum(x * y)
    mat[1, 0] = np.sum(x * y)
    mat[1, 1] = np.sum(y * y)

    e, v = np.linalg.eig(mat)

    index = np.argmax(e)
    diff_e = np.abs(e[0] - e[1])

    return e[index], v[index], diff_e


def extract_nematic_orientation(image, nb_orientation=12, bin_size=(50, 50), cells_per_block=(1, 1), channel_axis=None, plot=False):
    """
    Extract nematic orientations from an image using Histogram of Oriented Gradients (HOG).

    Parameters
    ----------
    image : ndarray
        Input image for analysis.
    nb_orientation : int
        Number of orientation bins for the HOG feature extraction.
    bin_size : tuple of int
        Size of the HOG pixel bin.
    cells_per_block : tuple of int
        Number of cells per block in the HOG feature extraction.
    channel_axis : int or None, optional
        Axis of color channels in the image. None for grayscale images.
    plot : bool, optional
        If True, displays the input image and the HOG visualization.

    Returns
    -------
    hog_image_rescaled : ndarray
        Rescaled HOG visualization.
    weighted_sum : ndarray
        Array containing weighted nematic vectors for each cell.
    hist_all : ndarray
        Histogram of orientations for each cell.

    Example
    -------
    >>> image = np.random.random((256, 256))
    >>> hog_image_rescaled, weighted_sum, hist_all = extract_nematic_orientation(image, 9, (8, 8), (2, 2))
    """
    # Compute HOG features and visualize
    fd, hog_image = hog(image,
                        orientations=nb_orientation,
                        pixels_per_cell=(bin_size[0], bin_size[1]),
                        cells_per_block=cells_per_block,
                        visualize=True,
                        transform_sqrt=True,
                        feature_vector=False,
                        channel_axis=channel_axis)
    
    # Rescale HOG visualization
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 3200))
    
    if plot:
        # Visualize the input image and the HOG features
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(32, 16), sharex=True, sharey=True)
        ax1.axis('off')
        ax1.imshow(image, cmap=plt.cm.gray)
        ax1.set_title('Input image')
        ax2.axis('off')
        im = ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray, 
                        vmin=0, vmax=np.max(hog_image_rescaled) / nb_orientation, aspect='auto')
        ax2.set_title('Histogram of Oriented Gradients')
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)
        plt.show()

    # Extract orientation data for each cell
    n_cells_y, n_cells_x, _, _, _ = fd.shape
    weighted_sum = np.zeros((2, n_cells_y, n_cells_x))
    hist_all = np.zeros((n_cells_y, n_cells_x, nb_orientation))

    for i in range(n_cells_y):
        for j in range(n_cells_x):
            hist = fd[i, j, 0, 0, :]
            hist_all[i, j, :] = hist

            e_max, v_max, diff_e = graner_tool(weights=hist, nb_orientation=nb_orientation)
            if diff_e > 2e-1:
                weighted_sum[0, i, j] = e_max * v_max[0]
                weighted_sum[1, i, j] = e_max * v_max[1]
            else:
                weighted_sum[:, i, j] = 0

    return hog_image_rescaled, weighted_sum, hist_all


def draw_segments_in_bin(i, j, bin_size, ax, weighted_sum):
    """
    Draw segments within a specific bin to represent nematic orientation.

    Parameters
    ----------
    i : int
        Row index of the bin.
    j : int
        Column index of the bin.
    bin_size : tuple of int
        Size of the bin as (height, width).
    ax : matplotlib.axes.Axes
        Matplotlib axis object where the segments will be drawn.
    weighted_sum : ndarray
        Array of shape (2, n_rows, n_cols) containing the weighted nematic vectors.

    Returns
    -------
    None
        The function draws the segments directly on the given axis.

    Example
    -------
    >>> fig, ax = plt.subplots()
    >>> draw_segments_in_bin(5, 5, (20, 20), ax, weighted_sum)
    """
    # Compute bin coordinates
    x_start = i * bin_size[1]
    x_end = (i + 1) * bin_size[1]
    y_start = j * bin_size[0]
    y_end = (j + 1) * bin_size[0]

    # Compute the center of the bin
    x_center = x_start + bin_size[1] / 2
    y_center = y_start + bin_size[0] / 2

    # Define the segment length
    segment_length = min(bin_size) / 2

    # Compute segment endpoints
    x1 = x_center + (segment_length / 2) * weighted_sum[0, i, j]
    y1 = y_center + (segment_length / 2) * weighted_sum[1, i, j]
    x2 = x_center - (segment_length / 2) * weighted_sum[0, i, j]
    y2 = y_center - (segment_length / 2) * weighted_sum[1, i, j]

    # Normalize segment if the vector's magnitude is above a threshold
    norm = np.sqrt(weighted_sum[0, i, j]**2 + weighted_sum[1, i, j]**2)
    if norm > 1e-2:
        x1_norm = x_center + (segment_length / 2) * (weighted_sum[0, i, j] / norm)
        y1_norm = y_center + (segment_length / 2) * (weighted_sum[1, i, j] / norm)
        x2_norm = x_center - (segment_length / 2) * (weighted_sum[0, i, j] / norm)
        y2_norm = y_center - (segment_length / 2) * (weighted_sum[1, i, j] / norm)
    else:
        x1_norm, y1_norm = x1, y1
        x2_norm, y2_norm = x2, y2

    # Draw the segment
    ax.plot([y1_norm, y2_norm], [x1_norm, x2_norm], color='grey', linewidth=2)
    ax.plot([y1, y2], [x1, x2], color='r', linewidth=2)

    # Draw a rectangle to visualize the bin
    rect = plt.Rectangle(
        (y_start, x_start),
        bin_size[1],
        bin_size[0],
        linewidth=0.2,
        edgecolor='r',
        facecolor='none'
    )
    ax.add_patch(rect)


def compute_distance_between_bins(size_x, size_y, xmax, ymax, mask=None):
    """
    Compute the distance between each bin in a grid.

    Parameters
    ----------
    size_x : int
        Number of columns in the grid.
    size_y : int
        Number of rows in the grid.
    xmax : float
        Maximum x-coordinate of the grid in physical units.
    ymax : float
        Maximum y-coordinate of the grid in physical units.
    mask : ndarray or None, optional
        Binary mask to include/exclude certain areas. Should match the grid's spatial dimensions.

    Returns
    -------
    distances : ndarray
        Array of shape (size_x * size_y, size_x * size_y) containing the pairwise distances
        between bins in the grid.

    Example
    -------
    >>> size_x, size_y = 16, 16
    >>> xmax, ymax = 100, 100
    >>> distances = compute_distance_between_bins(size_x, size_y, xmax, ymax)
    """
    # Generate the grid coordinates
    x_coords = np.linspace(0, xmax, size_x)
    y_coords = np.linspace(0, ymax, size_y)
    xv, yv = np.meshgrid(x_coords, y_coords, indexing='ij')

    # Apply conditional area if provided
    if mask is not None:
        if not isinstance(mask, np.ndarray) or mask.shape != (size_x, size_y):
            raise TypeError()
        xv, yv = xv[mask], yv[mask]

    # Flatten the coordinate arrays
    xv_flat = xv.flatten()
    yv_flat = yv.flatten()

    # Create an array to store distances
    distances = np.tile(np.zeros_like(xv_flat, dtype=np.float16), (xv_flat.size, 1))

    # Compute pairwise distances using broadcasting
    for idx, (x1, y1) in enumerate(zip(xv_flat, yv_flat)):
        dx = x1 - xv_flat
        dy = y1 - yv_flat
        distances[idx] = np.sqrt(dx**2 + dy**2)

    return distances


def compute_nematic_order(theta_i, theta_n):
    """
    Compute the nematic order for an object `i` with its neighbors `n`.

    Parameters
    ----------
    theta_i : float
        Orientation angle of the object `i` in radians.
    theta_n : ndarray of float
        Array of orientation angles of the neighboring objects in radians.

    Returns
    -------
    nematic_order : float
        The nematic order parameter for the object `i` with its neighbors `n`.

    Explanation
    -----------
    The nematic order parameter quantifies the alignment between the orientation
    of an object and its neighbors. It is calculated using:

        S = <2*cos²(Δθ) - 1>

    where:
        - `S` is the nematic order parameter.
        - Δθ = θ_i - θ_n is the difference between the angles.
        - `<...>` represents the average over all neighbors.

    The nematic order parameter ranges from:
        - 1 : perfect alignment (all neighbors are aligned with the object).
        - 0 : no alignment (random orientations).

    Example
    -------
    >>> theta_i = 0.5
    >>> theta_n = np.array([0.4, 0.6, 0.7])
    >>> nematic_order(theta_i, theta_n)
    0.92
    """
    # Compute the difference in angles
    delta_theta = theta_i - theta_n
    
    # Compute the nematic order parameter using cos²(Δθ)
    nematic_order = np.nanmean(2 * np.cos(delta_theta)**2 - 1)
    
    return nematic_order

def compute_nematic_order_over_distance(
    orientation,
    distances,
    delta_r,
    cond_area=None,
    max_distance=None,
    use_half=False,
    min_neighbors=1,
    return_counts=False,
):
    """
    Compute the nematic order per source bin as a function of pairwise distance,
    with optional fixed maximum distance (for cross-dataset comparability) and
    an optional half-range cap.

    Parameters
    ----------
    orientation : ndarray of float, shape (H, W)
        Orientation map (angles in radians), 2D. May contain NaNs for invalid bins.
    distances : ndarray of float, shape (N, N)
        Pairwise distance matrix between the kept bins. N must equal the number of
        True entries in `cond_area` (or H*W if cond_area is None and all bins are kept).
        Distances should be in the same physical unit as `max_distance` (if provided).
    delta_r : float
        Ring width used to select neighbors in [r, r + delta_r).
    cond_area : ndarray of bool, optional
        Boolean mask of shape (H, W) indicating which bins/pixels to include.
        If None, all bins are tentatively included (NaN handling is applied internally).
    max_distance : float or None, optional
        Fixed maximum radius to evaluate (e.g., a common cap like 3200*scale to
        compare across datasets). If None, uses the maximum value present in `distances`.
    use_half : bool, optional
        If True, the effective maximum radius becomes `max_distance/2` (or half of
        the observed max if `max_distance` is None). This reproduces the previous
        “distance/2” behavior.
    min_neighbors : int, optional
        Minimum number of neighbors required in a ring to compute a value. If fewer
        neighbors are present, the entry is set to NaN. Default is 1.
    return_counts : bool, optional
        If True, also return the neighbor counts per (ring, bin) as an array of ints.

    Returns
    -------
    array_nematic_order : ndarray of float, shape (n_r, N)
        Nematic order values for each distance ring (rows) and for each source bin (columns).
        If a source bin has too few valid neighbors in a ring (< min_neighbors), its entry is NaN.
        If a source bin's own angle is NaN, its entire column remains NaN.
    distances_r : ndarray of float, shape (n_r,)
        The starting radii r for each ring [r, r + delta_r).
    counts : ndarray of int, shape (n_r, N), optional
        Returned only if `return_counts=True`. Number of *valid* neighbors used in each (ring, bin).

    Notes
    -----
    - NaN-safe behavior:
        * Source bins with NaN orientation are skipped (leave NaN outputs).
        * Neighbor sets exclude bins with NaN orientation before calling `nematic_order`.
    - Requires an external function:
        nematic_order(theta_ref: float, thetas_neighbors: ndarray) -> float
      that computes the nematic order for a reference angle and a set of neighbor angles.
    """
    # -- Build/validate mask ---------------------------------------------------
    if cond_area is None:
        cond_area = np.ones_like(orientation, dtype=bool)

    kept_idx = cond_area.ravel()
    orientation_flat = orientation.ravel()[kept_idx]
    N = orientation_flat.size

    # Precompute validity of angles (exclude NaNs)
    valid_angle_mask = np.isfinite(orientation_flat)

    # -- Validate distances dimension -----------------------------------------
    if distances.shape[0] != distances.shape[1]:
        raise ValueError("`distances` must be a square (N x N) matrix.")
    if distances.shape[0] != N:
        raise ValueError(
            f"`distances` size ({distances.shape[0]}) does not match the number of kept bins ({N}). "
            "Ensure `distances` was computed on the same kept set (cond_area)."
        )

    # -- Determine maximum radius ---------------------------------------------
    observed_max = float(np.nanmax(distances))
    if max_distance is None:
        max_r = observed_max
    else:
        max_r = min(float(max_distance), observed_max)

    if use_half:
        max_r *= 0.5

    # -- Distance rings --------------------------------------------------------
    distances_r = np.arange(delta_r, max_r + 1e-9, delta_r)

    # -- Outputs ---------------------------------------------------------------
    array_nematic_order = np.full((len(distances_r), N), np.nan, dtype=float)
    counts = np.zeros((len(distances_r), N), dtype=int) if return_counts else None

    # -- Main loops ------------------------------------------------------------
    for i_r, r0 in enumerate(distances_r):
        r1 = r0 + delta_r
        ring_mask = (distances >= r0) & (distances < r1)

        for i_bin in range(N):
            # Skip sources with NaN orientation
            if not valid_angle_mask[i_bin]:
                if return_counts:
                    counts[i_r, i_bin] = 0
                continue

            nei_mask = ring_mask[i_bin].copy()

            # Exclude self if [0, delta_r) contains 0
            if r0 <= 0.0 < r1:
                nei_mask[i_bin] = False

            # Keep only neighbors with valid (non-NaN) orientations
            nei_mask &= valid_angle_mask

            n_nei = int(np.count_nonzero(nei_mask))
            if return_counts:
                counts[i_r, i_bin] = n_nei

            if n_nei < min_neighbors:
                # Not enough valid neighbors: leave NaN
                continue

            thetas_neighbors = orientation_flat[nei_mask]
            array_nematic_order[i_r, i_bin] = compute_nematic_order(
                orientation_flat[i_bin], thetas_neighbors
            )

    if return_counts:
        return array_nematic_order, distances_r, counts
    return array_nematic_order, distances_r



def compute_topological_charge_8_neighbors(grid_angles: np.ndarray) -> np.ndarray:
    """
    Computes the topological charges on a grid of angles using the 8-neighbor approach.
    
    Parameters:
        grid_angles (numpy.ndarray): 2D array of angles (in radians) representing orientations 
                                     on the grid.
        
    Returns:
        charges (numpy.ndarray): 2D array of topological charges for each grid cell,
                                 excluding the edges (output size is (nrows-2, ncols-2)).
    
    Example
    -------
    >>> grid_angles = np.array([
    ...     [0, 0.1, 0.2],
    ...     [0.3, 0.4, 0.5],
    ...     [0.6, 0.7, 0.8]
    ... ])
    >>> charges = compute_topological_charge_8_neighbors(grid_angles)
    >>> print(charges)
    [[0.]]  # Example output for a uniform gradient of angles.
    """
    # Get the dimensions of the grid
    nrows, ncols = grid_angles.shape

    # Normalize the angles to the range [-pi/2, pi/2]
    grid_angles[grid_angles <= -np.pi / 2] += np.pi
    grid_angles[grid_angles > np.pi / 2] -= np.pi
    
    # Initialize the array for storing topological charges (smaller than input to avoid edges)
    charges = np.zeros((nrows - 2, ncols - 2))
    
    # Iterate over the inner grid (excluding edges)
    for i in range(1, nrows - 1):
        for j in range(1, ncols - 1):
            # Extract the 8 neighbor angles in a clockwise order
            contour_angles = [
                grid_angles[i-1, j-1],  # Top-left
                grid_angles[i, j-1],     # Left
                grid_angles[i+1, j-1],  # Bottom-left
                grid_angles[i+1, j],    # Bottom
                grid_angles[i+1, j+1],  # Bottom-right
                grid_angles[i, j+1],    # Right
                grid_angles[i-1, j+1],  # Top-right
                grid_angles[i-1, j]    # Top
            ]
            
            # Compute angular differences along the contour
            # Append the first angle at the end to close the loop
            d_theta = np.diff(contour_angles + [contour_angles[0]])
            
            # Normalize the angular differences to the range [-pi, pi]
            d_theta[d_theta <= -np.pi / 2] += np.pi
            d_theta[d_theta > np.pi / 2] -= np.pi
            
            # Sum up the angular differences to calculate the total winding angle
            # Divide by 2*pi to normalize to topological charge
            # charge = np.sum(d_theta) / (2 * np.pi)
            charge = np.nansum(d_theta) / (2 * np.pi)

            # # If the grid cell is empty put the charge to nan
            # if np.isnan(grid_angles[i, j]):
            #     charge = np.nan
            
            # Store the calculated charge (shifting by -1 due to edge exclusion)
            charges[i-1, j-1] = charge
    
    return charges


def compute_defect_positions(charges, bin_size):
    """
    Compute the positions and centroids of +1/2 and -1/2 topological defects on a grid.

    Parameters:
    -----------
    charges : numpy.ndarray
        A 2D array representing the topological charges on the grid.
    bin_size : int
        The size of each grid cell (in pixel).
        
    Returns:
    --------
    centroids_plus_half : numpy.ndarray
        Array containing the centroids (x, y) of the +1/2 defects on the grid.
    centroids_minus_half : numpy.ndarray
        Array containing the centroids (x, y) of the -1/2 defects on the grid.
    position_plus_half : numpy.ndarray
        Array of positions (x, y) of all the grid cells with +1/2 defects.
    position_minus_half : numpy.ndarray
        Array of positions (x, y) of all the grid cells with -1/2 defects.
    """
    # Create binary masks for +1/2 and -1/2 defects
    cond_plus_half = np.isclose(charges, +0.5, atol=1e-2)
    cond_minus_half = np.isclose(charges, -0.5, atol=1e-2)
    
    # Label connected regions
    labels_plus_half = label(cond_plus_half)
    labels_minus_half = label(cond_minus_half)

    # All defects
    index_x = np.tile(np.arange(charges.shape[0]), (charges.shape[1],1))
    index_y = index_x.T
    position_plus_half = np.array([
        index_x[cond_plus_half] * bin_size + 3/2 * bin_size,
        index_y[cond_plus_half] * bin_size + 3/2 * bin_size
        ])
    position_minus_half = np.array([
        index_x[cond_minus_half] * bin_size + 3/2 * bin_size,
        index_y[cond_minus_half] * bin_size + 3/2 * bin_size
        ])
    
    # Compute centroids for +1/2 defects
    centroids_plus_half = np.array([
        [prop.centroid[1] * bin_size + 3/2* bin_size,
         prop.centroid[0] * bin_size + 3/2 * bin_size]
        for prop in regionprops(labels_plus_half)
    ])
    
    # Compute centroids for -1/2 defects
    centroids_minus_half = np.array([
        [prop.centroid[1] * bin_size + 3/2 * bin_size,
         prop.centroid[0] * bin_size + 3/2 * bin_size]
        for prop in regionprops(labels_minus_half)
    ])
    
    return centroids_plus_half, centroids_minus_half, position_plus_half, position_minus_half


def compute_track_indices_gap(df, frame_i, max_distance, gaps=1, col_id='id', col_t='frame', col_x='x', col_y='y'):
    """
    Search for the index at frame `i` associated with the indices at frame `j` 
    (where `j = i + gap`) based on proximity and constraints.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe containing columns for frame number, coordinates, and IDs.
    frame_i : int
        The reference frame index for tracking.
    max_distance : float
        Maximum allowed distance for associating points between frames.
    gaps : int, optional
        Number of frames to skip for gap tracking (default is 1).
    col_id : str, optional
        Column name for IDs (default is 'id').
    col_t : str, optional
        Column name for frame number (default is 'frame').
    col_x : str, optional
        Column name for x-coordinates (default is 'x').
    col_y : str, optional
        Column name for y-coordinates (default is 'y').

    Returns:
    --------
    df_tmp : pandas.DataFrame
        Updated dataframe with tracked IDs and interpolated points (if necessary).
    """
    df_tmp = df.copy()  # Create a copy of the dataframe to avoid modifying the original.
    cond_ti = df_tmp[col_t] == frame_i  # Filter rows for the current frame `i`.
    
    # Get coordinates of defects in frame `i`.
    coords_ti = df_tmp.loc[cond_ti, [col_x, col_y]].values
    
    # Build a KDTree for efficient nearest neighbor search in frame `i`.
    tree = cKDTree(coords_ti)
    id_ti = df_tmp.loc[cond_ti, col_id].values  # Extract IDs for frame `i`.

    # Initialize tracked indices for frame `i` as an empty list.
    target_id_local_frame_i = np.array([])

    # Loop over gaps (to allow tracking across multiple frames).
    for gap in range(1, gaps + 1):
        cond_tj = df_tmp[col_t] == frame_i + gap  # Condition for frame `j = i + gap`.

        # Check if there are defects in frame `i` and `j`.
        if np.sum(cond_ti) > 0 and np.sum(cond_tj) > 0:
            # Get coordinates of defects in frame `j`.
            coords_tj = df_tmp.loc[cond_tj, [col_x, col_y]].values

            # Find the nearest neighbor in frame `i` for each defect in frame `j`.
            target_dist, target_id_local = tree.query(coords_tj, k=1)

            # Map the local indices to the corresponding IDs in frame `i`.
            target_id = id_ti[target_id_local.astype(int)]

            # Handle duplicate matches (multiple defects associated with the same target).
            unique_elements, counts = np.unique(target_id, return_counts=True)
            duplicates = unique_elements[counts > 1]  # Identify duplicates.
            for duplicate in duplicates:
                mask = target_id == duplicate
                indices = np.where(mask)[0]
                if mask.any():
                    # Keep the closest defect and set others to NaN.
                    min_dist_index = indices[np.argmin(target_dist[mask])]
                target_id[mask] = np.nan
                if mask.any():
                    target_id[min_dist_index] = duplicate

            # Set target IDs to NaN for defects beyond the maximum distance.
            cond_high_dist = target_dist > max_distance
            target_id[cond_high_dist] = np.nan

            # Handle defects with no valid match (high distance or duplicates).
            cond_nan = np.isnan(target_id)
            target_id_local = target_id_local.astype(float)
            target_id_local[cond_nan] = np.nan

            if gap == 1:
                # Save the target indices for frame `i`.
                target_id_local_frame_i = target_id_local[~cond_nan].copy()
                
                # Assign new IDs to unmatched defects.
                index_max = np.nanmax(df_tmp[col_id])  # Find the current max ID.
                if np.isnan(index_max):
                    index_max = -1
                target_indices_nan = np.arange(index_max + 1, index_max + 1 + np.sum(cond_nan))
                target_id[cond_nan] = target_indices_nan
                
                # Update the dataframe with the new IDs for frame `j`.
                df_tmp.loc[cond_tj, col_id] = target_id
            else:
                indices_frame_i = np.arange(len(coords_ti))  # Indices for frame `i`.

                # Find unmatched indices in frame `i` not yet associated.
                non_associated_indices = np.setdiff1d(indices_frame_i, target_id_local_frame_i)

                # Loop over unmatched indices and interpolate for missing values.
                for non_associated_indice in non_associated_indices:
                    cond_index = target_id_local == non_associated_indice

                    if np.sum(cond_index) > 0:
                        if np.sum(cond_index) > 1:
                            print("Be careful something wrong in the code")
                        
                        # Add the new index to the detected list.
                        target_id_local_frame_i = np.concatenate((target_id_local_frame_i, np.array([non_associated_indice])))

                        # Interpolate coordinates for the missing defect.
                        new_id = id_ti[non_associated_indice.astype(int)]
                        new_x = coords_ti[non_associated_indice, 0] + (coords_tj[cond_index, 0][0] - coords_ti[non_associated_indice, 0]) / gap
                        new_y = coords_ti[non_associated_indice, 1] + (coords_tj[cond_index, 1][0] - coords_ti[non_associated_indice, 1]) / gap

                        # Create a new row for the interpolated defect.
                        new_row = {
                            col_id: new_id,
                            col_t: frame_i + 1,
                            col_x: new_x,
                            col_y: new_y
                        }
                        # Add the interpolated row to the dataframe.
                        df_tmp = pd.concat([df_tmp, pd.DataFrame([new_row])], ignore_index=True)

    return df_tmp


def track_defects_gap(df_defects, bin_size, gaps=1, col_id='id', col_t='frame', col_x='x', col_y='y', col_charge='charge'):
    """
    Track topological defects across frames based on the distance criterion.

    Parameters:
    -----------
    df_defects : pandas.DataFrame
        A dataframe containing columns: `frame`, `charge`, `x`, `y`, 
        representing the frame number, charge type, and coordinates of defects.
    bin_size : int
        The size of each grid cell in pixels (used to compute max distance for tracking).
    gaps : int, optional
        Number of frames to skip for gap tracking (default is 1).
    col_id : str, optional
        Column name for defect track IDs (default is 'id').
    col_t : str, optional
        Column name for frame numbers (default is 'frame').
    col_x : str, optional
        Column name for x-coordinates (default is 'x').
    col_y : str, optional
        Column name for y-coordinates (default is 'y').
    col_charge : str, optional
        Column name for defect charge types (default is 'charge').

    Returns:
    --------
    df_tracks_defects : pandas.DataFrame
        A dataframe with added columns: `id`, `frame`, `charge`, `x`, `y`, 
        and `len_traj`, representing the defect's track ID, frame, charge, 
        position, and trajectory length.
    """
    # Create a copy of the input dataframe to avoid modifying the original
    df_tracks_defects = df_defects.copy()
    
    # Initialize the 'id' column as NaN to store defect track IDs
    df_tracks_defects[col_id] = np.nan

    # Calculate the maximum allowed distance for tracking defects between frames
    max_distance = np.sqrt(2) * bin_size  # Maximum distance within a grid cell diagonal

    # Get unique defect charge types and frame numbers for iterating
    defect_types = np.unique(df_tracks_defects[col_charge])  # E.g., +1/2 and -1/2 charges
    frames = np.unique(df_tracks_defects[col_t])  # List of all unique frame indices

    # Iterate over each defect type (e.g., +1/2 or -1/2 charge)
    for defect_type in defect_types:
        # Filter defects of the current charge type
        cond_type = df_tracks_defects.loc[:, col_charge] == defect_type

        # Assign initial unique IDs to defects in the first frame of this defect type
        cond_first_frame = df_tracks_defects[cond_type].loc[:, col_t] == frames[0]
        
        if df_tracks_defects[col_id].isna().all():
            # If no IDs exist, start from 0
            id_correction = 0
        else:
            # Otherwise, continue from the current maximum ID
            id_correction = np.nanmax(df_tracks_defects[col_id]) + 1

        # Assign unique IDs to defects in the first frame
        df_tracks_defects.loc[cond_type & cond_first_frame, col_id] = np.arange(np.sum(cond_first_frame)) + id_correction

        # Filter the relevant columns for input to the tracking function
        df_input = df_tracks_defects.loc[cond_type, [col_id, col_t, col_x, col_y]]

        # Iterate over each frame (except the last one, since there's no frame after it to track)
        for frame in tqdm(frames[:-1]):  # Display progress with tqdm
            # Update tracking IDs for the current frame using the helper function
            df_input = compute_track_indices_gap(
                df=df_input, 
                frame_i=frame, 
                max_distance=max_distance, 
                gaps=gaps, 
                col_id=col_id, 
                col_t=col_t, 
                col_x=col_x, 
                col_y=col_y
            )
        
        # Add the defect charge type back to the tracked dataframe
        df_input.loc[:, col_charge] = defect_type

        # Perform an outer merge with the main dataframe to update IDs
        df_tracks_defects = pd.merge(
            df_tracks_defects,
            df_input,
            on=[col_t, col_x, col_y, col_charge],
            how='outer',
            suffixes=('_tracks', '_input')
        )
        
        # Use 'id_input' values where they exist; otherwise, fallback to 'id_tracks'
        df_tracks_defects[col_id] = df_tracks_defects['id_input'].combine_first(df_tracks_defects['id_tracks'])

        # Drop intermediate columns created by the merge
        df_tracks_defects = df_tracks_defects.drop(columns=['id_tracks', 'id_input'])

        # Remove duplicates based on the combination of frame, coordinates, charge, and ID
        df_tracks_defects = df_tracks_defects.drop_duplicates(subset=[col_t, col_x, col_y, col_charge, col_id])

        # Reset the index to ensure a clean output dataframe
        df_tracks_defects = df_tracks_defects.reset_index(drop=True)

    # Return the dataframe with updated tracking information
    return df_tracks_defects