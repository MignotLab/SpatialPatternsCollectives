"""
Author: Jean-Baptiste Saulnier
Date: 2024-11-21

This module provides internal functions for detecting objects from a labeled image.

"""
from skimage import io, morphology, measure
from skan.csr import Skeleton as skan_skel
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import utils as tl


def calculate_cumulative_distances(coords):
    """
    Calculate the cumulative distances between consecutive points in a list of coordinates.

    This function computes the Euclidean distance between each pair of consecutive points
    and returns the cumulative sum of these distances. The cumulative distances are calculated
    in a way that the first point starts with a distance of 0.

    Parameters
    ----------
    coords : numpy.ndarray
        A 2D array of shape (n, 2) where each row contains the x and y coordinates of a point.

    Returns
    -------
    numpy.ndarray
        A 1D array containing the cumulative distances of each point from the origin.
    """
    deltas = np.diff(coords, axis=0)  # Compute differences between consecutive points
    distances = np.sqrt((deltas ** 2).sum(axis=1))  # Euclidean distance between points
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)  # Cumulative sum of distances

    return cumulative_distances


def interpolate_coordinates(coords, distances, target_distances):
    """
    Interpolate the coordinates of a set of points to match target distances along a skeleton.

    This function linearly interpolates the x and y coordinates separately, ensuring that the
    points are distributed at the specified target distances along the skeleton.

    Parameters
    ----------
    coords : numpy.ndarray
        A 2D array of shape (n, 2) containing the x and y coordinates of the points to interpolate.
    
    distances : numpy.ndarray
        A 1D array of cumulative distances corresponding to the input coordinates.
    
    target_distances : numpy.ndarray
        A 1D array of target distances where the coordinates should be interpolated.

    Returns
    -------
    numpy.ndarray
        A 2D array of shape (n, 2) containing the interpolated x and y coordinates.
    """
    x_coords = np.interp(target_distances, distances, coords[:, 0])  # Interpolate x coordinates
    y_coords = np.interp(target_distances, distances, coords[:, 1])  # Interpolate y coordinates

    return np.vstack((x_coords, y_coords)).T  # Combine x and y coordinates


def extract_equidistant_coordinates(coords, num_points):
    """
    Extract equidistant coordinates along the skeleton of an object.

    This function calculates the cumulative distances along a skeleton, then interpolates
    the coordinates to generate a specified number of equidistant points along the skeleton.

    Parameters
    ----------
    coords : numpy.ndarray
        A 2D array of shape (n, 2) where each row contains the x and y coordinates of a point
        along the skeleton.

    num_points : int
        The number of equidistant points to extract from the skeleton.

    Returns
    -------
    tuple
        A tuple containing:
        - A 2D array of shape (num_points, 2) with the equidistant coordinates.
        - A float representing the total length of the skeleton.
    """
    cumulative_distances = calculate_cumulative_distances(coords)  # Get cumulative distances
    len_skel = cumulative_distances[-1]  # Total length of the skeleton
    target_distances = np.linspace(0, len_skel, num_points)  # Generate target distances
    equidistant_coords = interpolate_coordinates(coords, cumulative_distances, target_distances)  # Interpolate coordinates

    return equidistant_coords, len_skel


def get_points_generator(n_nodes, min_size_bacteria, max_size_bacteria):
    """
    Generate a function that extracts the coordinates of an object from a labeled mask.

    This function is customized for skeleton analysis and creates a version of the `get_points`
    function with specific parameters (`min_size_bacteria` and `n_nodes`) embedded, which can
    be used for detecting objects from a region mask.

    Parameters
    ----------
    n_nodes : int
        The number of points to sample along the skeleton of the object.

    min_size_bacteria : float
        The minimum size (in pixel) threshold for an object to be considered valid. Objects smaller than this size
        will be ignored.

    max_size_bacteria : float
        The maximum size (in pixel) threshold for an object to be considered valid. Objects larger than this size
        will be ignored.

    Returns
    -------
    function
        A function that, when given a region mask, returns the coordinates of the object, or NaN if
        the object is too small or invalid.
    """
    def get_points(regionmask):
        """
        Extract the properties of the skeleton from the region mask.

        This function performs skeletonization on the region mask, extracts coordinates along
        the skeleton, and samples the skeleton at equidistant points. It returns the coordinates
        and some additional properties such as the length and number of paths if the object meets
        the size threshold.

        Parameters
        ----------
        regionmask : numpy.ndarray
            A binary mask representing the object to analyze.

        Returns
        -------
        numpy.ndarray
            A 1D array containing the equidistant coordinates, skeleton length, and number of paths
            along the skeleton. If the object is too small, it returns NaN values.
        """
        skel = morphology.skeletonize(regionmask, method='lee')  # Perform skeletonization
        try:
            coords_skel = skan_skel(skel).path_coordinates(0)  # Extract path coordinates
            # if len(coords_skel) < n_nodes:  # Check if the skeleton is large enough
            #     return np.full(n_nodes * 2 + 2, np.nan)  # Return NaN if too small
        except:
            return np.full(n_nodes * 2 + 2, np.nan)  # Return NaN if skeleton extraction fails

        n_paths = skan_skel(skel).n_paths  # Number of paths in the skeleton
        equidistant_coords, len_skel = extract_equidistant_coordinates(coords_skel, n_nodes)  # Get equidistant coordinates

        if len_skel < min_size_bacteria:  # Check if the skeleton is below the minimum size
            return np.concatenate((np.full(n_nodes*2, np.nan), np.array([len_skel, n_paths])))  # Return NaN if below threshold
        
        if len_skel > max_size_bacteria:  # Check if the skeleton is below the minimum size
            return np.concatenate((np.full(n_nodes*2, np.nan), np.array([len_skel, n_paths])))  # Return NaN if below threshold

        return np.concatenate((equidistant_coords.flatten(), np.array([len_skel, n_paths])))  # Return the data

    return get_points


def detection(im_path, frame_id, min_size_bacteria, max_size_bacteria, columns_dict):
    """
    Detect objects in a labeled image and return their properties in a DataFrame.

    Parameters
    ----------
    im_path : str
        File path to the image sequence.
    frame_id : int
        Index of the frame to process.
    min_size_bacteria : float
        Minimum area threshold (in pixels) for valid objects.
    max_size_bacteria : float
        Maximum area threshold (in pixels) for valid objects.
    columns_dict : dict
        Dictionary with column name mappings.

    Returns
    -------
    pandas.DataFrame
        DataFrame with object properties.
    """
    n_nodes = len(columns_dict["x_nodes_columns"])  # Number of sampled points
    label_image = io.imread(im_path)  # Load the label image
    get_points = get_points_generator(n_nodes, min_size_bacteria, max_size_bacteria)

    # Extract region properties including the true seg_id via 'label'
    df_skel_nodes = pd.DataFrame(measure.regionprops_table(
        label_image=label_image,
        properties=['bbox', 'centroid', 'label', 'area'],
        extra_properties=(get_points,)
    ))

    # Adjust sampled node coordinates using bbox offset
    tmp = tl.gen_string_numbered(n=2 * n_nodes, str_name="get_points-")
    str_x = tmp[::2]
    str_y = tmp[1::2]
    df_skel_nodes.loc[:, str_x] += df_skel_nodes['bbox-0'].to_numpy()[:, None]
    df_skel_nodes.loc[:, str_y] += df_skel_nodes['bbox-1'].to_numpy()[:, None]

    # Add seg_id = pixel label value (for mask-based mapping)
    df_skel_nodes.rename(columns={"label": columns_dict["seg_id_column"]}, inplace=True)

    # Add frame index
    df_skel_nodes[columns_dict["t_column"]] = frame_id

    # Drop unnecessary bbox/centroid columns if not used
    df_skel_nodes = df_skel_nodes.iloc[:, 4:]

    # Rename and reorder columns
    df_skel_nodes.columns = columns_dict["columns_name_non_ordered"]
    df_skel_nodes = df_skel_nodes[columns_dict["columns_name"]]

    # Keep objects with only one path
    df_skel_nodes = df_skel_nodes[df_skel_nodes["n_paths"] == 1].copy()

    return df_skel_nodes


def run_detection(ims_all_paths, n_jobs, min_size_bacteria, max_size_bacteria, columns_dict):
    """
    Parrallelize the detection process
    
    """
    # Parallelization with joblib
    print("DETECTION")
    dfs = Parallel(n_jobs=n_jobs)(delayed(detection)(ims_all_paths[frame_id], frame_id, min_size_bacteria, max_size_bacteria, columns_dict) for frame_id in tqdm(range(0, len(ims_all_paths))))

    df = pd.concat(dfs)
    df.sort_values(by=[columns_dict["seg_id_column"], columns_dict["t_column"]], ignore_index=True, inplace=True)

    return df