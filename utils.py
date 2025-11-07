"""
Author: Jean-Baptiste Saulnier
Date: 2024-11-21

This module contains utility functions for various operations like 
generating coordinate strings, file handling, and vector operations.
It is used internally in the project to simplify common tasks.

"""
import os
import re
import csv

from collections.abc import Iterable
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def gen_coord_str(n: int, xy: bool = True) -> list[str] | tuple[list[str], list[str]]:
    """
    Generate a list of coordinate string names or tuples of lists of coordinate strings.

    If `xy=True`, generates a list like ['x0', 'y0', ..., 'xn', 'yn'].
    If `xy=False`, generates a tuple of two lists: 
    - First list: ['x0', 'x1', ..., 'xn']
    - Second list: ['y0', 'y1', ..., 'yn']

    Parameters
    ----------
    n : int
        The number of coordinate pairs (x, y).

    xy : bool, optional
        If True, return a flattened list of strings representing x and y coordinates. 
        If False, return a tuple of two lists (x and y coordinates separately).
        Default is True.

    Returns
    -------
    list or tuple of lists
        A list of coordinate strings or a tuple of two lists of coordinate strings.

    Examples
    --------
    >>> gen_coord_str(3)
    ['x0', 'y0', 'x1', 'y1', 'x2', 'y2']

    >>> gen_coord_str(3, xy=False)
    (['x0', 'x1', 'x2'], ['y0', 'y1', 'y2'])
    """
    if xy:
        name = []
        for i in range(n):
            name.append('x'+str(i))
            name.append('y'+str(i))

        return name

    else:
        name_x = []
        name_y = []
        for i in range(n):
            name_x.append('x'+str(i))
            name_y.append('y'+str(i))

        return name_x, name_y
        

def gen_string_numbered(n: int, str_name: str) -> list[str]:
    """
    Generate a list of strings with a given base name followed by an index, e.g., ['str_name0', ..., 'str_namen'].

    Parameters
    ----------
    n : int
        The number of strings to generate.
    
    str_name : str
        The base name for each string.

    Returns
    -------
    list
        A list of numbered strings.

    Examples
    --------
    >>> gen_string_numbered(3, "file")
    ['file0', 'file1', 'file2']
    """
    res = []
    
    for i in range(n):
        res += [str_name + str(i)]

    return res


def initialize_directory_or_file(path: str, columns: Optional[list[str]] = None) -> None:

    """
    Create the parent directories specified in the path (if they don't already exist).
    If the path contains a file name, create the file or reset it if it already exists.
    If the path does not contain a file name, create only the parent directories.

    Parameters
    ----------
    path : str
        The full path of the file or directories to initialize.

    columns : list or None, optional
        A list of column names to write as the header if the path contains a file name.
        Default is None, which means no header will be written.

    Returns
    -------
    None

    Notes
    -----
    This function creates the parent directories for the specified path, and if the path
    includes a file name (with extension), it will create or reset the file.

    Examples
    --------
    >>> initialize_directory_or_file("data/images/image.jpg")
    # Creates the 'data/images' directory if it doesn't exist and creates/reset the 'image.jpg' file.

    >>> initialize_directory_or_file("data/results/")
    # Creates the 'data/results' directory if it doesn't exist.

    >>> initialize_directory_or_file("data/data.csv", columns=["Name", "Age", "City"])
    # Creates/reset the 'data.csv' file with the specified columns as header.
    """
    # Get the parent directory of the path
    folder_path = os.path.dirname(path)

    # Check if the parent directory exists, if not, create it
    if folder_path and not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Check if the path contains a file name
    if os.path.splitext(path)[1]:  # If the path has an extension (it's a file)
        # Open the file in write mode and write the columns as header if provided
        with open(path, 'w', newline='') as file:
            if columns:
                header = ",".join(columns)
                file.write(header + '\n')


def save_dataframe(df: pd.DataFrame, path: str) -> None:
    """
    Save a Pandas DataFrame as a CSV file.

    This function first ensures the directory and file are initialized (or reset) before saving
    the DataFrame to the specified path.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be saved.

    path : str
        The file path where the DataFrame will be saved.

    Returns
    -------
    None
    """
    initialize_directory_or_file(path)
    df.to_csv(path, header=True, index=False)


def py_ang(v1: Sequence[float], v2: Sequence[float]) -> float:
    """
    Compute the angle between two 2D vectors `v1` and `v2`.

    The angle is returned in radians, within the range [-pi, pi].

    Parameters
    ----------
    v1 : array-like or list
        The first vector [x1, y1].

    v2 : array-like or list
        The second vector [x2, y2].

    Returns
    -------
    angle : float
        The angle between the vectors in radians.

    Examples
    --------
    >>> py_ang([1, 0], [0, 1])
    1.5707963267948966  # pi/2 radians (90 degrees)
    """
    dot = v1[0]*v2[0] + v1[1]*v2[1]      # dot product
    det = v1[0]*v2[1] - v1[1]*v2[0]      # determinant
    angle = np.arctan2(det, dot)

    return angle


def _extract_number(filename: str) -> int | float:
    """
    Extract the first number found in a filename.

    This function searches for the first sequence of digits in the filename and returns it as an integer.
    If no number is found, it returns infinity.

    Parameters
    ----------
    filename : str
        The filename from which the number will be extracted.

    Returns
    -------
    int
        The first number found in the filename, or infinity if no number is found.

    Examples
    --------
    >>> extract_number("image_42.png")
    42

    >>> extract_number("file_no_number.txt")
    inf
    """
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')


def list_files_in_directory(directory_path: str, ext: str = ".tif") -> list[str]:
    """
    List all files in a directory with a specific extension and return their full paths, 
    sorted by the numbers extracted from their names.

    Parameters
    ----------
    directory_path : str
        The path to the directory from which files will be listed.
    ext : str, optional
        File extension to filter by (default: '.tif').

    Returns
    -------
    list of str
        A sorted list of file paths matching the given extension.

    Examples
    --------
    >>> list_files_in_directory("data/images/", ext=".tif")
    ['data/images/image_1.tif', 'data/images/image_2.tif', 'data/images/image_10.tif']
    """
    files = [
        os.path.join(directory_path, filename)
        for filename in os.listdir(directory_path)
        if os.path.isfile(os.path.join(directory_path, filename)) and filename.lower().endswith(ext)
    ]
    return sorted(files, key=lambda f: _extract_number(os.path.basename(f)))


def compute_trajectory_length(
    df: pd.DataFrame,
    col_id: str,
    col_traj_length: str,
    inplace: bool = False
    ) -> None | pd.DataFrame:
    """
    Compute the length of trajectories in a tracking dataframe and add a specific column.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the tracking information.
    col_id : str
        The column name that stores the unique IDs of trajectories.
    col_traj_length : str
        The column name where the computed trajectory lengths will be stored.
    inplace : bool, optional
        If True, the operation modifies the input dataframe directly. 
        If False, a modified copy of the dataframe is returned (default is False).

    Returns:
    --------
    pandas.DataFrame or None
        - If `inplace` is False, returns a modified copy of the dataframe with the new column added.
        - If `inplace` is True, modifies the dataframe directly and returns None.
    """
    # Compute trajectory lengths
    traj_lengths = df[col_id].value_counts()

    # Map each ID to its trajectory length
    if inplace:
        df[col_traj_length] = df[col_id].map(traj_lengths)
    else:
        df_copy = df.copy()
        df_copy[col_traj_length] = df_copy[col_id].map(traj_lengths)
        return df_copy
    

def rotation_matrix_2d(theta: float | np.ndarray) -> np.ndarray:
    """
    Compute a 2D rotation matrix for changing the direction of bacteria.
    
    This function generates a rotation matrix based on the input angle `theta`. 
    It can be used to apply a rotation to a 2D vector, changing its direction 
    by the specified angle in radians.

    Parameters:
    -----------
    theta : float or numpy.ndarray
        The angle of rotation in radians. This can be a single value or an array 
        of values for batch rotations.

    Returns:
    --------
    numpy.ndarray
        A 2x2 rotation matrix or an array of rotation matrices, depending on the 
        shape of `theta`. The matrix is structured as:
            [[cos(theta), -sin(theta)],
             [sin(theta),  cos(theta)]]
    """
    tmp = np.array([[np.cos(theta), -np.sin(theta)], 
                    [np.sin(theta),  np.cos(theta)]])
    
    return tmp


def append_to_csv(filename: str, data: Iterable[Iterable[Any]]) -> None:
    """
    Append data to a CSV file.

    This function appends rows of data to an existing CSV file. If the file does not 
    exist, it will be created. The input data should be structured as an iterable 
    of rows, where each row is a list of values.

    Parameters:
    -----------
    filename : str
        The path to the CSV file where data will be appended. If the file does not 
        exist, it will be created.
    
    data : iterable of iterables
        The data to be written to the CSV file. Each element in `data` should be 
        an iterable representing a row of values.

    Returns:
    --------
    None
        This function does not return any value. It writes data directly to the file.
    """
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)


def nearest_even(number: float) -> int:
    """
    Returns the nearest even integer to a given number.

    Parameters:
    - number (float or int): The input number for which the nearest even integer is calculated.

    Returns:
    - int: The nearest even integer.
    """
    # Step 1: Round the number to the nearest integer.
    # This ensures we start with an integer close to the input number.
    rounded_number = round(number)
    
    # Step 2: Check if the rounded number is even.
    # If it is even, return it directly.
    if rounded_number % 2 == 0:
        return rounded_number
    
    # Step 3: If the rounded number is odd, adjust it to the nearest even number.
    # - If the original number is greater than the rounded number, add 1 to make it even.
    # - If the original number is less than the rounded number, subtract 1 to make it even.
    if number > rounded_number:
        return rounded_number + 1  # Adjust up
    else:
        return rounded_number - 1  # Adjust down


def generate_directions_from_matrix_centers(n: int) -> Tuple[np.ndarray, np.ndarray]:
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