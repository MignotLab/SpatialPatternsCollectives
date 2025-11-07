"""
Author: Jean-Baptiste Saulnier
Date: 2024-12-06
"""
import numpy as np
import matplotlib.pyplot as plt

import utils as tl


class Kymograph:
    """
    This class creates and analyzes kymographs based on bacterial positions and density 
    over time in a 2D simulation space. It includes functionality for saving positions, 
    computing density profiles, and extracting trajectories within defined rectangular 
    regions of interest. The class also provides tools to visualize these kymographs.

    Attributes:
    -----------
    par : object
        Instance of a parameter class containing simulation constants (e.g., number of nodes, 
        bacteria, and kymograph settings).
    gen : object
        Instance of a class managing bacterial data (e.g., positions over time).
    T : int
        Total simulation time.
    id_save_node : int
        Index of the node to save in the middle of the bacterium's body.
    save : numpy.ndarray
        Boolean array to track which bacteria's data should be saved.
    position_array : numpy.ndarray
        Array to store positions of bacteria at each save interval.
    time_array : numpy.ndarray
        Array to store the corresponding time steps for saved positions.
    count : int
        Counter for density kymograph updates.
    hist_kymo : numpy.ndarray or None
        2D histogram for intermediate density calculations.
    bins_kymo : int
        Number of bins for the density histogram along the spatial axis.
    density_kymo : numpy.ndarray
        2D array representing the density kymograph (time vs. spatial bins).
    """
    def __init__(self, inst_par, inst_gen, T):
        """
        Initialize the kymograph class and its attributes.

        Parameters:
        -----------
        inst_par : object
            Instance of a parameter class.
        inst_gen : object
            Instance of a data generator class for bacterial positions.
        T : int
            Total simulation time.
        """
        self.par = inst_par
        self.gen = inst_gen
        self.T = T
        self.id_save_node = int(self.par.n_nodes / 2)  # Use the central node for measurements.
        
        # Boolean array to determine if a bacterium's data should be saved.
        self.save = np.zeros(self.par.n_bact).astype(bool)
        
        # Initialize position and time storage for kymograph.
        self.position_array = np.zeros((2, int(self.T / self.par.save_freq_kymo), self.par.n_bact), dtype=self.par.float_type)
        self.time_array = np.tile(np.arange(0, self.T, self.par.save_freq_kymo).astype(self.par.float_type), (self.par.n_bact, 1)).T
        
        # Variables for density kymograph computation.
        self.count = 0
        self.hist_kymo = None
        self.bins_kymo = self.par.space_size  # Number of spatial bins for density histogram.
        self.density_kymo = np.zeros((int(self.T / self.par.save_freq_kymo + 2), self.bins_kymo), dtype=self.par.float_type)


    def build_kymograph_density(self, index, save_kymo):
        """
        Calculate the density kymograph by slicing the simulation space along the x-axis.

        Parameters:
        -----------
        index : int
            Current simulation time step.
        save_kymo : bool
            Flag to indicate whether the kymograph should be saved at this time step.
        """
        if save_kymo and (index % int(1 / self.par.dt * self.par.save_freq_kymo) == 0):
            # Define a slice in the y-direction for counting bacteria.
            cond_slice = (self.gen.data[1, self.id_save_node, :] > self.par.space_size / 2 - self.par.slice_width / 2) & \
                         (self.gen.data[1, self.id_save_node, :] < self.par.space_size / 2 + self.par.slice_width / 2)
            
            # Compute a 2D histogram of positions within the slice.
            self.hist_kymo, __, __ = np.histogram2d(
                self.gen.data[1, self.id_save_node, cond_slice],
                self.gen.data[0, self.id_save_node, cond_slice],
                bins=self.bins_kymo
            )
            
            # Aggregate along the y-axis to compute density per spatial bin.
            self.density_kymo[self.count, :] = np.sum(self.hist_kymo, axis=0)
            self.count += 1


    def save_position(self, index):
        """
        Save the positions of bacteria at the current time step if it matches the save frequency.

        Parameters:
        -----------
        index : int
            Current simulation time step.
        """
        if index % int(1 / self.par.dt * self.par.save_freq_kymo) == 0:
            self.position_array[:, int(index * self.par.dt / self.par.save_freq_kymo), :] = self.gen.data[:, self.id_save_node, :]


    def rectangle(self, axis_coord, width):
        """
        Compute the coordinates of the four corners of a rectangle perpendicular to a given axis.

        Parameters:
        -----------
        axis_coord : numpy.ndarray
            Coordinates of the two endpoints of the rectangle's central axis.
        width : float
            Width of the rectangle.

        Returns:
        --------
        Tuple of floats
            Coordinates of the four corners of the rectangle.
        """
        axis_vector = axis_coord[1, :] - axis_coord[0, :]
        axis_vector = axis_vector / np.linalg.norm(axis_vector)  # Normalize the central axis vector.
        rotation_matrix = tl.rotation_matrix_2d(theta=np.pi / 2)  # Compute a perpendicular direction.
        perp_axis_vector = np.sum(rotation_matrix * axis_vector, axis=1)
        
        # Compute the four corners.
        x1, y1 = perp_axis_vector * width / 2 + axis_coord[0, :]
        x2, y2 = -perp_axis_vector * width / 2 + axis_coord[0, :]
        x3, y3 = -perp_axis_vector * width / 2 + axis_coord[1, :]
        x4, y4 = perp_axis_vector * width / 2 + axis_coord[1, :]

        return x1, y1, x2, y2, x3, y3, x4, y4


    def triangle_area(self, x1, y1, x2, y2, x3, y3):
        """
        Calculate the area of a triangle given its vertices.

        Parameters:
        -----------
        x1, y1, x2, y2, x3, y3 : float
            Coordinates of the triangle vertices.

        Returns:
        --------
        float
            Area of the triangle.
        """
        return np.abs((x1 * (y2 - y3) +
                       x2 * (y3 - y1) +
                       x3 * (y1 - y2)) / 2.0)


    def inside_rectangle_check(self, X, x, y):
        """
        Check if a point lies inside a given rectangle.

        Parameters:
        -----------
        X : numpy.ndarray
            Coordinates of the rectangle's four corners.
        x, y : numpy.ndarray
            Coordinates of the point to check.

        Returns:
        --------
        numpy.ndarray
            Boolean array indicating whether each point lies inside the rectangle.
        """
        A = (self.triangle_area(X[0], X[1], X[2], X[3], X[4], X[5]) +
             self.triangle_area(X[0], X[1], X[6], X[7], X[4], X[5]))
        A1 = self.triangle_area(x, y, X[0], X[1], X[2], X[3])
        A2 = self.triangle_area(x, y, X[2], X[3], X[4], X[5])
        A3 = self.triangle_area(x, y, X[4], X[5], X[6], X[7])
        A4 = self.triangle_area(x, y, X[0], X[1], X[6], X[7])
        
        eps = 1e-3  # Allow small numerical errors.
        return ((A < A1 + A2 + A3 + A4 + eps) & (A > A1 + A2 + A3 + A4 - eps))


    def position_inside_rectangle(self, position_array, X, axis_coord):
        """
        Identify and save the positions of cells located within a defined rectangle. 
        Project their positions onto the rectangle's main axis and group these into trajectories.

        Parameters:
        - position_array: ndarray, positions of cells in the simulation space (shape: [2, T, n_bact]).
        - X: ndarray, coordinates of the four corners of the rectangle.
        - axis_coord: ndarray, start and end points of the rectangle's main axis.

        Returns:
        - trajectories: list, lists of projected positions of cells along the rectangle's axis, grouped by trajectory.
        - times: list, lists of corresponding time points for each trajectory.
        """
        # Check which cells are inside the rectangle at each time step.
        cond_inside_rectangle = self.inside_rectangle_check(X=X, x=position_array[0, :, :], y=position_array[1, :, :])
        
        # Compute the length of the main axis of the rectangle.
        norm_axis = np.linalg.norm(axis_coord[1] - axis_coord[0])

        # Project cell positions onto the rectangle's main axis.
        position_projection_x = (position_array[0, :, :] - axis_coord[0, 0]) * (axis_coord[1, 0] - axis_coord[0, 0])
        position_projection_y = (position_array[1, :, :] - axis_coord[0, 1]) * (axis_coord[1, 1] - axis_coord[0, 1])
        position_projection = (position_projection_x + position_projection_y) / norm_axis

        # Mark projections as NaN for cells outside the rectangle.
        position_projection[~cond_inside_rectangle] = np.nan

        # Group trajectories for each cell based on continuity and validity of projections.
        trajectories = []
        times = []
        for i in range(self.par.n_bact):
            trajectory_i = []
            time_i = []
            count = 1
            while count < int(self.T / self.save_freq):
                cond1 = ~np.isnan(position_projection[count, i])  # Check if the projection is valid.
                cond2 = np.abs(position_projection[count, i] - position_projection[count - 1, i]) < (self.save_freq * self.par.v0) * 2
                # Check if the projection change is within allowed limits.
                if cond1 & cond2:
                    trajectory_i.append(position_projection[count, i])
                    time_i.append(self.time_array[count, i])
                    count += 1
                else:
                    # End the current trajectory and start a new one.
                    trajectories.append(trajectory_i)
                    times.append(time_i)
                    trajectory_i = []
                    time_i = []
                    count += 1
            # Append the last trajectory for the cell.
            trajectories.append(trajectory_i)
            times.append(time_i)
        
        return trajectories, times