"""
Author: Jean-Baptiste Saulnier
Date: 2024-12-06
"""
import numpy as np


class AlignmentTypeError(Exception):
    """
    Custom exception raised when an invalid alignment type is specified.
    """
    pass


class Alignment:
    """
    This class manages the alignment of bacterial nodes based on predefined 
    alignment types. The alignment behavior is controlled by the 
    `alignment_type` parameter, which can be set to options such as
    "head_alignment", "global_alignment", or "head_alignment_and_global_alignment".

    Attributes:
    -----------
    par : object
        Instance of the `Parameters` class containing alignment settings.
    gen : object
        Instance of the class managing bacterial data generation.
    pha : object
        Instance of the class managing phantom data for simulation purposes.
    dir : object
        Instance of the class managing directional data for bacterial nodes.
    nei : object
        Instance of the class managing neighbor-related calculations.
    uti : object
        Instance of a utility class providing helper functions.
    chosen_alignment_fonction : method
        Selected alignment function based on the `alignment_type` parameter.
    nodes_head_enum : np.ndarray
        Array of integers representing the indices of head nodes for alignment calculations.
    """
    def __init__(self, inst_par, inst_gen, inst_pha, inst_dir, inst_nei, inst_uti):
        # Store references to external class instances
        self.par = inst_par
        self.gen = inst_gen
        self.pha = inst_pha
        self.dir = inst_dir
        self.nei = inst_nei
        self.uti = inst_uti

        # Select the appropriate alignment function based on the alignment type
        if self.par.alignment_type == 'head_alignment':
            self.chosen_alignment_fonction = self.head_alignment
        elif self.par.alignment_type == 'global_alignment':
            self.chosen_alignment_fonction = self.global_alignment
        elif self.par.alignment_type == 'head_alignment_and_global_alignment':
            self.chosen_alignment_fonction = self.head_alignment_and_global_alignment
        elif self.par.alignment_type == 'no_alignment':
            self.chosen_alignment_fonction = self.function_doing_nothing
        else:
            # Raise an error for invalid alignment types
            print(
                'alignment_type could be: "head_alignment", "global_alignment", '
                '"head_alignment_and_global_alignment", "no_alignment"; default is "no_alignment"\n'
            )
            raise AlignmentTypeError()

        # Generate indices for the head nodes of each bacterium
        self.nodes_head_enum = np.arange(self.par.n_bact).astype(self.par.int_type)


    def function_alignment_type(self):
        """
        Execute the chosen alignment function.
        """
        self.chosen_alignment_fonction()


    def nodes_angle(self, x, y):
        """
        Calculate the angle of nodes in the range [-π, π].

        Parameters:
        -----------
        x : np.ndarray
            X-coordinates of the direction vectors.
        y : np.ndarray
            Y-coordinates of the direction vectors.

        Returns:
        --------
        np.ndarray
            Array of angles corresponding to the input vectors.
        """
        return np.arctan2(y, x)


    def function_doing_nothing(self):
        """
        Placeholder function for cases where no alignment is performed.
        """
        pass


    def head_alignment(self):
        """
        Align the head of each bacterium to the direction of its closest neighbor.

        This function calculates the relative direction of the closest neighbor for each bacterium
        and adjusts the angle of the bacteria head to align with it.
        """
        # Copy neighbor directions
        nei_dir = self.dir.neighbours_direction[:, :self.par.n_bact, :].copy()

        # Apply angle view condition
        xy_dir = np.reshape(
            np.repeat(self.dir.nodes_direction[:, 0, :], self.par.kn),
            (2, self.par.n_bact, self.par.kn)
        )
        cond_angle_view = (
            xy_dir[0] * self.dir.nodes_to_nei_dir_torus[0, :self.par.n_bact, :] +
            xy_dir[1] * self.dir.nodes_to_nei_dir_torus[1, :self.par.n_bact, :]
        ) > np.cos(self.par.at_angle_view / 2)

        # Apply distance condition
        cond_dist = self.nei.dist[:self.par.n_bact] > self.par.max_align_dist
        nei_dir[:, self.nei.cond_same_bact[:self.par.n_bact] | cond_dist | ~cond_angle_view] = 0.

        # Identify the closest neighbor for alignment
        ind_first_non_zero = ((nei_dir[0] != 0) | (nei_dir[1] != 0)).argmax(axis=1)
        nei_dir = nei_dir[:, self.nodes_head_enum, ind_first_non_zero]

        # Compute angles of nodes and their closest neighbors
        neighbours_angle_head = self.nodes_angle(x=nei_dir[0, :], y=nei_dir[1, :])
        nodes_angle_head = self.nodes_angle(
            x=self.dir.nodes_direction[0, 0, :],
            y=self.dir.nodes_direction[1, 0, :]
        )

        # Calculate rotation angle for alignment
        angle = self.par.j_t * np.sin(2 * (neighbours_angle_head - nodes_angle_head)) * self.par.dt

        # Set rotation angle to 0 for nodes with no close neighbors
        cond_no_neighbours = (nei_dir[0, :] == 0.) & (nei_dir[1, :] == 0.)
        angle[cond_no_neighbours] = 0.

        # Construct rotation matrix and update positions
        rotation_matrix = self.uti.rotation_matrix_2d(theta=angle)
        self.gen.data[:, 0, :] = (
            np.sum(rotation_matrix * (self.pha.data_phantom[:, 0, :] - self.pha.data_phantom[:, 1, :]), axis=1) +
            self.gen.data[:, 1, :]
        )
        self.pha.data_phantom[:, 0, :] = (
            np.sum(rotation_matrix * (self.pha.data_phantom[:, 0, :] - self.pha.data_phantom[:, 1, :]), axis=1) +
            self.pha.data_phantom[:, 1, :]
        )


    def global_alignment(self):
        """
        Align cells in a specific region to a global direction.

        This function rotates bacteria in a predefined direction specified by `global_angle`.
        It applies to bacteria that are within a certain alignment space (`cond_space_alignment`).
        """
        nodes_angle_head = self.nodes_angle(
            x=self.dir.nodes_direction[0, 0, self.gen.cond_space_alignment],
            y=self.dir.nodes_direction[1, 0, self.gen.cond_space_alignment]
        )
        angle = self.par.j_t * np.sin(2 * (self.par.global_angle - nodes_angle_head)) * self.par.dt
        rotation_matrix = self.uti.rotation_matrix_2d(theta=angle)

        # Update positions of aligned nodes
        self.gen.data[:, 0, self.gen.cond_space_alignment] = np.sum(
            rotation_matrix * (self.pha.data_phantom[:, 0, self.gen.cond_space_alignment] -
                               self.pha.data_phantom[:, 1, self.gen.cond_space_alignment]),
            axis=1
        ) + self.gen.data[:, 1, self.gen.cond_space_alignment]
        self.pha.data_phantom[:, 0, self.gen.cond_space_alignment] = np.sum(
            rotation_matrix * (self.pha.data_phantom[:, 0, self.gen.cond_space_alignment] -
                               self.pha.data_phantom[:, 1, self.gen.cond_space_alignment]),
            axis=1
        ) + self.pha.data_phantom[:, 1, self.gen.cond_space_alignment]


    def head_alignment_and_global_alignment(self):
        """
        Combine head alignment and global alignment functions.
        """
        self.head_alignment()
        self.global_alignment()
