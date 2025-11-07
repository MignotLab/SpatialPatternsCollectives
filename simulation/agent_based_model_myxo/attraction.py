"""
Author: Jean-Baptiste Saulnier
Date: 2024-12-06
"""
import numpy as np


class AttractionTypeError(Exception):
    """
    Custom exception raised when an invalid attraction type is specified.
    """
    pass


class Attraction:
    """
    This class calculates and applies attraction forces between bacteria in a 
    simulation. The attraction can be applied between the heads of bacteria, 
    between the body of one bacterium and its closest neighbor, or between 
    the heads and all neighbors. The class also provides a mechanism for handling 
    cases where no attraction is applied. The `attraction_type` parameter 
    controls these behaviors and can be set to one of the following options: 
    "attraction_head", "attraction_head_all_neighbours", "attraction_body", or 
    "no_attraction".

    Attributes:
    -----------
    par : object
        Instance of the Parameters class containing simulation settings such as attraction type and constants.
    gen : object
        Instance of the class managing bacterial data generation (e.g., bacterial positions).
    pha : object
        Instance of the class managing phantom data for simulation purposes.
    dir : object
        Instance of the class managing directional data for bacterial nodes.
    nei : object
        Instance of the class managing neighbour relationships and distances between bacteria.
    chosen_attraction_fonction : method
        The selected function for calculating the attraction force based on the `attraction_type` parameter.
    bact_enum : numpy.ndarray
        Array representing the enumeration of bacteria.
    bact_enum_body : numpy.ndarray
        Array representing the enumeration of the body parts of bacteria.
    """
    def __init__(self, inst_par, inst_gen, inst_pha, inst_dir, inst_nei):
        # Store references to external class instances
        self.par = inst_par
        self.gen = inst_gen
        self.pha = inst_pha
        self.dir = inst_dir
        self.nei = inst_nei

        # Select the appropriate attraction function based on the 'attraction_type' parameter
        if self.par.attraction_type == 'attraction_head':
            self.chosen_attraction_fonction = self.attraction_head
        elif self.par.attraction_type == 'attraction_head_all_neighbours':
            self.chosen_attraction_fonction = self.attraction_head_all_neighbours
        elif self.par.attraction_type == 'attraction_body':
            self.chosen_attraction_fonction = self.attraction_body
        elif self.par.attraction_type == 'no_attraction':
            self.chosen_attraction_fonction = self.function_doing_nothing
        else:
            # Raise an error and display a message if an invalid attraction type is specified
            print('attraction_type could be: "attraction_head", "attraction_head_all_neighbours", "attraction_body", "no_attraction"; default is "no_attraction"\n')
            raise AttractionTypeError()

        # Initialize arrays for bacterial enumeration (for body and overall)
        self.bact_enum = np.arange(self.par.n_bact).astype(self.par.int_type)
        self.bact_enum_body = np.arange(self.par.n_bact * self.par.n_nodes).astype(self.par.int_type)


    def function_attraction_type(self):
        """
        Executes the chosen attraction function based on the specified attraction type.
        
        This method is used to apply the selected attraction model (head, body, or none).
        """
        self.chosen_attraction_fonction()


    def f_at_norm(self, d, r):
        """
        Calculates the normalized attractive force between two objects at distance `r`.

        The force is 0 if the distance is greater than 2 times `d`, or if the force is less than `d`.

        Parameters:
        -----------
        d : float
            The characteristic length scale (e.g., the length of pili).
        r : numpy.ndarray
            The distances between bacteria.

        Returns:
        --------
        numpy.ndarray
            The normalized attractive forces.
        """
        x = np.minimum(np.maximum(r / d, 1), 2)

        return -self.par.k_a * (x - 1) * (x - 2) * (4 + (x - 1.5) * (24 - 16 * x))


    def f_at_pili_norm(self, r, k, w0, w1, w2):
        """
        Calculates the normalized attractive force between two objects at distance `r`, 
        with custom force characteristics based on the values `w0`, `w1`, and `w2`.

        Parameters:
        -----------
        r : numpy.ndarray
            The distances between objects.
        k : float
            The force constant.
        w0, w1, w2 : float
            The characteristic distances defining the force profile.

        Returns:
        --------
        numpy.ndarray
            The normalized attractive forces.
        """
        # Compute the coefficients of the force function based on the distances
        a = -((k * (w0**2 + 3*w1**2 - 3*w1*w2 + w2**2 + w0*(-3*w1 + w2))) / ((w0 - w1)**3 * (w1 - w2)**3))
        b = (k * (w0**2 * (3*w1 - w2) - w0 * (-3*w1 + w2)**2 + w1 * (8*w1**2 - 9*w1*w2 + 3*w2**2))) / ((w0 - w1)**3 * (w1 - w2)**3)
        c = -((k * (w0 * w1 * (-8*w1**2 + 9*w1*w2 - 3*w2**2) + w0**2 * (3*w1**2 - 3*w1*w2 + w2**2) + w1**2 * (6*w1**2 - 8*w1*w2 + 3*w2**2))) / ((w0 - w1)**3 * (w1 - w2)**3))

        # Calculate the force as a function of the distance
        res = (r - w0) * (r - w2) * (a * r**2 + b * r + c)
        res[r < w0] = 0
        res[r > w2] = 0

        return res
    

    def function_doing_nothing(self):
        """
        A placeholder function that does nothing, used for the "no_attraction" case.
        """
        pass


    def attraction_head(self):
        """
        Applies attraction between the head of each bacterium and its closest neighbour.
        """
        # Extract direction data for the nearest neighbour
        x_dir_n = self.dir.nodes_to_nei_dir_torus[0, :self.par.n_bact, :]
        y_dir_n = self.dir.nodes_to_nei_dir_torus[1, :self.par.n_bact, :]

        # Compute the attraction based on the head and neighbour
        xy_dir = np.reshape(np.repeat(self.dir.nodes_direction[:, 0, :], self.par.kn), (2, self.par.n_bact, self.par.kn))
        cond_angle_view = (xy_dir[0] * x_dir_n + xy_dir[1] * y_dir_n) > np.cos(self.par.at_angle_view / 2)
        
        # Compute the normalized force based on pili length and neighbor distance
        f_norm = self.f_at_pili_norm(r=self.nei.dist[:self.par.n_bact], k=self.par.k_a, w0=self.par.width, w1=(self.par.pili_length + self.par.width) / 2, w2=self.par.pili_length)
        f_norm[self.nei.cond_same_bact[:self.par.n_bact] | ~cond_angle_view] = 0

        # Find the closest non-zero neighbor
        ind_first_non_zero = (f_norm != 0).argmax(axis=1)
        f_norm = f_norm[self.bact_enum, ind_first_non_zero]

        # Calculate the attraction force and apply it
        f_x = x_dir_n[self.bact_enum, ind_first_non_zero] * f_norm
        f_y = y_dir_n[self.bact_enum, ind_first_non_zero] * f_norm
        force = np.array([f_x, f_y])

        # Apply the force to the bacterial data and phantom data
        self.gen.data[:, 0, :] += force * self.par.dt
        self.pha.data_phantom[:, 0, :] += force * self.par.dt


    def attraction_head_all_neighbours(self):
        """
        Applies attraction between the head of each bacterium and all its neighbours.
        """
        # Extract direction data for all neighbours
        x_dir_n = self.dir.nodes_to_nei_dir_torus[0, :, :]
        y_dir_n = self.dir.nodes_to_nei_dir_torus[1, :, :]

        # Compute the attraction for all neighbouring bacteria
        xy_dir = np.reshape(np.repeat(self.dir.nodes_direction[:, 0, :], self.par.kn), (2, self.par.n_bact, self.par.kn))
        cond_angle_view = (xy_dir[0] * x_dir_n + xy_dir[1] * y_dir_n) > np.cos(self.par.at_angle_view / 2)
        
        f_norm = self.f_at_pili_norm(r=self.nei.dist[:self.par.n_bact], k=self.par.k_a, w0=self.par.width, w1=(self.par.pili_length + self.par.width) / 2, w2=self.par.pili_length)
        f_norm[~cond_angle_view] = 0

        # Sum forces over all neighbours and apply the force
        f_x = np.sum(x_dir_n * f_norm, axis=1)
        f_y = np.sum(y_dir_n * f_norm, axis=1)
        force = np.array([f_x, f_y])

        # Apply the force to the bacterial data and phantom data
        self.gen.data[:, 0, :] += force * self.par.dt
        self.pha.data_phantom[:, 0, :] += force * self.par.dt


    def attraction_body(self):
        """
        Attraction of the entire body of the bacterium with its closest neighbor and force application along the body.
        """
        # Direction data for all neighbors
        x_dir_n = self.dir.nodes_to_nei_dir_torus[0, :, :]
        y_dir_n = self.dir.nodes_to_nei_dir_torus[1, :, :]

        # Flatten the node directions to apply force calculations across the body
        nodes_direction_flat = np.reshape(self.dir.nodes_direction, (2, self.par.n_bact * self.par.n_nodes))
        xy_dir = np.reshape(np.repeat(nodes_direction_flat, self.par.kn), (2, self.par.n_bact * self.par.n_nodes, self.par.kn))

        # Create the condition for the angle view of the force calculation
        cond_angle_view = (xy_dir[0] * x_dir_n + xy_dir[1] * y_dir_n) > np.cos(self.par.at_angle_view / 2)

        # Compute the normalized attraction force using the function for head-to-head attraction
        f_norm = self.f_at_norm(d=self.par.width, r=self.nei.dist)
        f_norm[self.nei.cond_same_bact | ~cond_angle_view] = 0.  # Set force to 0 for same bacteria or out-of-view angles

        # Find the closest neighbor with non-zero force values (set to 0 if no valid neighbor)
        ind_first_non_zero = (f_norm != 0).argmax(axis=1)
        f_norm = f_norm[self.bact_enum_body, ind_first_non_zero]

        # Reshape direction data and apply it based on the closest neighbor
        f_x = x_dir_n[self.bact_enum_body, ind_first_non_zero] * f_norm
        f_y = y_dir_n[self.bact_enum_body, ind_first_non_zero] * f_norm

        # Create the attraction force array for all points along the bacteria
        f_att = np.array([np.reshape(f_x, (self.par.n_nodes, self.par.n_bact)), np.reshape(f_y, (self.par.n_nodes, self.par.n_bact))])

        # Calculate the parallel component of the force (along the direction of the bacteria)
        f_att_norm_par = np.sum(f_att[:, :, :] * self.dir.nodes_direction[:, :, :], axis=0)
        f_att_par = f_att_norm_par[:, :] * self.dir.nodes_direction[:, :, :]

        # Compute the perpendicular force and normalize it based on the total force
        f_att_per = f_att[:, :, :] - f_att_par[:, :, :]
        f_att_norm_per = np.linalg.norm(f_att_per, axis=0)
        f_att_norm_per[f_att_norm_per == 0] = np.inf  # Avoid division by zero
        f_att_per = f_att_per / f_att_norm_per * np.linalg.norm(f_att, axis=0)

        # Apply all the attraction force on the head and only the perpendicular force on the body
        self.gen.data[:, 0, :] += f_att[:, 0, :] * self.par.dt
        self.pha.data_phantom[:, 0, :] += f_att[:, 0, :] * self.par.dt
        self.gen.data[:, 1:, :] += f_att_per[:, 1:, :] * self.par.dt
        self.pha.data_phantom[:, 1:, :] += f_att_per[:, 1:, :] * self.par.dt


