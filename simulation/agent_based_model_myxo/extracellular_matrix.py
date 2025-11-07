"""
Author: Jean-Baptiste Saulnier
Date: 2024-12-06
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import convolve
np.random.seed(42)  # Fixe la graine


class EcmTypeError(Exception):
    pass


class Ecm:
    """
    This class simulates the generation and behavior of EPS 
    (extracellular polymeric substances) as bacteria move across the environment. 
    The type of EPS follower behavior is determined by the `eps_follower_type` 
    parameter. Possible values include: "igoshin_eps_road_follower", 
    "follow_gradient", "no_eps" (default is "no_eps").

    Attributes:
    -----------
    par : object
        Instance of the Parameters class containing simulation settings such as EPS follower type, space size, and other related constants.
    gen : object
        Instance of the class managing bacterial data generation, such as bacterial positions and conditions.
    pha : object
        Instance of the class managing phantom data for simulation purposes (e.g., bacterial shapes and projections).
    dir : object
        Instance of the class managing directional data for bacterial nodes.
    nei : object
        Instance of the class managing neighbor relationships and distances between bacteria.
    ali : object
        Instance of the class managing alignment data, which controls bacterial alignment behavior.
    uti : object
        Instance of the class containing utility functions for geometry operations, such as rotation matrices.
    chosen_eps_follower_fonction : method
        The selected function for calculating EPS following behavior, based on the `eps_follower_type` parameter.
    angle_section : float
        The angle of view for each EPS section, derived from the total EPS angle view divided by the number of sections.
    l : float
        The length of the space, as specified in the simulation parameters.
    edges_width : float
        The width of the EPS edges in micrometers, calculated as twice the pili length.
    l_eps : float
        The total EPS space length in micrometers, accounting for both the space size and the EPS edges.
    bins : int
        The number of bins used in the EPS grid for discretizing the EPS space.
    edges_width_bins : int
        The bin width corresponding to the EPS edge, in terms of the EPS grid.
    r : int
        The radius of influence (in pixels) for EPS interactions, calculated based on the pili length.
    g_x : numpy.ndarray
        The 2D grid of X coordinates for the EPS space.
    g_y : numpy.ndarray
        The 2D grid of Y coordinates for the EPS space (transpose of `g_x`).
    rate_eps_evaporation : float
        The rate at which EPS evaporates, determined by the EPS mean lifetime.
    eps_grid : numpy.ndarray
        The 2D grid representing the amount of EPS at each point in the space.
    eps_grid_blur : numpy.ndarray
        A blurred version of the EPS grid, used for smoothing and calculating local EPS values.
    eps_angle : numpy.ndarray
        The array of angles representing the direction of EPS following for each bacterium.
    bact_angle : numpy.ndarray
        The array of angles representing the direction of each bacterium.
    eps_diff : numpy.ndarray
        The array of EPS diffusion values for each bacterium, controlling how EPS diffuses across space.
    """
    
    def __init__(self, inst_par, inst_gen, inst_pha, inst_dir, inst_nei, inst_ali, inst_uti):
        # Store references to external instances
        self.par = inst_par
        self.gen = inst_gen
        self.pha = inst_pha
        self.dir = inst_dir
        self.nei = inst_nei
        self.ali = inst_ali
        self.uti = inst_uti

        # Select the appropriate EPS follower function based on the 'eps_follower_type' parameter
        if self.par.eps_follower_type == 'igoshin_eps_road_follower_new_test':
            self.chosen_eps_follower_fonction = self.igoshin_eps_road_follower_new_test
        elif self.par.eps_follower_type == 'igoshin_eps_road_follower':
            self.chosen_eps_follower_fonction = self.igoshin_eps_road_follower
        elif self.par.eps_follower_type == 'follow_gradient':
            self.chosen_eps_follower_fonction = self.follow_eps_gradient
        elif self.par.eps_follower_type == 'no_eps':
            self.chosen_eps_follower_fonction = self.function_doing_nothing
        else:
            # Raise an error and display a message if an invalid EPS type is specified
            print('eps_follower_type parameter could be: "igoshin_eps_road_follower", "follow_gradient", "no_eps"; default is "no_eps"\n')
            raise EcmTypeError()
        
        # Select the appropriate prey follower function based on the 'eps_follower_type' parameter
        if self.par.prey_follower_type == 'follow_gradient':
            self.chosen_prey_follower_fonction = self.follow_prey_gradient
        elif self.par.prey_follower_type == 'no_prey_ecm':
            self.chosen_prey_follower_fonction = self.function_doing_nothing
        else:
            # Raise an error and display a message if an invalid EPS type is specified
            print('prey_follower_type parameter could be: "follow_gradient", "no_prey_ecm"; default is "no_prey_ecm"\n')
            raise EcmTypeError()

        # Initialize EPS parameters and grid settings
        self.angle_section = self.par.eps_angle_view / self.par.n_sections  # Angle of view for each section
        self.edges_width = 2 * self.par.pili_length  # Length of EPS edges in µm
        # self.edges_width = self.par.pili_length  # Length of EPS edges in µm

        # If there is prey the egde of the ecm map is taken as the highest one
        if self.par.n_bact_prey > 0:
            edge_width_ecm_prey = 2*self.par.radius_prey_ecm_effect + self.par.n_nodes * self.par.d_n_prey
            if edge_width_ecm_prey > self.edges_width:
                self.edges_width = edge_width_ecm_prey

        self.ecm_space_size = self.par.space_size + 2 * self.edges_width  # Total EPS space length in µm
        self.ecm_space_size_px = self.uti.nearest_even(self.ecm_space_size / self.par.width_bins)  # Number of bins in the EPS grid
        self.edges_width_px = self.uti.nearest_even(self.edges_width * self.ecm_space_size_px / self.ecm_space_size)  # Edge bin width

        self.r = int(self.par.pili_length * self.ecm_space_size_px / self.ecm_space_size)  # Pili length in pixels
        # self.r = self.edges_width_px  # Pili length in pixels
        self.g_x = np.tile(np.arange(self.ecm_space_size_px), (self.ecm_space_size_px, 1))  # X coordinates in the EPS grid
        self.g_y = self.g_x.T  # Y coordinates in the EPS grid
        self.rate_eps_evaporation = 1 / self.par.eps_mean_lifetime  # Rate of EPS evaporation based on its mean lifetime

        # The eps is generate in a specific space but not by the preys (possible because the prey are not moving)
        self.cond_eps_generation = self.gen.cond_space_eps & ~self.gen.cond_prey

        # Initialize the EPS grid based on the bacterial positions and the simulation parameters
        # self.eps_grid = np.zeros((self.ecm_space_size_px, self.ecm_space_size_px), dtype=self.par.float_type)
        self.eps_grid = np.random.rand(self.ecm_space_size_px, self.ecm_space_size_px) * 2
        self.eps_grid_blur = np.zeros((self.ecm_space_size_px, self.ecm_space_size_px), dtype=self.par.float_type)

        self.eps_angle = np.zeros(self.par.n_bact, dtype=self.par.float_type)  # EPS following angles for each bacterium
        self.bact_angle = np.zeros(self.par.n_bact, dtype=self.par.float_type)  # Angles of each bacterium
        self.eps_diff = np.ones(self.par.n_bact, dtype=self.par.float_type)  # EPS diffusion values for each bacterium

        if self.par.n_bact_prey > 0:
            self.radius_ecm_prey_px = round(self.par.radius_prey_ecm_effect * self.ecm_space_size_px / self.par.space_size)
            self.size_kernel_detection_ecm_prey = 3
            self.local_angles_prey_ecm, self.local_directions_prey_ecm = self.uti.generate_directions_from_matrix_centers(self.size_kernel_detection_ecm_prey)

            # Id of the ecm prey edges
            self.prey_grid = np.zeros((self.ecm_space_size_px, self.ecm_space_size_px), dtype=self.par.float_type)
            self.ecm_prey_generation()


    def function_ecm_follower_type(self):
        """
        Executes the movement behavior based on the selected ECM follower type.
        """
        self.chosen_eps_follower_fonction()
        self.chosen_prey_follower_fonction()


    def function_doing_nothing(self):
        """
        No ECM production or following behavior.
        """
        pass


    def igoshin_eps_road_follower_new_test(self):
        """
        Simulates the behavior where bacteria follow the EPS road as described by Igoshin et al. (2015).
        """
        self.eps_generation()
        self.eps_direction_igoshin_new_test()
        self.follow_eps()


    def igoshin_eps_road_follower(self):
        """
        Simulates the behavior where bacteria follow the EPS road as described by Igoshin et al. (2015).
        """
        self.eps_generation()
        self.eps_direction_igoshin_old()
        self.follow_eps()


    def _create_grid_with_sphere_chains(self, data, grid_size, radius):
        """
        Create a grid with chains of spheres placed at specified positions.

        Parameters:
        - chains (numpy.ndarray): Array of shape (2, num_nodes, num_chains) representing (x, y) coordinates 
                                of spheres in each chain.
        - grid_size (tuple): Tuple (height, width) representing the dimensions of the grid.
        - radius (int): Radius of the spheres in pixels.

        Returns:
        - grid (numpy.ndarray): A 2D array of size `grid_size` with spheres drawn at each position in each chain.
        """
        # Initialize an empty grid
        height, width = grid_size
        grid = np.zeros((height, width), dtype=self.par.float_type)

        # Create a circular mask once based on the radius
        y_mask, x_mask = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        mask = (x_mask**2 + y_mask**2) <= radius**2

        # Get all (x, y) coordinates from the chains array
        x_coords = data[0, :, :].flatten()
        y_coords = data[1, :, :].flatten()

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


    def _create_exponential_decay_kernel(self, size, sharpness):
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


    def eps_generation(self):
        """
        Generates the current EPS map based on bacterial positions.
        """
        # Condition for bacteria having to follow the eps
        self.cond_follow_eps = self.cond_eps_generation & ~self.nei.cond_prey_neighbour
        eps_local, _, _ = np.histogram2d(
            self.gen.data[0, -1, self.cond_follow_eps],
            self.gen.data[1, -1, self.cond_follow_eps],
            bins=self.ecm_space_size_px,
            range=[[-self.edges_width, self.par.space_size + self.edges_width], [-self.edges_width, self.par.space_size + self.edges_width]]
        )
        # Reorder the matrix to fit with the position x, y
        # For example coord (0, 0) will be then the left bottom bin
        self.eps_grid += self.par.deposit_amount * eps_local * self.par.v0 * self.par.dt * np.exp(-self.rate_eps_evaporation * self.par.dt)

        # Symmetrically copy the edges to create a continuous EPS space
        self.eps_grid[:, :self.edges_width_px] = self.eps_grid[:, self.ecm_space_size_px - 2 * self.edges_width_px: self.ecm_space_size_px - self.edges_width_px]
        self.eps_grid[:, self.ecm_space_size_px - self.edges_width_px:] = self.eps_grid[:, self.edges_width_px:2 * self.edges_width_px]
        self.eps_grid[:self.edges_width_px, :] = self.eps_grid[self.ecm_space_size_px - 2 * self.edges_width_px: self.ecm_space_size_px - self.edges_width_px, :]
        self.eps_grid[self.ecm_space_size_px - self.edges_width_px:, :] = self.eps_grid[self.edges_width_px:2 * self.edges_width_px, :]
        
        # Blur EPS grid if required, very, very costly
        if self.par.sigma_blur_eps_map > 0:
            self.eps_grid_blur = gaussian_filter(input=self.eps_grid, sigma=self.par.sigma_blur_eps_map)
        else:
            self.eps_grid_blur = self.eps_grid.copy()


    def ecm_prey_generation(self):
        """
        Generate the prey ecm grid based on prey postition
        """
        # Generate de position inside a square where empty edges
        # prey_node_position_on_grid = ((self.gen.data[:, :, self.gen.first_index_prey_bact:] + self.edges_width) * self.ecm_space_size_px / self.ecm_space_size).astype(int)
        if self.nei.nb_prey_alive > 0:
            data_prey = self.gen.data[:, self.nei.cond_alive_bacteria].reshape(2, self.par.n_nodes, int(self.par.n_bact - self.par.n_bact_prey + self.nei.nb_prey_alive))[:, :, self.gen.first_index_prey_bact:]
            prey_node_position_on_grid = ((data_prey + self.edges_width) * self.ecm_space_size_px / self.ecm_space_size).astype(int)
            grid_tmp = self._create_grid_with_sphere_chains(data=prey_node_position_on_grid, 
                                                            grid_size=(self.ecm_space_size_px, self.ecm_space_size_px), 
                                                            radius=self.radius_ecm_prey_px)
            kernel_size_prey = round((self.par.radius_prey_ecm_effect * 2 + 1) * self.ecm_space_size_px / self.ecm_space_size)
            kernel_ecm_prey = self._create_exponential_decay_kernel(size=kernel_size_prey, sharpness=0.001)
            # Appliquer la convolution pour ajouter la diffusion
            self.prey_grid[:, :] = convolve(grid_tmp, kernel_ecm_prey, mode='reflect', cval=0)
            layer_value = 0  # La valeur du bord
            # Add self.size_bin_max_detection layer set to 0 around the matrix to be able to use the function `extract_local_squared` on a torus
            # self.prey_grid = np.pad(self.prey_grid, pad_width=self.radius_ecm_prey_px, mode='constant', constant_values=layer_value)

            # # Symmetrically copy the edges to create a torus ECM space
            # self.prey_grid[:,:self.edges_width_px] += self.prey_grid[:, self.ecm_space_size_px - 2 * self.edges_width_px: self.ecm_space_size_px - self.edges_width_px]
            # self.prey_grid[:, self.ecm_space_size_px - self.edges_width_px:] += self.prey_grid[:, self.edges_width_px:2 * self.edges_width_px]
            # self.prey_grid[:self.edges_width_px, :] += self.prey_grid[self.ecm_space_size_px - 2 * self.edges_width_px: self.ecm_space_size_px - self.edges_width_px, :]
            # self.prey_grid[self.ecm_space_size_px - self.edges_width_px:, :] += self.prey_grid[self.edges_width_px:2 * self.edges_width_px, :]

            # Flip vertically along the y-axis (top becomes bottom)
            self.prey_grid = self.prey_grid[::-1, :]
        else:
            self.prey_grid[:, :] = np.zeros((self.ecm_space_size_px, self.ecm_space_size_px), dtype=self.par.float_type)


    def compute_max_bin_rotation_matrix(self, grid, x, y, epsilon, cond_follow, min_ecm_prey_factor=1):
        """
        Find the neighbouring bin among the 8 neighbours bin of the ECM field (`grid`) 
        and returns a rotation matrix to align the bacteria with the direction of this bin
        under the condition that the scalar product between the bacteria direction and the
        direction difine by the actual head bin position to the max bin is positive.
        """
        nb_bact_impacted = np.sum(cond_follow)
        x_bins = ((x + self.edges_width) * self.ecm_space_size_px / self.ecm_space_size).astype(int)
        # Reverse the y coordinate to match with the map indices
        y_bins = ((y + self.edges_width) * self.ecm_space_size_px / self.ecm_space_size).astype(int)
        y_bins = np.abs(y_bins - self.ecm_space_size_px)
        
        bact_centers = np.array([x_bins, y_bins])
        local_bin_values = self.extract_local_squared(grid, bact_centers, self.size_kernel_detection_ecm_prey - 2)
        local_bin_value_flattened = local_bin_values.reshape(bact_centers.shape[1], (self.size_kernel_detection_ecm_prey)**2)

        bact_directions = self.dir.nodes_direction[:, 0, cond_follow].copy()
        scalar_product = np.sum(self.local_directions_prey_ecm[:, np.newaxis, :] * bact_directions[:, :, np.newaxis], axis=0)
        cond_angle_view = scalar_product >= 0
        # Set the central bin to False
        cond_angle_view[:, int(self.size_kernel_detection_ecm_prey**2 / 2)] = False

        local_bin_value_flattened[~cond_angle_view] = -1
        cond_maxs_local_bin_value = local_bin_value_flattened >= min_ecm_prey_factor * np.max(local_bin_value_flattened, axis=1)[:, np.newaxis]

        number_of_close_ecm_prey_bins_lvl = np.sum(cond_maxs_local_bin_value, axis=1)
        bact_angle = np.arctan2(bact_directions[1, :], bact_directions[0, :])
        angle_diff_bins_cells = np.abs(self.local_angles_prey_ecm[np.newaxis, :] - bact_angle[:, np.newaxis])
        # Set a big angle for the bins that are not selected by cond_maxs_local_bin_value
        angle_diff_bins_cells[~cond_maxs_local_bin_value] = np.pi

        # Build a new condition array with only False elements
        cond_maxs_local_bin_value_unique = np.zeros((nb_bact_impacted, int(self.size_kernel_detection_ecm_prey**2)), dtype=bool)
        # Condition that keep unique max bin values
        cond_maxs_local_bin_value_unique[number_of_close_ecm_prey_bins_lvl==1, :] = cond_maxs_local_bin_value[number_of_close_ecm_prey_bins_lvl==1, :]

        for i in range(2, int(self.size_kernel_detection_ecm_prey**2 / 2 + 2)):
            cond_i_bin_value_same_lvl = number_of_close_ecm_prey_bins_lvl == i
            if np.sum(cond_i_bin_value_same_lvl) > 0:
                tmp = angle_diff_bins_cells[cond_i_bin_value_same_lvl, :]
                idx_sector = np.argmin(tmp * 
                                        (1 + np.random.uniform(low=-0.001, high=0.001, size=tmp.shape)), axis=1)
                tmp = cond_maxs_local_bin_value_unique[cond_i_bin_value_same_lvl, :].copy()
                tmp[np.arange(idx_sector.size), idx_sector] = True
                cond_maxs_local_bin_value_unique[cond_i_bin_value_same_lvl, :] = tmp

        cond_no_ecm = np.sum(cond_maxs_local_bin_value_unique, axis=1) == 0
        # Put to True the center to find at least one True with np.where in the case there is no
        # bin detected (can be the case when all the bin are 0, e.g. there is no ecm)
        cond_maxs_local_bin_value_unique[cond_no_ecm, int(self.size_kernel_detection_ecm_prey**2 / 2)] = True
        # Get the indices of the True values
        rows, cols = np.where(cond_maxs_local_bin_value_unique)
        # Extract the corresponding values from `local_angles`
        ecm_prey_angle = self.local_angles_prey_ecm[cols]
        # If there is no ecm detected, the bacterium keep the same direction
        ecm_prey_angle[cond_no_ecm] = bact_angle[cond_no_ecm]
        rotation_angle = epsilon * np.sin(2 * (ecm_prey_angle - bact_angle)) * self.par.dt
        rotation_matrix = self.uti.rotation_matrix_2d(theta=rotation_angle)

        return rotation_matrix
        


    def compute_gradient_rotation_matrix(self, grid, x, y, epsilon, cond_follow, bins, l_ecm, edges_width):
        """
        Computes the gradient of the ECM field and returns a rotation matrix to align the bacteria
        with the gradient direction.
        THIS MODULE DO NOT WORK CORRECTLY
        """
        grad_y, grad_x = np.gradient(grid)
        x_bins = ((x + edges_width) * bins / l_ecm).astype(int)
        # Reverse the y coordinate to match with the map indices
        y_bins = ((y + edges_width) * bins / l_ecm).astype(int)

        x_dir = grad_x[y_bins, x_bins]
        y_dir = grad_y[y_bins, x_bins]

        # # # Get gradient values at these positions
        # y_dir = grad_x[x_bins, y_bins]
        # x_dir = grad_y[x_bins, y_bins]

        epsilon_grad = np.linalg.norm(np.array([x_dir, y_dir]), axis=0)
        direction_rotation = - np.sign(np.sum(np.array([self.dir.nodes_direction[0, 0, cond_follow], self.dir.nodes_direction[1, 0, cond_follow]]) * np.array([x_dir, y_dir]), axis=0))
        # direction_rotation[direction_rotation < 0] = 0
        epsilon_grad[epsilon_grad > 0] = 1.
        angle_rotation = epsilon * epsilon_grad * direction_rotation * self.par.dt
        rotation_matrix = self.uti.rotation_matrix_2d(theta=angle_rotation)

        return rotation_matrix


    def follow_eps_gradient(self):
        """
        Bacteria will follow the gradient of the EPS field.
        """
        self.eps_generation()
        rotation_matrix = self.compute_gradient_rotation_matrix(grid=self.eps_grid_blur.T, 
                                                                x=self.gen.data[0, 0, self.cond_follow_eps], 
                                                                y=self.gen.data[1, 0, self.cond_follow_eps], 
                                                                epsilon=self.par.epsilon_eps, 
                                                                cond_follow=self.cond_follow_eps,
                                                                bins=self.ecm_space_size_px,
                                                                l_ecm=self.ecm_space_size,
                                                                edges_width=self.edges_width)
        rotation_data = np.sum(rotation_matrix * (self.pha.data_phantom[:, 0, self.cond_follow_eps] - self.pha.data_phantom[:, 1, self.cond_follow_eps]), axis=1) + self.gen.data[:, 1, self.cond_follow_eps]
        rotation_data_phantom = np.sum(rotation_matrix * (self.pha.data_phantom[:, 0, self.cond_follow_eps] - self.pha.data_phantom[:, 1, self.cond_follow_eps]), axis=1) + self.pha.data_phantom[:, 1, self.cond_follow_eps]
        self.gen.data[:, 0, self.cond_follow_eps] = rotation_data[:, :]
        self.pha.data_phantom[:, 0, self.cond_follow_eps] = rotation_data_phantom[:, :]


    def follow_prey_gradient(self):
        """
        Bacteria will follow the gradient of the ECM of the prey.
        """
        # Update the prey ecm matrix after a death and put the update to False
        if self.nei.update_prey_ecm_matrix:
            self.ecm_prey_generation()
            self.nei.update_prey_ecm_matrix = False

        cond_follow_prey_gradient = ~self.gen.cond_prey & ~self.nei.cond_prey_neighbour
        
        rotation_matrix = self.compute_max_bin_rotation_matrix(grid=self.prey_grid, 
                                                               x=self.gen.data[0, 0, cond_follow_prey_gradient], 
                                                               y=self.gen.data[1, 0, cond_follow_prey_gradient], 
                                                               epsilon=self.par.epsilon_prey, 
                                                               cond_follow=cond_follow_prey_gradient)
        
        rotation_data = np.sum(rotation_matrix * (self.pha.data_phantom[:, 0, cond_follow_prey_gradient] - self.pha.data_phantom[:, 1, cond_follow_prey_gradient]), axis=1) + self.gen.data[:, 1, cond_follow_prey_gradient]
        rotation_data_phantom = np.sum(rotation_matrix * (self.pha.data_phantom[:, 0, cond_follow_prey_gradient] - self.pha.data_phantom[:, 1, cond_follow_prey_gradient]), axis=1) + self.pha.data_phantom[:, 1, cond_follow_prey_gradient]
        self.gen.data[:, 0, cond_follow_prey_gradient] = rotation_data[:, :]
        self.pha.data_phantom[:, 0, cond_follow_prey_gradient] = rotation_data_phantom[:, :]


    def eps_bisectors_old(self,bact_direction):
        """
        Find the bisector of each section in the angle view of the bacterium
        
        """
        bact_angle = np.arctan2(bact_direction[1], bact_direction[0])
        start = bact_angle - self.par.eps_angle_view / 2 + self.par.eps_angle_view / (self.par.n_sections * 2)
        end = bact_angle + self.par.eps_angle_view / 2 - self.par.eps_angle_view / (self.par.n_sections * 2)
        angles_bisectors_sections = np.linspace(start, end, self.par.n_sections)

        return angles_bisectors_sections, bact_angle
    

    def eps_bisectors(self):
        """
        Calculates the bisectors of each section in the bacterium's angle of view.
        """
        self.bact_angle = np.arctan2(self.dir.nodes_direction[1, 0, :], self.dir.nodes_direction[0, 0, :])
        start = self.bact_angle - self.par.eps_angle_view / 2 + self.par.eps_angle_view / (self.par.n_sections * 2)
        end = self.bact_angle + self.par.eps_angle_view / 2 - self.par.eps_angle_view / (self.par.n_sections * 2)
        self.angles_bisectors_sections = np.linspace(start, end, self.par.n_sections)
    

    def extract_local_squared(self, matrix, centers, radius):
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
    

    def eps_values_sections_old(self,x,y,angles_bisectors_sections):
        """
        Compute the value of the total eps in each section
        
        """
        sum_eps_section = np.zeros(self.par.n_sections)
        max_eps_section = np.zeros(self.par.n_sections)
        average_eps_section = np.zeros(self.par.n_sections)

        local_eps_grid = self.eps_grid_blur[x-self.r:x+self.r-1, y-self.r:y+self.r-1].T.copy()
        # local_eps_grid = self.eps_grid_blur.T[y-self.r:y+self.r, x-self.r:x+self.r].copy()
        local_coord_grid_x = self.g_x[x-self.r:x+self.r-1, x-self.r:x+self.r-1].copy()
        local_coord_grid_y = self.g_y[y-self.r:y+self.r-1, y-self.r:y+self.r-1].copy()
        local_dir_x = local_coord_grid_x - x
        local_dir_y = local_coord_grid_y - y
        eps_bins_sector_x = []
        eps_bins_sector_y = []

        for sector, angle in enumerate(angles_bisectors_sections):

            norm_dir = np.linalg.norm(np.array([local_dir_x, local_dir_y]), axis=0)
            scalar_product = local_dir_x * np.cos(angle) + local_dir_y * np.sin(angle)
            cond_sphere = (norm_dir <= self.r) & (norm_dir > 0)
            cond_section_view = scalar_product >= np.cos(self.par.angle_section / 2) * norm_dir
            local_section = local_eps_grid[cond_sphere & cond_section_view]
            sum_eps_section[sector] = np.sum(local_section)
            max_eps_section[sector] = np.max(local_section)
            # Compute the average of the bins values and add some noise in weights in case of weights == 0
            background_noise = np.ones(len(local_section)) * 1e-6
            weights = local_section + background_noise
            average_eps_section[sector] = np.average(local_section, axis=None, weights=weights)
            eps_bins_sector_x.append(local_coord_grid_x[cond_sphere & cond_section_view])
            eps_bins_sector_y.append(local_coord_grid_y[cond_sphere & cond_section_view])

        return sum_eps_section, max_eps_section, average_eps_section, np.array(eps_bins_sector_x,dtype=object), np.array(eps_bins_sector_y,dtype=object)


    def eps_values_sections(self):
        """
        Compute the value of the total EPS in each section.
        
        This method calculates the local EPS values for each bacterial section and the directional components of the EPS. 
        It also identifies the highest EPS value within each section.
        """
        centers = ((self.gen.data[:, 0, :] + self.edges_width) * self.ecm_space_size_px / self.ecm_space_size).astype(int)

        # Extract local EPS values and coordinates
        index_test = 0
        x = centers[0, index_test]
        y = centers[1, index_test]
        local_eps_grid_old = self.eps_grid_blur[x-self.r:x+self.r-1, y-self.r:y+self.r-1].T.copy()
        local_coord_grid_x_old = self.g_x[x-self.r:x+self.r-1, x-self.r:x+self.r-1].copy()
        local_coord_grid_y_old = self.g_y[y-self.r:y+self.r-1, y-self.r:y+self.r-1].copy()

        local_eps_grid = self.extract_local_squared(self.eps_grid_blur.T, centers - self.r, self.r)
        local_coord_grid_x = self.extract_local_squared(self.g_x, centers - self.r, self.r)
        local_coord_grid_y = self.extract_local_squared(self.g_y, centers - self.r, self.r)
        # local_eps_grid = self.extract_local_squared(self.eps_grid_blur.T, centers, self.r)
        # local_coord_grid_x = self.extract_local_squared(self.g_x, centers, self.r)
        # local_coord_grid_y = self.extract_local_squared(self.g_y, centers, self.r)

        # Calculate directional components based on local coordinates
        local_dir_x_old = local_coord_grid_x_old - x
        local_dir_y_old = local_coord_grid_y_old - y

        local_dir_x = local_coord_grid_x - centers[0, :, np.newaxis, np.newaxis]
        local_dir_y = local_coord_grid_y - centers[1, :, np.newaxis, np.newaxis]

        # Initialize arrays for storing EPS sums and maxima for each section
        self.sum_eps_section = np.zeros((self.par.n_sections, centers[0, :].shape[0]))
        self.max_eps_section = np.zeros((self.par.n_sections, centers[0, :].shape[0]))

        sum_eps_section_old = np.zeros(self.par.n_sections)
        max_eps_section_old = np.zeros(self.par.n_sections)

        # Loop through each sector and calculate EPS values
        for sector, angle in enumerate(self.angles_bisectors_sections):
            norm_dir = np.linalg.norm(np.array([local_dir_x, local_dir_y]), axis=0)
            scalar_product = local_dir_x * np.cos(angle[:, np.newaxis, np.newaxis]) + local_dir_y * np.sin(angle[:, np.newaxis, np.newaxis])
            cond_sphere = (norm_dir <= self.r) & (norm_dir > 0)
            cond_section_view = scalar_product >= np.cos(self.angle_section / 2) * norm_dir
            local_section = local_eps_grid * cond_sphere * cond_section_view

            norm_dir_old = np.linalg.norm(np.array([local_dir_x_old, local_dir_y_old]), axis=0)
            scalar_product_old = local_dir_x_old * np.cos(angle[index_test]) + local_dir_y_old * np.sin(angle[index_test])
            cond_sphere_old = (norm_dir_old <= self.r) & (norm_dir_old > 0)
            cond_section_view_old = scalar_product_old >= np.cos(self.par.angle_section / 2) * norm_dir_old
            local_section_old = local_eps_grid_old[cond_sphere_old & cond_section_view_old]
            sum_eps_section_old[sector] = np.sum(local_section_old)
            max_eps_section_old[sector] = np.max(local_section_old)

            # Update EPS section sums and maxima
            self.sum_eps_section[sector, :] = np.sum(local_section[:, :, :], axis=(1, 2))
            self.max_eps_section[sector, :] = np.max(local_section[:, :, :], axis=(1, 2))



    def eps_direction_igoshin_old(self, min_eps_factor=0.8):
        """
        Find the direction of the cell which follow the eps road

        """
        # head_direction = data[:,:,0] - data[:,:,1]
        # Transform data heads coordinates into coordinates on the eps grid
        x_head = ((self.gen.data[0,0,:] + self.edges_width) * self.ecm_space_size_px / self.ecm_space_size).astype(int)
        y_head = ((self.gen.data[1,0,:] + self.edges_width) * self.ecm_space_size_px / self.ecm_space_size).astype(int)
        # eps_dir = np.zeros(data[:,0,:].shape)
        # array_coord_x_sector = np.array([])
        # array_coord_y_sector = np.array([])
        # array_coord_x_sectors = np.array([])
        # array_coord_y_sectors = np.array([])
        indices = np.where(self.gen.cond_space_eps)[0]

        for index in indices:
            angles_bisectors_sections, bact_angle_i = self.eps_bisectors_old(bact_direction=self.dir.nodes_direction[:,0,index].T)
            sum_eps_section, max_eps_section, average_eps_section, eps_bins_sector_x, eps_bins_sector_y = self.eps_values_sections_old(x=x_head[index],
                                                                                                                                   y=y_head[index],
                                                                                                                                   angles_bisectors_sections=angles_bisectors_sections)
            # Compute the difference of the values between the section with the higher
            # amouth of eps with the eps in the middle section
            idx_sector = np.argmax(sum_eps_section)
            self.eps_angle[index] = angles_bisectors_sections[idx_sector]
            self.bact_angle[index] = bact_angle_i
            if np.max(max_eps_section) < self.par.max_eps_value / 100:
                # pass
                self.eps_angle[index] = bact_angle_i
                self.bact_angle[index] = bact_angle_i
                # eps_dir[:,index] = np.array([0, 0])
                # array_coord_x_sector = np.concatenate((array_coord_x_sector, eps_bins_sector_x[int(self.n_sec/2)]))
                # array_coord_y_sector = np.concatenate((array_coord_y_sector, eps_bins_sector_y[int(self.n_sec/2)]))
            else:
                # Select sections with more than min_eps_factor of the max section value
                cond = sum_eps_section >= min_eps_factor * np.max(sum_eps_section)
                selected_angles = angles_bisectors_sections[cond]
                # selected_averages = average_eps_section[cond]
                # selected_coord_sectors_x = eps_bins_sector_x[cond]
                # selected_coord_sectors_y = eps_bins_sector_y[cond]
                idx_sector = np.argmin(np.abs(selected_angles - bact_angle_i) * (1 + np.random.uniform(-0.001,0.001,len(selected_angles))))
                self.eps_angle[index] = selected_angles[idx_sector]
                self.bact_angle[index] = bact_angle_i
                # eps_dir[index,:] = np.array([np.cos(eps_angle), np.sin(eps_angle)])
                # eps_dir[index,:] = eps_dir[index,:] * selected_averages[idx_sector] / self.max_eps
                # array_coord_x_sector = np.concatenate((array_coord_x_sector, selected_coord_sectors_x[idx_sector]))
                # array_coord_y_sector = np.concatenate((array_coord_y_sector, selected_coord_sectors_y[idx_sector]))
            # array_coord_x_sectors = np.concatenate((array_coord_x_sectors, np.concatenate(eps_bins_sector_x)))
            # array_coord_y_sectors = np.concatenate((array_coord_y_sectors, np.concatenate(eps_bins_sector_y)))

        # return eps_angle, bact_angle


    def eps_direction_igoshin_new_test(self, min_eps_factor=0.8):
        """
        Find the direction of the cell which follows the EPS gradient.

        This method determines which section the bacteria will choose based on the highest EPS value and adjusts
        the bacteria's movement direction accordingly. If multiple sections have high EPS values, the closest section
        to the bacteria's current direction is chosen.

        Parameters:
        -----------
        min_eps_factor (float): The minimum EPS value relative to the maximum EPS value for selecting sections.
        """
        self.eps_bisectors()
        self.eps_values_sections()

        # Determine the section with the highest total EPS value
        idx_sector = np.argmax(self.sum_eps_section[:, :], axis=0)
        self.eps_angle = self.angles_bisectors_sections[idx_sector, np.arange(len(idx_sector))]

        # If multiple sections have EPS values above a threshold, select the section closest to the bacteria's direction
        cond_same_eps_lvl = self.sum_eps_section[:, :] >= min_eps_factor * np.max(self.sum_eps_section, axis=0)
        number_of_close_eps_lvl = np.sum(cond_same_eps_lvl, axis=0)
        angle_diff_sections_main = np.abs(self.angles_bisectors_sections - self.bact_angle)

        # Adjust direction based on EPS gradient.
        # In the case several section have the same value, choose the one closer than the actual cell direction.
        # In the case two sections have the same lvl either the same distance compare to the actual cell direction,
        # choose a random section
        for i in range(2, self.par.n_sections + 1):
            cond_i_sections_same_lvl = number_of_close_eps_lvl == i
            idx_sector = np.argmin(angle_diff_sections_main[:, cond_i_sections_same_lvl] * 
                                   (1 + np.random.uniform(-0.001, 0.001, np.sum(cond_i_sections_same_lvl))), axis=0)
            self.eps_angle[cond_i_sections_same_lvl] = self.angles_bisectors_sections[:, cond_i_sections_same_lvl][idx_sector, np.arange(len(idx_sector))]

        # If EPS is too low in all sections, move in the current direction
        cond_change_min_eps = np.max(self.max_eps_section, axis=0) < self.par.max_eps_value / 100
        self.eps_angle[cond_change_min_eps] = self.bact_angle[cond_change_min_eps].copy()


    def follow_eps(self):
        """
        Align the cells with the EPS direction.

        This method applies a rotation based on the calculated EPS direction, adjusting the bacteria's position accordingly.
        """
        # Construct the rotation matrix based on EPS heterogeneity and bacteria's current angle
        rotation_angle = self.par.epsilon_eps * self.eps_diff * np.sin(2 * (self.eps_angle - self.bact_angle)) * self.par.dt

        # Apply rotation to update the position of bacteria and their phantom data
        rotation_matrix = self.uti.rotation_matrix_2d(theta=rotation_angle[self.cond_follow_eps])
        rotation_data = np.sum(rotation_matrix * (self.pha.data_phantom[:, 0, self.cond_follow_eps] - self.pha.data_phantom[:, 1, self.cond_follow_eps]), axis=1) + self.gen.data[:, 1, self.cond_follow_eps]
        rotation_data_phantom = np.sum(rotation_matrix * (self.pha.data_phantom[:, 0, self.cond_follow_eps] - self.pha.data_phantom[:, 1, self.cond_follow_eps]), axis=1) + self.pha.data_phantom[:, 1, self.cond_follow_eps]

        # Update the data arrays with new rotated positions
        self.gen.data[:, 0, self.cond_follow_eps] = rotation_data[:, :]
        self.pha.data_phantom[:, 0, self.cond_follow_eps] = rotation_data_phantom[:, :]
