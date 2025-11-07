"""
Author: Jean-Baptiste Saulnier
Date: 2024-12-05
"""
from scipy.spatial import KDTree as kdtree
import numpy as np
import matplotlib.pyplot as plt


class Neighbours:
    """
    Finds and manages the nearest neighbours of bacterial nodes in 2D space.

    This class is responsible for finding the nearest neighbours of bacterial 
    nodes in 2D space. The behavior of distance calculation is controlled by 
    the `neighbour_detection_type` parameter, which can be set to options such 
    as "euclidean" or "torus".  The toroidal distance accounts for periodic 
    boundary conditions, ensuring continuity in simulations of confined spaces. 
    This class also manages node indices, bacterial IDs, and conditions for
    identifying nodes belonging to the same bacterium.

    Attributes:
    -----------
    par : object
        Instance of the `Parameters` class containing simulation parameters.
    gen : object
        Instance of the generator class, providing bacterial node positions.
    coord : ndarray
        Array to store node coordinates for KDTree processing.
    ind : ndarray
        Indices of the k nearest neighbours for each node.
    dist : ndarray
        Distances to the k nearest neighbours for each node.
    ind_flat : ndarray
        Flattened version of `ind` for performance optimization.
    dist_flat : ndarray
        Flattened version of `dist` for performance optimization.
    id_node : ndarray
        Node indices of the k nearest neighbours.
    id_bact : ndarray
        Bacterium IDs of the k nearest neighbours.
    cond_same_bact : ndarray
        Boolean array indicating if neighbouring nodes belong to the same bacterium.
    array_same_bact : ndarray
        Array used to generate conditions for determining shared bacterium membership.
    cond_prey_neighbour : ndarray
        Boolean array that is True if a cell has a prey as a neighbours
    chosen_neighbour_function : method
        Method used to determine the k nearest neighbours based on the chosen distance metric.
    """
    def __init__(self, inst_par, inst_gen):
        # Store references to external class instances
        self.par = inst_par
        self.gen = inst_gen

        # Prepare arrays for KDTree input and output
        self.coord = np.zeros((int(self.par.n_bact * self.par.n_nodes), 2), dtype=self.par.float_type)
        self.ind = np.zeros((int(self.par.n_bact * self.par.n_nodes), self.par.kn), dtype=self.par.int_type)
        self.dist = np.zeros(self.ind.shape)
        self.ind_flat = np.zeros(int(self.par.n_bact * self.par.n_nodes * self.par.kn), dtype=self.par.int_type)
        self.dist_flat = np.zeros(self.ind_flat.shape)

        # Node and bacterium IDs, with condition for shared bacterium
        self.id_node = np.zeros(self.ind.shape, dtype=self.par.int_type)
        self.id_bact = np.zeros(self.ind.shape, dtype=self.par.int_type)
        self.cond_same_bact = np.zeros(self.ind.shape, dtype=bool)

        # Array for checking if neighbours belong to the same bacterium
        self.array_same_bact = np.tile(
            (np.ones((self.par.kn, self.par.n_bact)) * np.arange(self.par.n_bact)).T.astype(self.par.int_type),
            (self.par.n_nodes, 1),
        )
        
        # Array for checking the prey neibhours
        self.cond_prey_neighbour = np.zeros(self.par.n_bact, dtype=bool)

        # Choose the appropriate neighbour function based on the distance metric
        if self.par.neighbour_detection_type == "torus":
            self.chosen_neighbour_function = self.set_kn_nearest_neighbours_torus
        elif self.par.neighbour_detection_type == "euclidean":
            self.chosen_neighbour_function = self.set_kn_nearest_neighbours_euclidian
        else:
            print('movement_type could be: "torus", "euclidean" \n')
            raise ValueError("Invalid movement type in parameters. Please choose 'torus' or 'euclidean'.")
        
        # Boolean array that is True if the bacterium is alive and False otherwise
        # Initially all bacteria are alive
        self.cond_alive_bacteria = np.ones((self.par.n_nodes, self.par.n_bact), dtype=bool)
        if self.par.n_bact_prey > 0:
            # Initialized the index of dead prey.
            # Initially an empty array
            self.ind_dead = np.where(~self.cond_alive_bacteria.flatten())[0]
            self.nb_prey_alive = self.par.n_bact_prey
            self.count_prey_alive = self.par.n_bact_prey
            self.update_prey_ecm_matrix = False


    def tore_dist(self, a, b):
        """
        Compute the toroidal distance between two points in 2D.

        Parameters:
        -----------
        a, b : array-like
            Coordinates of two points (x, y).

        Returns:
        --------
        float
            The toroidal distance between the two points.
        """
        dx = (b[0] - a[0] + self.par.space_size) % self.par.space_size
        dy = (b[1] - a[1] + self.par.space_size) % self.par.space_size

        return np.sqrt(np.minimum(dx, self.par.space_size - dx) ** 2 + np.minimum(dy, self.par.space_size - dy) ** 2)


    def set_kn_nearest_neighbours_torus(self):
        """
        Detect the k closest neighbours using toroidal distance (2D space with periodic boundaries).
        """
        # Reshape data for KDTree
        self.coord[:, :] = np.column_stack((self.gen.data[0, :, :].flatten(), self.gen.data[1, :, :].flatten()))
        # Build KDTree with toroidal boundary conditions
        tree = kdtree(self.coord, boxsize=[self.par.space_size, self.par.space_size])
        self.dist[:, :], self.ind[:, :] = tree.query(self.coord, k=self.par.kn)

        if self.par.n_bact_prey > 0:
            # Compute a boolean array of the same size of ind that is True for the 
            # nodes of prey cells tht are dead
            cond_dead_prey = np.isin(self.ind, self.ind_dead)
            # Set a big distances For the dead prey cell to avoid any interaction with them
            self.dist[cond_dead_prey] = 100 * self.par.width

        self.dist_flat[:], self.ind_flat[:] = np.concatenate(self.dist), np.concatenate(self.ind)


    def set_kn_nearest_neighbours_euclidian(self):
        """
        Detect the k closest neighbours using Euclidean distance.
        """
        # Reshape data for KDTree
        self.coord[:, :] = np.column_stack((self.gen.data[0, :, :].flatten(), self.gen.data[1, :, :].flatten()))
        # Build KDTree without boundary conditions
        tree = kdtree(self.coord)
        self.dist[:, :], self.ind[:, :] = tree.query(self.coord, k=self.par.kn)

        if self.par.n_bact_prey > 0:
            # Compute a boolean array of the same size of ind that is True for the 
            # nodes of prey cells tht are dead
            cond_dead_prey = np.isin(self.ind, self.ind_dead)
            # Set a big distances For the dead prey cell to avoid any interaction with them
            self.dist[cond_dead_prey] = 100 * self.par.width

        self.dist_flat[:], self.ind_flat[:] = np.concatenate(self.dist), np.concatenate(self.ind)


    def set_bacteria_index(self):
        """
        Compute bacterial and node indices for neighbours, and identify shared bacterium conditions.
        """
        # Compute node and bacterium indices
        self.id_node[:, :], self.id_bact[:, :] = np.divmod(self.ind, self.par.n_bact)
        # Determine if nodes belong to the same bacterium
        self.cond_same_bact[:, :] = self.id_bact == self.array_same_bact


    def find_neighbours(self):
        """
        Find the nearest neighbours using the chosen distance metric.
        """
        self.chosen_neighbour_function()
    

    def _compute_number_of_neighbors(self, cond_id, cond_dist):
        """
        Compute the number of different neighbors for each bacterium.
        A bacterium is considered a neighbor if at least one of its nodes is 
        within a given threshold distance from a node of a different bacterium.

        Parameters:
        -----------
        cond_id : np.ndarray (bool)
            A boolean mask used to filter which bacterium IDs should be considered as 
            potential neighbors, excluding unwanted IDs.

        cond_dist : np.ndarray (bool)
            A boolean mask indicating which node pairs are within the allowed 
            distance threshold to be considered neighbors.

        Returns:
        --------
        np.ndarray (int)
            An array of shape (n_bact,) containing the number of different 
            neighboring bacteria for each bacterium.
        """ 
        # Identify valid neighbors: a node pair is considered valid if:
        # - cond_id is True (the neighbor ID is considered valid)
        # - cond_dist is True (distance is within the threshold)
        # - ~self.cond_same_bact ensures that a bacterium is not counted as its own neighbor
        valid_neighbors = cond_id & cond_dist & ~self.cond_same_bact  

        # Reshape the bacterium indices and valid neighbors mask to align with 
        # the number of bacteria and their nodes.
        count_n = self.id_bact.reshape((self.par.n_bact, self.par.n_nodes * self.par.kn), order='F')
        valid_neighbors = valid_neighbors.reshape((self.par.n_bact, self.par.n_nodes * self.par.kn), order='F')

        # Sort neighbors based on bacterium index to ensure consistency in counting.
        valid_neighbors_sorted = np.take_along_axis(valid_neighbors, count_n.argsort(axis=1), axis=1)
        count_n_sorted = np.sort(count_n, axis=1)

        # Smart counting of neighbors: 
        # - Invalid neighbors (those marked as False in valid_neighbors_sorted) are penalized 
        #   by subtracting (n_bact + 1) to make them easily distinguishable.
        count_n_smart = count_n_sorted.copy()
        count_n_smart[valid_neighbors_sorted] -= (self.par.n_bact + 1)  

        # Sort again after penalizing invalid neighbors.
        count_n_smart = np.sort(count_n_smart, axis=1)  

        # Compute the difference between consecutive sorted neighbor indices:
        # - If the difference is nonzero, it means a new bacterium neighbor was found.
        count_n_smart = np.diff(count_n_smart, axis=1)
        count_n_smart[count_n_smart != 0] = 1  # Convert nonzero differences to 1 to count unique neighbors.

        # Perform the same difference-based counting on the original sorted neighbor indices.
        count_n = np.diff(count_n_sorted, axis=1)
        count_n[count_n != 0] = 1  

        # Compute the final number of valid neighboring bacteria:
        # - Subtracting ensures that redundant neighbor counts are eliminated.
        return np.sum(count_n_smart, axis=1) - np.sum(count_n, axis=1)



    def find_prey_neighbours(self):
        """
        Set element in cond_prey_neighbour to True if a prey is a neighbour of a cell
        """
        if self.par.n_bact_prey > 0:
            # PREY NEIGHBOURS
            # Detection if a prey is a neighbour of another cell
            cond_id_prey_reshaped = self.id_bact.reshape(self.par.n_nodes, self.par.n_bact, self.par.kn) >= self.gen.first_index_prey_bact
            # Condition that is True if the neighbour node is in contact with the node
            min_neighbour_dist = 1.1 * self.par.width
            cond_dist_reshaped = self.dist.reshape(self.par.n_nodes, self.par.n_bact, self.par.kn) < min_neighbour_dist
            # Condition that is True if a bacterium has at least one prey in contact (dim = n_bact)
            self.cond_prey_neighbour[:] = np.sum(cond_id_prey_reshaped & cond_dist_reshaped, axis=(0, 2)) > 0
            # Set to 0 this condition for the prey
            self.cond_prey_neighbour[self.gen.first_index_prey_bact:] = False

            # PREDATOR NEIGHBOUR
            # Compute the number of predator neighbours
            # nb_predator_neighbour = np.sum(~cond_id_prey_reshaped & cond_dist_reshaped, axis=(0, 2)) # Warning; here is the number of node in contact with a prey
            cond_id_predator = self.id_bact < self.gen.first_index_prey_bact # Predator id
            cond_dist = self.dist < min_neighbour_dist
            nb_predator_neighbour = self._compute_number_of_neighbors(cond_id=cond_id_predator, cond_dist=cond_dist)
            # Set to 0 the number of predator neighbours for the predator
            nb_predator_neighbour[:self.gen.first_index_prey_bact] = 0

            # UPDATE THE BACTERIA THAT ARE ALIVED 
            # Compute the probability of death
            prob = 1 - np.exp(-self.par.rate_prey_death * self.par.dt) * np.ones(self.par.n_bact)
            # Compute the condition that a death occurs. 
            # The probability increases with the number of predator neighbours
            cond_death = np.random.binomial(1, prob * nb_predator_neighbour).astype(bool)
            # Set to False for the prey that are killed by the predator
            self.cond_alive_bacteria[:, cond_death] = False
            # Compute the index of dead prey nodes
            self.ind_dead = np.where(~self.cond_alive_bacteria.flatten())[0]
            # Compute the number of prey still alive
            self.nb_prey_alive = np.sum(self.cond_alive_bacteria[0, :]) - (self.par.n_bact - self.par.n_bact_prey)

            if self.count_prey_alive - self.nb_prey_alive > 0:
                self.update_prey_ecm_matrix = True
                self.count_prey_alive = self.nb_prey_alive