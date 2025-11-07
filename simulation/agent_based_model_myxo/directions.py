"""
Author: Jean-Baptiste Saulnier
Date: 2024-12-05
"""
import numpy as np


class Direction:
    """
    This class handles the computation of directional information between 
    bacterial nodes in a 2D space, considering both Euclidean and toroidal 
    distances. The class computes the directions and distances between nodes, 
    as well as the directions between nodes and their k-nearest neighbors. It 
    also provides methods to handle the special case of periodic boundary 
    conditions (torus space), ensuring accurate direction calculations for 
    simulations in confined spaces.

    Attributes:
    -----------
    par : object
        Instance of the `Parameters` class defining simulation settings.
    gen : object
        Instance of the class managing bacterial data generation.
    pha : object
        Instance of the class managing phantom node data for simulation purposes.
    nei : object
        Instance of the class managing neighbor calculations and relationships.
    nodes_direction : ndarray
        Array storing the directions of bacterial nodes, shaped `(2, n_nodes, n_bact)`.
    nodes_distance : ndarray
        Array storing the distances between bacterial nodes and a reference point, shaped `(n_nodes, n_bact)`.
    neighbours_direction : ndarray
        Array storing the directions to neighbors for each node, shaped `(2, n_bact * n_nodes, kn)`.
    nodes_to_nei_dir_torus : ndarray
        Array storing toroidal directions to neighbors for each node, shaped `(2, n_bact * n_nodes, kn)`.
    nodes_to_nei_dist : ndarray
        Array storing toroidal distances to neighbors for each node, shaped `(n_bact * n_nodes, kn)`.
    """
    def __init__(self, inst_par, inst_gen, inst_pha, inst_nei):
        # Store references to external class instances
        self.par = inst_par
        self.gen = inst_gen
        self.pha = inst_pha
        self.nei = inst_nei

        ### Initialize Direction class attributes
        # Array for storing the directional vectors of bacterial nodes
        self.nodes_direction = np.zeros(
            (2, self.par.n_nodes, self.par.n_bact), dtype=self.par.float_type
        )

        # Array for storing distances of bacterial nodes from a reference point
        self.nodes_distance = np.zeros(
            (self.par.n_nodes, self.par.n_bact), dtype=self.par.float_type
        )
        
        # Array for storing the directional vectors to neighbors
        self.neighbours_direction = np.zeros(
            (2, int(self.par.n_bact * self.par.n_nodes), self.par.kn), dtype=self.par.float_type
        )
        
        # Array for storing toroidal directions to neighbors
        self.nodes_to_nei_dir_torus = np.zeros(
            (2, int(self.par.n_bact * self.par.n_nodes), self.par.kn), dtype=self.par.float_type
        )
        
        # Array for storing toroidal distances to neighbors
        self.nodes_to_nei_dist = np.zeros(
            (int(self.par.n_bact * self.par.n_nodes), self.par.kn), dtype=self.par.float_type
        )


    def torus_dist(self, a, b):
        """
        Computes the distance between two points in a 2D toroidal space, considering periodic boundary conditions.
        
        Parameters:
        -----------
        a : ndarray
            Coordinates of the first point.
        b : ndarray
            Coordinates of the second point.
        
        Returns:
        --------
        float
            The toroidal distance between the two points.
        """
        
        # Calculate the periodic distances in the x and y directions
        dx = (a[0] - b[0] + self.par.space_size) % self.par.space_size
        dy = (a[1] - b[1] + self.par.space_size) % self.par.space_size

        # Return the Euclidean distance accounting for periodic boundary conditions
        return np.sqrt(np.minimum(dx, self.par.space_size - dx) ** 2 + np.minimum(dy, self.par.space_size - dy) ** 2)


    def set_nodes_direction(self):
        """
        Computes the direction and the distance between consecutive nodes of each bacterium.
        
        This method calculates the vector between each node and its successive node within a bacterium and stores 
        the results in the `nodes_direction` and `nodes_distance` arrays. It normalizes the direction vectors 
        to ensure they represent unit vectors.
        """
        
        # Compute the direction vectors between successive nodes for each bacterium
        self.nodes_direction[:, 1:, :] = self.pha.data_phantom[:, :-1, :] - self.pha.data_phantom[:, 1:, :]
        
        # Copy the direction of the first node to handle boundary conditions (for circular topology of nodes)
        self.nodes_direction[:, 0, :] = self.nodes_direction[:, 1, :].copy()
        
        # Calculate the Euclidean distance between each pair of consecutive nodes
        self.nodes_distance[:, :] = np.linalg.norm(self.nodes_direction, axis=0)
        
        # Normalize the direction vectors to unit vectors
        self.nodes_direction[:, :, :] = self.nodes_direction[:, :, :] / self.nodes_distance[:, :]


    def set_neighbours_direction(self):
        """
        Extracts the direction vectors of each node's k-nearest neighbors.

        This method uses the global node directions stored in `self.nodes_direction` 
        and the neighbor indices `self.nei.ind` to extract, for each focal node, 
        the direction of its `kn` neighbors. The result is stored in `self.neighbours_direction`,
        with shape (2, n_total_nodes, kn), aligned with neighbor structure.

        Sets
        -----
        self.neighbours_direction : ndarray of shape (2, n_total_nodes, kn)
            Direction vectors of the k nearest neighbors for each node.
        """
        # Reshape the `nodes_direction` array to match the neighbor indices
        self.neighbours_direction = np.reshape(self.nodes_direction, (2, self.par.n_nodes * self.par.n_bact))[:, self.nei.ind]


    def nodes_to_neighbours_euclidian_direction(self, x_nodes, y_nodes, ind):
        """
        Computes the direction between each node and its k-nearest neighbors using Euclidean distance.
        
        Parameters:
        -----------
        x_nodes : ndarray
            x-coordinates of the nodes.
        y_nodes : ndarray
            y-coordinates of the nodes.
        ind : ndarray
            Indices of the k-nearest neighbors for each node.
        
        Returns:
        --------
        x_dir : ndarray
            Normalized x-direction vectors between each node and its k-nearest neighbors.
        y_dir : ndarray
            Normalized y-direction vectors between each node and its k-nearest neighbors.
        neighbours_distance : ndarray
            Distances to the k-nearest neighbors.
        """
        
        # Flatten the array of neighbor indices for easier processing
        ind_flat = np.concatenate(ind)

        # Repeat the coordinates of the nodes to match the number of neighbors
        x, y = np.repeat(x_nodes, self.par.kn), np.repeat(y_nodes, self.par.kn)

        # Compute the direction vectors between the nodes and their k-nearest neighbors
        x_dir = x_nodes[ind_flat] - x
        y_dir = y_nodes[ind_flat] - y

        # Compute the Euclidean distance between the nodes and their neighbors
        norm_dir = np.linalg.norm(np.array([x_dir, y_dir]), axis=0)
        neighbours_distance = norm_dir.copy()
        
        # Handle the case where the direction is a zero vector (for overlapping neighbors)
        norm_dir[norm_dir == 0] = np.inf

        # Return the normalized direction vectors and distances
        return np.reshape(x_dir / norm_dir, ind.shape), np.reshape(y_dir / norm_dir, ind.shape), neighbours_distance


    def nodes_torus_direction(self, data):
        """
        Computes the direction between nodes on a toroidal space, considering periodic boundary conditions.
        
        This method first computes the Euclidean direction and distance, then adjusts for the toroidal distance 
        by checking if the nodes are "wrapped around" and applying the periodic boundary conditions.
        
        Parameters:
        -----------
        data : ndarray
            The node positions for which the direction and distance will be computed.
        
        Returns:
        --------
        nodes_torus_direction : ndarray
            The direction vectors between nodes considering periodic boundary conditions.
        nodes_torus_distance : ndarray
            The distance between nodes considering periodic boundary conditions.
        """
        
        # Compute the Euclidean direction and distance between nodes
        nodes_euclidian_direction, nodes_euclidian_distance = self.nodes_euclidian_direction(data)
        
        # Compute the toroidal distance between consecutive nodes
        nodes_torus_distance = self.torus_dist(a=data[:, :-1, :], b=data[:, 1:, :])
        nodes_torus_distance = np.concatenate((nodes_torus_distance[0, :][None, :], nodes_torus_distance), axis=1)

        # Identify nodes where the Euclidean and toroidal distances are different
        cond_dist = ~np.isclose(nodes_euclidian_distance, nodes_torus_distance)

        # Handle boundary conditions for nodes near the edge of the space
        cond_bottom_diagonal = (data[0, :-1, :] + data[1, :-1, :]) < self.par.space_size
        cond_bottom_anti_diagonal = (np.abs(data[0, :-1, :] - self.par.space_size) + data[1, :-1, :]) < self.par.space_size

        data_tmp = data.copy()
        data_tmp[0, :-1, :][cond_dist & cond_bottom_diagonal & ~cond_bottom_anti_diagonal] += self.par.space_size
        data_tmp[0, :-1, :][cond_dist & ~cond_bottom_diagonal & cond_bottom_anti_diagonal] -= self.par.space_size
        data_tmp[1, :-1, :][cond_dist & cond_bottom_diagonal & cond_bottom_anti_diagonal] += self.par.space_size
        data_tmp[1, :-1, :][cond_dist & ~cond_bottom_diagonal & ~cond_bottom_anti_diagonal] -= self.par.space_size

        # Compute the toroidal direction using adjusted coordinates
        nodes_torus_direction, __ = nodes_euclidian_direction(data_tmp)

        # For nodes with the same Euclidean and toroidal distances, use the Euclidean direction
        nodes_torus_direction[:, ~cond_dist] = nodes_euclidian_direction[:, ~cond_dist]

        # Return the toroidal direction and distance
        return nodes_torus_direction, nodes_torus_distance


    def set_nodes_to_neighbours_direction_torus(self):
        """
        Computes the direction between nodes and their k-nearest neighbors in a toroidal space.
        
        This method adjusts the positions of the neighbors for periodic boundary conditions and calculates the 
        direction vectors between nodes and their k-nearest neighbors, considering toroidal distance.
        """
        # Flatten the array of the neighbours indices
        x_nei_coord = self.nei.coord[:, 0][self.nei.ind_flat]
        y_nei_coord = self.nei.coord[:, 1][self.nei.ind_flat]

        # Define the coordinate of each bacterium self.k times (for each neighbour)
        x, y = np.repeat(self.nei.coord[:, 0], self.par.kn), np.repeat(self.nei.coord[:, 1], self.par.kn)

        # Compute the direction between the bacterium and their k-neighbour
        dist_euclidian_flat = np.sqrt((x_nei_coord - x)**2 + (y_nei_coord - y)**2)

        cond_dist = ~np.isclose(dist_euclidian_flat, self.nei.dist_flat)

        cond_bottom_diagonal = (x_nei_coord + y_nei_coord) < self.par.space_size
        cond_bottom_anti_diagonal = (np.abs(x_nei_coord - self.par.space_size) + y_nei_coord) < self.par.space_size

        x_nei_coord[cond_dist & cond_bottom_diagonal & ~cond_bottom_anti_diagonal] += self.par.space_size
        x_nei_coord[cond_dist & ~cond_bottom_diagonal & cond_bottom_anti_diagonal] -= self.par.space_size
        y_nei_coord[cond_dist & cond_bottom_diagonal & cond_bottom_anti_diagonal] += self.par.space_size
        y_nei_coord[cond_dist & ~cond_bottom_diagonal & ~cond_bottom_anti_diagonal] -= self.par.space_size

        # Compute the direction between the bacterium and their kn-neighbours
        x_dir = x_nei_coord - x
        y_dir = y_nei_coord - y

        # Normalise the previous directions
        norm_dir = np.linalg.norm(np.array([x_dir, y_dir]), axis=0)
        self.nodes_to_nei_dist[:, :] = np.reshape(norm_dir, self.nodes_to_nei_dist.shape)
        # Direction is null vector for superposed neighbours
        norm_dir[norm_dir == 0] = np.inf
        self.nodes_to_nei_dir_torus[:, :, :] = np.array([np.reshape(x_dir / norm_dir, self.nodes_to_nei_dist.shape),np.reshape(y_dir / norm_dir, self.nodes_to_nei_dist.shape)])
