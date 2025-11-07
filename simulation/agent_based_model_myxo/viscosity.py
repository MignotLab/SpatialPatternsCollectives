"""
Author: Jean-Baptiste Saulnier
Date: 2024-12-06
"""
import numpy as np


class Viscosity:
    """
    This class applies viscosity forces between the closest nodes of different bacteria based on their relative
    positions. The forces depend on the change in distances between nodes at two different time points.

    Attributes:
    -----------
    par : object
        Instance of the `Parameters` class containing simulation parameters such as the number of nodes, 
        viscosity constant (`epsilon_v`), and maximum node distance (`kn`).
    gen : object
        Instance of the class managing bacterial data, such as positions of nodes across bacteria.
    pha : object
        Instance of the class managing phantom data, used to apply viscosity forces in simulations.
    distance_t0 : np.ndarray
        Array of distances between nodes at time step t0, used for calculating the change in distance.
        The array is reshaped to accommodate all nodes across all bacteria.
    """
    def __init__(self, inst_par, inst_gen, inst_pha):
        # Store references to the parameter, general, and phantom data objects
        self.par = inst_par  # Contains simulation parameters like the number of nodes (n_nodes) and other constants
        self.gen = inst_gen  # Contains general data related to bacteria, such as positions of nodes
        self.pha = inst_pha  # Contains phantom data used in force calculations

        # Initialize the distance array with ones for all nodes at the initial time step (t0)
        self.distance_t0 = np.ones((int(self.par.n_nodes * self.par.n_bact), self.par.kn))
        # Shape: (n_nodes * n_bact, kn), where kn is the number of nearest neighbors per node


    def viscosity_bact(self, distance_t1, x_dir_n, y_dir_n, nodes_direction, cond_same_bact, max_dist):
        """
        Apply a viscosity force between the closest nodes which are not from the same bacterium.

        The force is calculated based on the change in the distance between nodes at two different time points
        and the direction of the nodes. The viscosity force depends on the relative movement of the nodes.

        Parameters
        ----------
        distance_t1 : np.ndarray
            Array of distances between nodes at time step t1, used to compute the change in distance.
        
        x_dir_n : np.ndarray
            x-components of the directions for each node.
        
        y_dir_n : np.ndarray
            y-components of the directions for each node.
        
        nodes_direction : np.ndarray
            Directions of the nodes in a 2D space (x, y).
        
        cond_same_bact : np.ndarray
            Condition array for identifying nodes belonging to the same bacterium, to exclude them from interaction.
        
        max_dist : float
            Maximum distance for considering two nodes as "close" and applying viscosity forces.

        Returns
        -------
        None
            This function modifies the `gen.data` and `pha.data_phantom` attributes in-place by applying viscosity forces.
        """
        # Condition on the distance between nodes at time step t0
        cond_dist = self.distance_t0 > max_dist  # Check if the distance is greater than the maximum allowed distance
        # Compute the change in distance between time step t0 and t1
        distance_change = distance_t1 - self.distance_t0
        # Condition where the distance change is negative, indicating the nodes are getting closer
        cond_dist_change = distance_change < 0

        # Reshape the nodes' directions into a 2D array
        direction = np.reshape(nodes_direction, (2, int(self.par.n_nodes * self.par.n_bact)))
        # Repeat the direction array for each nearest neighbor (kn), to create a 3D array for all neighbors
        direction = np.repeat(direction, self.par.kn).reshape(2, int(self.par.n_nodes * self.par.n_bact), self.par.kn)

        # Compute the viscosity force on each node
        visc_force = np.sign(np.sum(np.array([x_dir_n, y_dir_n]) * direction, axis=0)) * direction * distance_change
        # Apply conditions: exclude forces if the nodes are too far apart or if they are from the same bacterium
        visc_force[:, cond_dist | cond_dist_change | cond_same_bact] = np.nan
        # Sum over the neighbors (axis=2) to get the net viscosity force for each node
        visc_force = np.nansum(visc_force, axis=2)
        # Reshape the force array into the appropriate shape (2D array for each bacterium)
        visc_force = np.reshape(visc_force, (2, self.par.n_nodes, self.par.n_bact))

        # Apply the calculated viscosity force to update the positions of the nodes
        self.gen.data += self.par.epsilon_v * visc_force  # Apply force to the general data (bacteria positions)
        self.pha.data_phantom += self.par.epsilon_v * visc_force  # Apply force to the phantom data (used for simulation)
        
        # Update the reference distances for the next calculation (t1 becomes t0)
        self.distance_t0 = distance_t1.copy()  # Copy the distances from time step t1 to time step t0 for the next iteration