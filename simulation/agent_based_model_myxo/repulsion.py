"""
Author: Jean-Baptiste Saulnier
Date: 2024-12-05
"""
import numpy as np


class RepulsionTypeError(Exception):
    """
    Custom exception to handle invalid repulsion type selection.
    """
    pass


class Repulsion:
    """
    This class computes the repulsion interactions between bacterial nodes in a
    simulation. The behavior of repulsion is controlled by the `repulsion_type` 
    parameter, which can be set to options such as "standard_repulsion", 
    "propagation_based_repulsion", or "perpendicular_repulsion". These 
    repulsive forces are applied based on the relative distances and alignments
    of the bacterial nodes, with conditions for handling neighboring nodes 
    belonging to the same bacterium or different bacteria.

    Attributes:
    -----------
    par : object
        Instance of the `Parameters` class containing simulation parameters, including repulsion type, bacterial dimensions, and simulation time step.
    gen : object
        Instance of a class managing the positions and states of bacterial nodes during the simulation.
    pha : object
        Instance of a class managing the positions and states of phantom bacterial nodes for additional effects.
    dir : object
        Instance of a class managing directional information of bacterial nodes, including alignment and directional forces.
    nei : object
        Instance of a class handling neighbor computations, including distances and relationships between bacterial nodes.
    chosen_repulsion_fonction : method
        Reference to the method selected for computing repulsion, based on the `repulsion_type` parameter.
    f_norm : ndarray
        Array storing the norm of repulsion forces between nodes, shaped `(n_bact * n_nodes, kn)`.
    rep_force : ndarray
        Array storing the repulsion forces applied to each node, shaped `(2, n_nodes, n_bact)`.
    f_rep_norm : ndarray
        Array storing the normalized repulsion forces between nodes, shaped `(n_bact * n_nodes, kn)`.
    f_rep_norm_ext : ndarray
        Array storing the normalized repulsion forces between nodes of different bacteria.
    f_rep_norm_int : ndarray
        Array storing the normalized repulsion forces between nodes of the same bacterium.
    f_rep : ndarray
        Array storing the total repulsion forces applied to nodes, reshaped to match bacterial node positions.
    f_rep_ext : ndarray
        Array storing the repulsion forces applied between bacteria, reshaped to match node positions.
    f_rep_int : ndarray
        Array storing the internal repulsion forces applied within bacteria, reshaped to match node positions.
    array_same_point : ndarray
        Array used to determine conditions for repulsion between nodes of the same bacterium, shaped `(n_bact * n_nodes, n_nodes)`.
    right_neighbour_node : ndarray
        Array indicating the right neighbor node index for each node in the same bacterium.
    left_neighbour_node : ndarray
        Array indicating the left neighbor node index for each node in the same bacterium.
    """
    def __init__(self, inst_par, inst_gen, inst_pha, inst_dir, inst_nei):
        # Store references to external class instances
        self.par = inst_par
        self.gen = inst_gen
        self.pha = inst_pha
        self.dir = inst_dir
        self.nei = inst_nei

        # Select the appropriate repulsion function based on the parameter.
        if self.par.repulsion_type == 'repulsion':
            self.chosen_repulsion_fonction = self.repulsion
        elif self.par.repulsion_type == 'repulsion_propagation':
            self.chosen_repulsion_fonction = self.repulsion_propagation
        elif self.par.repulsion_type == 'repulsion_perpendicular':
            self.chosen_repulsion_fonction = self.repulsion_perpendicular
        elif self.par.repulsion_type == 'no_repulsion':
            self.chosen_repulsion_fonction = self.function_doing_nothing
        else:
            # Inform the user of valid repulsion types and raise an exception if invalid.
            print('repulsion_type could be: "repulsion", "repulsion_propagation", "repulsion_perpendicular", "no_repulsion"; default is "repulsion"\n')
            raise RepulsionTypeError()

        ### Define class variables for storing intermediate computation results.
        # Repulsion forces and related arrays
        self.f_norm = np.zeros((int(self.par.n_bact * self.par.n_nodes), self.par.kn), dtype=self.par.float_type)
        self.rep_force = np.zeros((2, self.par.n_nodes, self.par.n_bact), dtype=self.par.float_type)
        self.f_rep_norm = np.zeros((self.par.n_bact * self.par.n_nodes, self.par.kn), dtype=self.par.float_type)

        # Additional arrays for specific repulsion methods
        self.f_rep_norm_ext = np.zeros(self.f_rep_norm.shape, dtype=self.par.float_type)
        self.f_rep_norm_int = np.zeros(self.f_rep_norm.shape, dtype=self.par.float_type)
        self.f_rep = np.zeros(self.gen.data.shape, dtype=self.par.float_type)
        self.f_rep_ext = np.zeros(self.gen.data.shape, dtype=self.par.float_type)
        self.f_rep_int = np.zeros(self.gen.data.shape, dtype=self.par.float_type)

        # Arrays for handling interactions between nodes of the same bacteria
        self.array_same_point = np.repeat(
            (np.ones((self.par.kn, self.par.n_nodes), dtype=self.par.float_type) * np.arange(self.par.n_nodes)).T.astype(self.par.int_type),
            np.ones(self.par.n_nodes, dtype=self.par.int_type) * self.par.n_bact,
            axis=0
        )
        self.right_neighbour_node = np.roll(self.array_same_point, shift=-self.par.n_bact, axis=0)
        self.left_neighbour_node = np.roll(self.array_same_point, shift=+self.par.n_bact, axis=0)


    def function_repulsion_type(self):
        """
        Execute the selected repulsion function based on the chosen type.
        """
        self.chosen_repulsion_fonction()


    def function_rep_norm(self, d, r, k_r):
        """
        Calculate repulsive force between two objects at distance r.
        The force is zero if r > d (beyond the range of repulsion).
        """
        force = -np.minimum(r - d, 0)**2 * (1 / np.maximum(r - 0.9 * d, 2 / k_r))
        # force = - k_r * np.minimum(r - d, 0)**2
        return force


    def function_rep_norm_overlap(self, r, max_force, mu1, mu2, sig):
        """
        Experimental force allowing overlap; decreases when cells are very close.
        """
        force = max_force * (np.exp(-np.power(r - mu1, 2) / (2 * np.power(sig, 2))) +
                             np.exp(-np.power(r - mu2, 2) / (2 * np.power(sig, 2))))
        force[(r > mu2) & (r < mu1)] = max_force
        return -force


    def function_doing_nothing(self):
        """
        No repulsion applied (used for the 'no_repulsion' type).
        """
        pass


    def repulsion(self):
        """
        Compute repulsion forces between bacterial nodes.
        Forces between nodes of the same bacterium are handled separately.
        """
        # Compute repulsion forces between nodes based on distances.
        self.f_norm = self.function_rep_norm(d=self.par.width, r=self.nei.dist, k_r=self.par.k_r)

        # Remove forces for adjacent nodes of the same bacterium.
        self.f_norm[self.nei.cond_same_bact & (self.right_neighbour_node == self.nei.id_node)][:int(self.par.n_bact * (self.par.n_nodes - 1))] = 0
        self.f_norm[self.nei.cond_same_bact & (self.left_neighbour_node == self.nei.id_node)][self.par.n_bact:] = 0

        # Sum repulsion forces and apply them to the nodes.
        self.rep_force[:, :, :] = np.reshape(
            np.sum(self.dir.nodes_to_nei_dir_torus[:, :, :] * self.f_norm, axis=2),
            self.gen.data[:, :, :].shape
        )

        self.gen.data[:, :, :self.gen.first_index_prey_bact] += self.rep_force[:, :, :self.gen.first_index_prey_bact] * self.par.dt
        self.pha.data_phantom[:, :, :self.gen.first_index_prey_bact] += self.rep_force[:, :, :self.gen.first_index_prey_bact] * self.par.dt


    def repulsion_propagation(self):
        """
        Apply repulsion forces and propagate the effects along the body of the bacterium.
        """
        # Compute repulsion using the basic method.
        self.repulsion()

        # Compute parallel force components and propagate along the body.
        force_norm_parallel = np.sum(self.rep_force[:, :, :] * self.dir.nodes_direction[:, :, :], axis=0)
        self.gen.data[:, 1:, :self.gen.first_index_prey_bact] += (self.dir.nodes_direction[:, 1:, :] * force_norm_parallel[0, :] * self.par.dt)[:, :, :self.gen.first_index_prey_bact]
        self.gen.data[:, :-1, :self.gen.first_index_prey_bact] += (self.dir.nodes_direction[:, :-1, :] * force_norm_parallel[-1, :] * self.par.dt)[:, :, :self.gen.first_index_prey_bact]
        self.pha.data_phantom[:, 1:, :self.gen.first_index_prey_bact] += (self.dir.nodes_direction[:, 1:, :] * force_norm_parallel[0, :] * self.par.dt)[:, :, :self.gen.first_index_prey_bact]
        self.pha.data_phantom[:, :-1, :self.gen.first_index_prey_bact] += (self.dir.nodes_direction[:, :-1, :] * force_norm_parallel[-1, :] * self.par.dt)[:, :, :self.gen.first_index_prey_bact]


    def repulsion_perpendicular(self):
        """
        Apply repulsion perpendicular to the local body shape of the neighboring bacterium.
        """
        # Calculate directional vectors and interaction distances.
        x_dir_n = self.dir.nodes_to_nei_dir_torus[0, :, :]
        y_dir_n = self.dir.nodes_to_nei_dir_torus[1, :, :]
        dir_nei = np.array([x_dir_n, y_dir_n])

        # Initialize distances of interaction and compute perpendicular forces.
        int_dist = np.ones((self.par.n_bact * self.par.n_nodes, self.par.kn)) * self.par.width
        length = np.abs(self.nei.id_node - self.array_same_point) * self.par.d_n
        cond_dist = length < self.par.width
        int_dist[self.nei.cond_same_bact & cond_dist] = length[self.nei.cond_same_bact & cond_dist] * self.par.d_n

        # Compute and apply perpendicular forces.
        self.f_rep_norm = self.function_rep_norm(d=int_dist, r=self.nei.dist, k_r=self.par.k_r)
        self.f_rep = np.reshape(np.sum(dir_nei * self.f_rep_norm, axis=2), self.gen.data.shape)

        self.gen.data[:, 1:-1, :self.gen.first_index_prey_bact] += self.f_rep[:, 1:-1, :self.gen.first_index_prey_bact] * self.par.dt
        self.pha.data_phantom[:, 1:-1, :self.gen.first_index_prey_bact] += self.f_rep[:, 1:-1, :self.gen.first_index_prey_bact] * self.par.dt
