"""
Author: Jean-Baptiste Saulnier
Date: 2024-12-05
"""
import numpy as np


class MovementTypeError(Exception):
    """
    Custom exception raised when an invalid movement type is specified.
    """
    pass


class RandomMovementTypeError(Exception):
    """
    Custom exception raised when an invalid random movement type is specified.
    """
    pass


class Move:
    """
    This class handles the movement of bacteria in a simulation. The movement 
    behavior is controlled by the `movement_type` parameter, which can be set 
    to options such as "tracted". For random variation in the head set the 
    parameter `random_movement` to "True" with a specific "sigma_random".

    Attributes:
    -----------
    par : object
        Instance of the `Parameters` class defining simulation settings such as movement type and speed.
    gen : object
        Instance of the class managing bacterial data generation.
    pha : object
        Instance of the class managing phantom data for simulation purposes.
    dir : object
        Instance of the class managing directional data for bacterial nodes.
    uti : object
        Instance of a utility class providing helper functions.
    chosen_motility_fonction : method
        Selected function for determining bacterial motility, based on the `movement_type` parameter.
    chosen_random_fonction : method
        Selected function for applying random movement, or a placeholder function if disabled.
    v0_tracted : float
        The base velocity for tracted movement, scaled by the number of nodes in each bacterium.
    """
    def __init__(self, inst_par, inst_gen, inst_pha, inst_nei, inst_dir, inst_uti):
        # Store references to external class instances
        self.par = inst_par
        self.gen = inst_gen
        self.pha = inst_pha
        self.nei = inst_nei
        self.dir = inst_dir
        self.uti = inst_uti

        # Select the appropriate movement function based on the movement type
        if self.par.movement_type == 'tracted':
            self.chosen_motility_fonction = self.tracted_movement

        elif self.par.movement_type == 'pushed':
            self.chosen_motility_fonction = self.pushed_movement

        elif self.par.movement_type == 'tracted_stop_prey':
            self.chosen_motility_fonction = self.tracted_stop_prey_movement

        else:
            # Raise an error for invalid movement types
            print('movement_type could be: "tracted", "pushed", "tracted_stop_prey" \n')
            raise MovementTypeError()

        # Select the appropriate random movement function based on the random movement setting
        if self.par.random_movement:
            # Enable random movement if the parameter is True
            self.chosen_random_fonction = self.random_movement

        elif not self.par.random_movement:
            # Disable random movement by selecting a placeholder function
            self.chosen_random_fonction = self.function_doing_nothing

        else:
            # Raise an error for invalid random movement settings
            print('random_movement could be: True or False, default is False\n')
            raise RandomMovementTypeError()

        # Calculate the velocity for tracted movement, scaled by the number of nodes
        self.v0 = np.zeros(self.par.n_bact)
        self.v0[:self.gen.first_index_prey_bact] += self.par.v0
        self.v0_tracted = self.v0 * self.par.n_nodes
        # Array condition that is True for the predator and False for the prey
        self.cond_predator = np.zeros(self.par.n_bact, dtype=bool)
        self.cond_predator[:self.gen.first_index_prey_bact] = True


    def function_movement_type(self):
        """
        Executes the chosen movement and random movement functions based on the parameters.

        This method is the entry point for movement operations in each simulation step.
        """
        self.chosen_motility_fonction()  # Apply the selected motility function
        self.chosen_random_fonction()   # Apply random movement if enabled


    def function_doing_nothing(self):
        """
        A placeholder function that does nothing.

        This is used when random movement is disabled.
        """
        pass


    def random_movement(self):
        """
        Applies random displacement to the head of each bacterium.
        The displacement is determined by a Gaussian noise added to the direction.

        The method rotates the head node's position relative to the second node.
        """
        # Generate random noise for rotation angles
        noise = np.random.normal(loc=0, scale=self.par.sigma_random, size=self.par.n_bact).astype(self.par.float_type)
        # Construct rotation matrices for the generated noise
        rotation_matrix = self.uti.rotation_matrix_2d(theta=noise)

        # Update the positions of the head node in both data and phantom arrays
        self.gen.data[:, 0, :self.gen.first_index_prey_bact] = (
            np.sum(rotation_matrix * (self.pha.data_phantom[:, 0, :] - self.pha.data_phantom[:, 1, :]), axis=1)
            + self.gen.data[:, 1, :]
        )[:, :self.gen.first_index_prey_bact]
        self.pha.data_phantom[:, 0, :self.gen.first_index_prey_bact] = (
            np.sum(rotation_matrix * (self.pha.data_phantom[:, 0, :] - self.pha.data_phantom[:, 1, :]), axis=1) 
            + self.pha.data_phantom[:, 1, :]
        )[:, :self.gen.first_index_prey_bact]


    def pushed_movement(self):
        """
        Moves all nodes of a bacterium in the direction of their adjacent forward node.
        """
        # Update the positions of all nodes based on the direction and velocity
        self.gen.data[:, :, :] += self.par.v0 * self.dir.nodes_direction[:, :, :] * self.par.dt
        self.pha.data_phantom[:, :, :] += self.par.v0 * self.dir.nodes_direction[:, :, :] * self.par.dt


    def tracted_movement(self):
        """
        Moves the head of the chain, which pulls the other nodes along.
        """
        # Update the position of the head node based on the direction and velocity
        self.gen.data[:, 0, :] += self.v0_tracted * self.dir.nodes_direction[:, 0, :] * self.par.dt
        self.pha.data_phantom[:, 0, :] += self.v0_tracted * self.dir.nodes_direction[:, 0, :] * self.par.dt


    def tracted_stop_prey_movement(self):
        """
        Moves the head of the chain, which pulls the other nodes along, 
        with a probability to stop the movement if next to a prey
        """
        prob = 1 - np.exp(-self.par.rate_stop_at_prey * self.par.dt) * np.ones(self.par.n_bact)  # Probability of stopping at prey contact
        # Determine if the predator can stop depending on the probability to stops (prob)
        cond = np.random.binomial(1, prob).astype(bool)
        cond_stop = self.nei.cond_prey_neighbour & cond
        # The predator stops if it has a prey neighbours and if cond is True
        self.v0_tracted[cond_stop] = 0
        # If a predator has no prey neighbour (after a death) reset the velocity at its normal value
        self.v0_tracted[~self.nei.cond_prey_neighbour & self.cond_predator] = self.par.v0 * self.par.n_nodes
        self.tracted_movement()