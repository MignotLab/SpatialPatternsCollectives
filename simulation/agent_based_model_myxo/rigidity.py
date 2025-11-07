"""
Author: Jean-Baptiste Saulnier
Date: 2024-12-06
"""
import numpy as np


class RigidityTypeError(Exception):
    """
    Custom exception raised when an invalid rigidity type is specified.
    """
    pass


class Rigidity:
    """
    This class computes and applies rigidity forces to bacterial nodes based on the chosen rigidity type.
    The rigidity behavior is controlled by the `rigidity_type` parameter, which determines whether forces
    are applied to the bacteria or not.

    Attributes:
    -----------
    par : object
        Instance of the `Parameters` class containing physical constants and simulation settings.
    gen : object
        Instance of the class managing bacterial data generation (e.g., positions, velocities).
    pha : object
        Instance of the class managing phantom data for the simulation.
    force_correction : np.ndarray
        Array storing the force corrections applied to each node during the rigidity computation.
    alpha : np.ndarray
        Array storing intermediate values used to calculate rigidity forces between node triplets.
    chosen_rigidity_fonction : method
        The function selected based on the `rigidity_type` parameter to apply the rigidity forces or do nothing.
    """
    def __init__(self, inst_par, inst_gen, inst_pha):
        """
        Initializes the Rigidity class with the provided instances of parameters, general data, and phantom data.

        Arguments:
        inst_par -- instance of the `Parameters` class containing physical and simulation constants.
        inst_gen -- instance of the class managing bacterial data (e.g., positions, velocities).
        inst_pha -- instance of the class managing phantom data used for rigidity force calculations.
        """
        # Store references to the parameter, general, and phantom data objects
        self.par = inst_par  # Contains parameter values like the number of nodes, distance, and constants
        self.gen = inst_gen  # Contains general data related to bacteria, such as their positions and velocities
        self.pha = inst_pha  # Contains phantom data used for reference in force calculations

        # Initialize arrays to hold force corrections and intermediate rigidity values
        self.force_correction = np.zeros((2, self.par.n_nodes, self.par.n_bact), dtype=self.par.float_type)
        # Stores the corrections applied to each bacterial node in the x and y directions (2D forces).
        # Shape: (2, n_nodes, n_bact), where 2 corresponds to the x and y directions.
        
        self.alpha = np.zeros((2, self.par.n_nodes - 2, self.par.n_bact), dtype=self.par.float_type)
        # Stores the computed alpha values used in force calculations. These are based on node triplets.
        # Shape: (2, n_nodes-2, n_bact), where `n_nodes-2` reflects the number of triplet groups.

        # Select the rigidity function based on the `rigidity_type` parameter
        if self.par.rigidity_type == True:
            self.chosen_rigidity_fonction = self.rigidity_force_nodes  # Apply rigidity force if True
        elif self.par.rigidity_type == False:
            self.chosen_rigidity_fonction = self.function_doing_nothing  # Do nothing if False
        else:
            # Raise an error if `rigidity_type` is invalid
            print('rigidity_type could be: True or False; default is False\n')
            raise RigidityTypeError()


    def function_rigidity_type(self):
        """
        Executes the selected rigidity function based on the `rigidity_type` parameter.

        This method applies the rigidity forces or does nothing, depending on the configuration.
        """
        self.chosen_rigidity_fonction()  # Call the appropriate function to apply or skip rigidity forces


    def function_doing_nothing(self):
        """
        Placeholder function that does nothing, used when no rigidity forces are applied.

        This method is used when the `rigidity_type` is set to False, and no forces will be applied to the bacteria.
        """
        pass  # No action is taken here


    def rigidity_force_nodes(self):
        """
        Computes and applies rigidity forces based on the configuration of the bacteria's nodes.

        This method calculates the rigidity forces for each node triplet and updates the bacterial positions
        by applying the computed forces to the general and phantom data.
        """
        for i in range(self.par.rigidity_iteration):  # Repeat the rigidity force computation for a specified number of iterations

            # Compute w0 and w1, the differences between adjacent nodes, normalized by `d_n`
            w0 = (self.pha.data_phantom[:, :-2, :] - self.pha.data_phantom[:, 1:-1, :]) / self.par.d_n
            w1 = (self.pha.data_phantom[:, 2:, :] - self.pha.data_phantom[:, 1:-1, :]) / self.par.d_n
            
            # Calculate alpha, the product of w0 and w1, used for force adjustments
            self.alpha[0, :, :] = np.sum(w0 * w1, axis=0)  # Alpha for the first direction
            self.alpha[1, :, :] = np.sum(w0 * w1, axis=0)  # Alpha for the second direction

            # Calculate the forces (v0, v1, v2) for each node triplet based on rigidity
            v0 = - self.par.k_rigid * (w1 - self.alpha * w0)  # Force for the first set of nodes in the triplet
            v2 = - self.par.k_rigid * (w0 - self.alpha * w1)  # Force for the second set of nodes
            v1 = (- v0 - v2).copy()  # Force for the middle node in the triplet (calculated as negative sum)

            # Update the force corrections for the nodes in each triplet group
            self.force_correction[:, :-2, :] += v0  # Apply force to the first set of nodes
            self.force_correction[:, 1:-1, :] += v1  # Apply force to the middle nodes
            self.force_correction[:, 2:, :] += v2  # Apply force to the last set of nodes

            # Apply the computed force corrections to update the bacteria's positions
            self.gen.data[:, self.par.rigidity_first_node:, :] += self.force_correction[:, self.par.rigidity_first_node:, :] * self.par.dt
            self.pha.data_phantom[:, self.par.rigidity_first_node:, :] += self.force_correction[:, self.par.rigidity_first_node:, :] * self.par.dt
            # Positions are updated by adding the force corrections multiplied by the squared time step (`dt^2`)
