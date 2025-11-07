"""
Author: Jean-Baptiste Saulnier
Date: 2024-12-06
"""
import numpy as np


class Velocity:
    """
    This class calculates the velocity and displacement of bacteria in a simulation.
    It tracks the positions of bacterial nodes over time, computes their displacement 
    and velocity, and calculates the magnitude of velocity for each bacterium. The class 
    provides methods to store positions before and after movement, calculate displacement, 
    and compute velocity.

    Attributes:
    -----------
    par : object
        Instance of the Parameters class containing simulation settings such as the number of bacteria, 
        number of nodes, and the time step for the simulation.
    pha : object
        Instance of the Phase class containing data on the positions of the bacteria (phantom data).
    coord_in : numpy.ndarray
        A 3D array (2, n_nodes, n_bact) storing the position of bacterial nodes before movement.
    coord_out : numpy.ndarray
        A 3D array (2, n_nodes, n_bact) storing the position of bacterial nodes after movement.
    displacement : numpy.ndarray
        A 3D array (2, n_nodes, n_bact) storing the displacement of each node between consecutive frames.
    velocity : numpy.ndarray
        A 3D array (2, n_nodes, n_bact) storing the velocity of each node (displacement per time step).
    velocity_norm : numpy.ndarray
        A 2D array (n_nodes, n_bact) storing the magnitude (norm) of the velocity for each bacterium.
    """
    def __init__(self, inst_par, inst_pha):
        # Instance objects: Assign the passed-in instances to class attributes.
        self.par = inst_par
        self.pha = inst_pha

        # Initialize arrays for storing coordinates, displacement, and velocity of bacteria
        self.coord_in = np.zeros((2, self.par.n_nodes, self.par.n_bact), dtype=self.par.float_type)
        self.coord_out = np.zeros((2, self.par.n_nodes, self.par.n_bact), dtype=self.par.float_type)
        self.displacement = np.zeros((2, self.par.n_nodes, self.par.n_bact), dtype=self.par.float_type)
        self.velocity = np.zeros((2, self.par.n_nodes, self.par.n_bact), dtype=self.par.float_type)
        self.velocity_norm = np.zeros((self.par.n_nodes, self.par.n_bact), dtype=self.par.float_type)


    def head_position_in(self):
        """
        Store the head position of all cells before they move.
        
        This method copies the current position of the bacteria (from phase data) 
        into the `coord_in` attribute for later displacement calculation.
        """
        self.coord_in = self.pha.data_phantom.copy()


    def head_position_out(self):
        """
        Store the head position of all cells after they move.
        
        This method copies the new position of the bacteria (from phase data) 
        into the `coord_out` attribute for displacement calculation.
        """
        self.coord_out = self.pha.data_phantom.copy()


    def displacement_in_out(self):
        """
        Compute the displacement of the bacteria between consecutive time points.
        
        The displacement is calculated as the difference between the position 
        after movement (`coord_out`) and the position before movement (`coord_in`).
        """
        self.displacement = self.coord_out - self.coord_in


    def velocity_in_out(self):
        """
        Compute the velocity of the bacteria between consecutive time points.
        
        The velocity is computed as the displacement divided by the time step 
        (`self.par.dt`), and the norm of the velocity is calculated for each bacterium.
        """
        self.velocity = self.displacement / self.par.dt
        self.velocity_norm = np.linalg.norm(self.velocity, axis=0)
