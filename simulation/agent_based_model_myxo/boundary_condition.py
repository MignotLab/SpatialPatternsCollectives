"""
Author: Jean-Baptiste Saulnier
Date: 2024-12-06
"""
import numpy as np


class Boundaries:
    """
    This module defines the `Boundaries` class, which manages boundary conditions 
    for the nodes in the simulation. It includes support for periodic boundary 
    conditions to ensure continuity across the defined simulation space.

    Attributes:
    -----------
    par : object
        Instance of the `Parameters` class containing simulation settings.
    gen : object
        Instance of the class managing bacterial data generation.
    cond_boundary_l : np.ndarray
        Boolean array marking nodes exceeding the upper boundary of the simulation space.
    cond_boundary_0 : np.ndarray
        Boolean array marking nodes below the lower boundary of the simulation space.
    """
    def __init__(self, inst_par, inst_gen):
        # Store references to the `Parameters` and generation instances
        self.par = inst_par
        self.gen = inst_gen

        # Initialize boolean arrays for boundary conditions
        self.cond_boundary_l = np.zeros(self.gen.data.shape).astype(bool)  # Nodes exceeding the upper boundary
        self.cond_boundary_0 = np.zeros(self.gen.data.shape).astype(bool)  # Nodes below the lower boundary


    def periodic(self):
        """
        Apply periodic boundary conditions to nodes in the simulation space.
        
        Nodes exceeding the simulation space's upper boundary are repositioned to the lower boundary.
        Similarly, nodes below the lower boundary are repositioned to the upper boundary.
        """
        # Identify nodes exceeding the upper boundary
        self.cond_boundary_l = self.gen.data >= self.par.space_size

        # Identify nodes below the lower boundary
        self.cond_boundary_0 = self.gen.data < 0

        # Adjust positions of nodes exceeding the upper boundary
        self.gen.data[self.cond_boundary_l] -= self.par.space_size

        # Adjust positions of nodes below the lower boundary
        self.gen.data[self.cond_boundary_0] += self.par.space_size
