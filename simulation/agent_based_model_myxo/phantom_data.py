"""
Author: Jean-Baptiste Saulnier
Date: 2024-12-05

This module defines the `Phantom` class that manages the phantom positions of 
bacterial nodes, representing their theoretical positions without boundary 
conditions. It facilitates calculations of angles and distances between nodes, 
which can become complex when boundary conditions cause a cell to be split into 
distant segments. These calculations are crucial for managing node positions 
within cells effectively.
"""
class Phantom:
    """
    Represents a "phantom" layer of bacterial nodes that mirrors the current 
    positions from a generator object. This is used to simulate independent 
    effects on a copy of the bacterial data.

    Attributes:
    -----------
    gen : object
        An instance of the generator class, containing the current positions of 
        bacterial nodes.
    data_phantom : ndarray
        A copy of the node positions from the generator instance, used for 
        independent computations.
    """
    def __init__(self, inst_gen):
        # Store references to external class instances
        self.gen = inst_gen
        
        # Create a separate phantom layer by copying the generator's node positions
        self.data_phantom = self.gen.data.copy()