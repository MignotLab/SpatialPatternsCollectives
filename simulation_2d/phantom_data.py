import numpy as np


class Phantom:
    """
    Position of the nodes if there was not the boundary conditions
    
    """

    def __init__(self,inst_gen):
        
        self.gen = inst_gen
        self.data_phantom = self.gen.data.copy()
