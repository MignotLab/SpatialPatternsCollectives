import numpy as np

class Boundaries:

    def __init__(self,inst_par,inst_gen):
        
        # Parameters from parameters.py
        self.par = inst_par
        self.gen = inst_gen

        # Class objetcs
        self.cond_boundary_l = np.zeros(self.gen.data.shape).astype(bool)
        self.cond_boundary_0 = np.zeros(self.gen.data.shape).astype(bool)

    def periodic(self):
        """
        Apply periodic bounadry condition on the nodes
        
        """

        self.cond_boundary_l = self.gen.data >= (self.par.space_size - self.par.epsilon_float32)
        self.cond_boundary_0 = self.gen.data < (0 + self.par.epsilon_float32)

        self.gen.data[self.cond_boundary_l] -= (self.par.space_size + 1.1*self.par.epsilon_float32)
        self.gen.data[self.cond_boundary_0] += (self.par.space_size + 1.1*self.par.epsilon_float32)
