import numpy as np

class Velocity:


    def __init__(self,inst_par,inst_pha):
        
        self.par = inst_par
        self.pha = inst_pha

        # Class parameters
        self.coord_in = np.zeros((2,self.par.n_nodes,self.par.n_bact), dtype=self.par.float_type)
        self.coord_out = np.zeros((2,self.par.n_nodes,self.par.n_bact), dtype=self.par.float_type)
        self.displacement = np.zeros((2,self.par.n_nodes,self.par.n_bact), dtype=self.par.float_type)
        self.velocity = np.zeros((2,self.par.n_nodes,self.par.n_bact), dtype=self.par.float_type)
        self.velocity_norm = np.zeros((self.par.n_nodes,self.par.n_bact), dtype=self.par.float_type)


    def head_position_in(self):
        """
        Store the head position of all cells before they move
        
        """

        self.coord_in = self.pha.data_phantom.copy()

    
    def head_position_out(self):
        """
        Store the head position of all cells after they move

        """

        self.coord_out = self.pha.data_phantom.copy()


    def displacement_in_out(self):
        """
        Compute the displacement of the bacteria between consecutive space_time
        
        """

        self.displacement = self.coord_out - self.coord_in


    def velocity_in_out(self):
        """
        Compute the velocity of the bacteria between consecutive space_time
        
        """

        self.velocity = self.displacement / self.par.dt
        self.velocity_norm = np.linalg.norm(self.velocity, axis=0)

