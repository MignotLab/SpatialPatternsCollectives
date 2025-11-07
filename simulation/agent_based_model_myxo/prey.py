"""
Author: Jean-Baptiste Saulnier
Date: 2024-12-16
"""
import numpy as np


class Prey:
    """
    
    """
    def __init__(self, inst_par, inst_gen):
         # Store references to external instances
        self.par = inst_par
        self.gen = inst_gen

        # Initialize prey map parameters and grid settings
        self.edges_width = 2 * self.par.pili_length  # Length of prey effect map edges in µm
        self.l = self.par.space_size  # Length of the space
        self.l_prey_map = self.par.space_size + 2 * self.edges_width  # Total prey effect space length in µm
        self.bins = int(self.l_eps / self.par.width_bins)  # Number of bins in the prey grid
        self.edges_width_bins = int(self.edges_width * self.bins / self.l_eps)  # Edge bin width
        
        self.prey_grid, _, _ = np.histogram2d(
            self.gen.data[0, :, self.par.first_index_prey_bact:].flatten(),
            self.gen.data[1, :, self.par.first_index_prey_bact:].flatten(),
            bins=self.bins,
            range=[[-self.edges_width, self.l + self.edges_width], [-self.edges_width, self.l + self.edges_width]]
        )
        self.prey_grid = self.prey_grid.astype(self.par.float_type) * self.par.deposit_amount * self.par.v0 * self.par.dt * np.exp(-self.rate_eps_evaporation * self.par.dt)
        