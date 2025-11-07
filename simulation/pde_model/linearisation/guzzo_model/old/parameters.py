import numpy as np


class Parameters:

    def __init__(self):

        self.n_jobs = 24
        
        self.ds = 0.1
        # self.ds = 1
        self.delta = 1 / self.ds
        self.n = 15
        # self.n = 5
        self.ns = int(self.n / self.ds)
        # self.s = np.arange(0, self.n, self.ds)
        self.s = np.arange(0, self.ns, 1)

        self.S_array = np.arange(self.ds, self.n, 2*self.ds)
        self.C_array = np.arange(0, 0.6, 0.01)
        self.xi_array = np.arange(0, 6, 0.05)

        # self.S_array = np.arange(self.ds, self.n, 2*self.ds)
        # self.C_array = np.arange(0, 0.6, 0.1)
        # self.xi_array = np.arange(0, 6, 0.5)

        # Création des grilles pour chaque tableau
        self.S_grid, self.C_grid, self.xi_grid = np.meshgrid(self.S_array, self.C_array, self.xi_array)
        # Empilement des grilles pour obtenir un tableau 2D
        self.combined_array = np.vstack((self.S_grid.flatten(), self.C_grid.flatten(), self.xi_grid.flatten())).T
