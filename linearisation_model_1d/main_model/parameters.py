import numpy as np


class Parameters:

    def __init__(self):

        self.n_jobs = 24

        self.f_star = 2 # F* value for 1d plot
        self.r_star = 5 # R* value for 1d plot
        self.ds = 0.05
        self.delta = 1 / self.ds
        self.n = self.f_star * (self.r_star + 2.5) # maximum of aging in minutes (should be less than the maximal refractory period)
        self.ns = int(self.n / self.ds)
        self.s = np.arange(0, self.ns, 1)
        self.v = 4 # velocity of myxo in µm/min
        self.rho_t = 0.5

        # Plot
        self.fontsize = 40
        self.figsize = (8,8)

        # index
        self.index_diag = np.arange(self.ns)
        self.index_sub_diag = np.arange(self.ns-1)
        self.array_tmp = np.zeros(self.ns)

        # Parameter array
        self.xi_array = np.arange(0, 6, 0.05)
        self.S_array = np.arange(self.ds, self.n, 5*self.ds)
        self.C_array = np.arange(0, 1.2, 0.01)

        # Objects for parallelisation
        # 1D plot
        self.xi_grid_1, self.S_grid_1 = np.meshgrid(self.xi_array, self.S_array, indexing='ij')
        self.combined_array_1 = np.vstack((self.xi_grid_1.flatten(), self.S_grid_1.flatten())).T
        # 2D plot
        self.xi_grid_2, self.S_grid_2, self.C_grid = np.meshgrid(self.xi_array, self.S_array, self.C_array, indexing='ij')
        self.combined_array_2 = np.vstack((self.xi_grid_2.flatten(), self.S_grid_2.flatten(), self.C_grid.flatten())).T
