import numpy as np


class Parameters:

    def __init__(self):
        
        self.n_jobs = 22
        
        # Parameters
        self.phi_max = np.pi
        self.d_phi = 0.1 # \Delta\Phi
        self.w_0 = 0.2 * self.phi_max # \omega_0
        self.w_n = 3 * self.w_0 # \omega_n
        self.delta_phi = 0.2 * self.phi_max # \Delta_{Phi_R}
        self.rho_t = 0.5
        self.v = 10

        # Matrices size
        self.n = int(self.phi_max / self.d_phi)
        self.n_rp = int(self.delta_phi / self.d_phi)
        

        # Parameters to vary
        self.rho_w = 0.5 # \rho_w
        self.rho_bar_max = 4 * self.rho_w # Maximum of \bar\rho to compute eigenvalues
        self.d_rho_bar = 0.02 # \matrm{d}\rho
        self.rho_bar_array = np.arange(0.2*self.rho_t+self.d_rho_bar, self.rho_bar_max+self.d_rho_bar, self.d_rho_bar)
        self.xi_array = np.arange(0, 6, 0.05)
        # Grids creation for each parameter that vary
        self.rho_bar_grid, self.xi_grid = np.meshgrid(self.rho_bar_array, self.xi_array)
        # Obtain all the parameters combination in a 2D array
        self.combined_array = np.vstack((self.rho_bar_grid.flatten(), self.xi_grid.flatten())).T
