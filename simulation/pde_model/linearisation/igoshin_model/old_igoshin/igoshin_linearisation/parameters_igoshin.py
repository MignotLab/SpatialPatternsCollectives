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


        # Matrices size
        self.n = int(self.phi_max / self.d_phi)
        self.n_rp = int(self.delta_phi / self.d_phi)
        

        # Parameters to vary
        self.rho_w = 0.5 # \rho_w
        self.rho_bar_max = 4 * self.rho_w # Maximum of \bar\rho to compute eigenvalues
        self.d_rho_bar = 0.1 # \matrm{d}\rho
        self.rho_bar_array = np.arange(1, self.rho_bar_max+self.d_rho_bar, self.d_rho_bar)
        self.q_array = np.arange(2, 20, 0.5)
        self.xi_array = np.arange(0, 6, 0.05)
        # Grids creation for each parameter that vary
        self.rho_bar_grid, self.q_grid, self.xi_grid = np.meshgrid(self.rho_bar_array, self.q_array, self.xi_array)
        # Obtain all the parameters combination in a 2D array
        self.combined_array = np.vstack((self.rho_bar_grid.flatten(), self.q_grid.flatten(), self.xi_grid.flatten())).T
        

        # Parameters to vary rp modulated
        self.rho_w_rp = 0.5 # \rho_w
        self.rho_bar_max_rp = 4 * self.rho_w_rp # Maximum of \bar\rho to compute eigenvalues
        self.d_rho_bar_rp = 0.05 # \matrm{d}\rho
        self.rho_bar_array_rp = np.arange(self.d_rho_bar_rp, self.rho_bar_max_rp+self.d_rho_bar_rp, self.d_rho_bar_rp)
        self.rho_t_array_rp = self.rho_bar_array_rp.copy()
        self.xi_array_rp = np.arange(0, 6, 0.05)
        # Parameters for regulated refractory period
        self.rho_bar_grid_rp, self.rho_t_grid_rp, self.xi_grid_rp = np.meshgrid(self.rho_bar_array_rp, self.rho_t_array_rp, self.xi_array_rp)
        self.combined_array_rp_modulated = np.vstack((self.rho_bar_grid_rp.flatten(), self.rho_t_grid_rp.flatten(), self.xi_grid_rp.flatten())).T
