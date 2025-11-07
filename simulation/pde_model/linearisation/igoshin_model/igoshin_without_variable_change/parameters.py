import numpy as np


class Parameters:

    def __init__(self):
        
        self.n_jobs = 2
        

        # Parameters
        self.phi_max = np.pi
        self.d_phi = 0.01 # \Delta\Phi
        self.w_0 = 0.2 * self.phi_max # \omega_0
        self.w_n = 3 * self.w_0 # \omega_n
        self.delta_phi_r = 0.2 * self.phi_max # \Delta_{Phi_R}
        self.q_value_constant = 4 # When q is fixed
        self.v = 8
        self.rho_w = 0.5 # \rho_w
        self.d_rho_bar = 0.05 # \matrm{d}\rho
        self.rho_bar_max = 4 * self.rho_w # Maximum of \bar\rho to compute eigenvalues
        self.rho_t = self.rho_w


        # Matrices size
        self.n = int(self.phi_max / self.d_phi)
        self.n_rp = int(self.delta_phi_r / self.d_phi)
        self.index_diag = np.arange(self.n)
        self.index_sub_diag = np.arange(self.n - 1)
        

        # Plot
        self.fontsize = 40
        self.figsize = (8,8)


        # Parameters to vary
        self.xi_array = np.arange(0, 6, 0.05)
        self.rho_bar_array = np.arange(self.d_rho_bar, self.rho_bar_max+self.d_rho_bar, self.d_rho_bar)
        self.rho_bar_rp_array = np.arange(self.delta_phi_r*self.rho_t/self.phi_max, self.rho_bar_max+self.d_rho_bar, self.d_rho_bar)
        self.q_array = np.arange(2, 20, 0.5)
        self.w_1_array = np.arange(0, 5, 5/50)
        self.K_1_array = np.arange(0, 1, 1/50)
        self.phi_r_array = np.arange(0, self.phi_max, self.phi_max/50)
        self.K_2_array = np.arange(0, 1, 1/50)
        

        # Combined array when only rho_bar is modulated  (for meshrid > 2 use indexing='ij')
        self.xi_grid_1, self.rho_bar_grid_1 = np.meshgrid(self.xi_array, self.rho_bar_array, indexing='ij')
        self.combined_array_1 = np.vstack((self.xi_grid_1.flatten(), self.rho_bar_grid_1.flatten())).T


        # Combined array when only rho_bar is modulated (rp modulated)
        self.xi_grid_2, self.rho_bar_grid_2 = np.meshgrid(self.xi_array, self.rho_bar_rp_array, indexing='ij')
        self.combined_array_2 = np.vstack((self.xi_grid_2.flatten(), self.rho_bar_grid_2.flatten())).T
        
        
        # Combined array when only w_1 and K_1 are modulated
        self.xi_grid_3, self.w_1_grid, self.K_1_grid = np.meshgrid(self.xi_array, self.w_1_array, self.K_1_array, indexing='ij')
        self.combined_array_3 = np.vstack((self.xi_grid_3.flatten(), self.w_1_grid.flatten(), self.K_1_grid.flatten())).T


        # Combined array when only delta_phi and K_2 are modulated (rp modulated)
        self.xi_grid_4, self.phi_r_grid, self.K_2_grid = np.meshgrid(self.xi_array, self.phi_r_array, self.K_2_array, indexing='ij')
        self.combined_array_4 = np.vstack((self.xi_grid_4.flatten(), self.phi_r_grid.flatten(), self.K_2_grid.flatten())).T