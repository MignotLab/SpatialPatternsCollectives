import numpy as np


class Parameters:

    def __init__(self):
        
        self.n_jobs = 24
        

        # Parameters
        self.phi_max = np.pi
        self.dp = 0.1 # \Delta s
        self.w_0 = 0.2 * self.phi_max # \omega_0
        self.signal_max = 6
        self.w_n = self.signal_max * self.w_0 # \omega_n
        self.delta_phi_r = 0.2 * self.phi_max # \Delta_{Phi_R}
        self.q_value_constant = 3 # When q is fixed
        self.v = 8 / self.w_0
        self.rho_w = 1 # \rho_w
        self.d_rho_bar = 0.01 # \matrm{d}\rho
        self.rho_t = self.rho_w
        self.rho_bar_max = 4 * self.rho_w # Maximum of \bar\rho to compute eigenvalues
        self.rho_bar_min = (self.rho_w**self.q_value_constant * self.dp / (self.w_n - self.dp))**(1 / self.q_value_constant) + self.dp * 1e-8
        self.rho_bar_min_rp = self.delta_phi_r * self.rho_t * self.w_n / (np.pi * self.w_0)


        # Matrices size
        self.n = int(self.phi_max / self.dp)
        self.n_rp = int(self.delta_phi_r / self.dp)
        self.index_diag = np.arange(self.n)
        self.index_sub_diag = np.arange(self.n - 1)
        

        # Plot
        self.fontsize = 40
        self.figsize = (8,8)


        # Parameters to vary
        n_step = 20
        self.xi_array = np.linspace(0, 1, n_step*2)
        # self.rho_bar_array = np.arange(self.rho_bar_min, self.rho_bar_max+self.d_rho_bar, self.d_rho_bar)
        self.rho_bar_array = np.linspace(0, 8*self.rho_w, n_step)
        # self.rho_bar_rp_array = np.arange(self.rho_bar_min_rp, self.rho_bar_max, (self.rho_bar_max-self.rho_bar_min_rp) / n_step)
        # print("rho_bar_rp_array=", self.rho_bar_rp_array)
        self.S_array = np.linspace(0, self.signal_max, n_step)
        self.K_1_array = np.linspace(0, 0.6, n_step)
        # self.phi_r_array = np.linspace(self.dp, np.pi-self.dp, n_step)
        

        # Combined array when only S is modulated  (for meshrid > 2 use indexing='ij')
        self.xi_grid_1, self.S_grid_1 = np.meshgrid(self.xi_array, self.S_array, indexing='ij')
        self.combined_array_1 = np.vstack((self.xi_grid_1.flatten(), self.S_grid_1.flatten())).T


        # Combined array when only rho_bar is modulated  (for meshrid > 2 use indexing='ij')
        self.xi_grid_2, self.rho_bar_grid_2 = np.meshgrid(self.xi_array, self.rho_bar_array, indexing='ij')
        self.combined_array_2 = np.vstack((self.xi_grid_2.flatten(), self.rho_bar_grid_2.flatten())).T


        # Combined array when only rho_bar is modulated (rp modulated)
        self.xi_grid_3, self.S_grid_3 = np.meshgrid(self.xi_array, self.S_array, indexing='ij')
        self.combined_array_3 = np.vstack((self.xi_grid_3.flatten(), self.S_grid_3.flatten())).T
        
        
        # Combined array when only w_1 and K_1 are modulated
        self.xi_grid_4, self.S_grid_4, self.K_1_grid_4 = np.meshgrid(self.xi_array, self.S_array, self.K_1_array, indexing='ij')
        self.combined_array_4 = np.vstack((self.xi_grid_4.flatten(), self.S_grid_4.flatten(), self.K_1_grid_4.flatten())).T


        # # Combined array when only delta_phi and K_2 are modulated (rp modulated)
        # self.xi_grid_5, self.S_grid_5, self.K_1_grid_5 = np.meshgrid(self.xi_array, self.S_array, self.K_1_array, indexing='ij')
        # self.combined_array_5 = np.vstack((self.xi_grid_5.flatten(), self.S_grid_5.flatten(), self.K_1_grid.flatten())).T