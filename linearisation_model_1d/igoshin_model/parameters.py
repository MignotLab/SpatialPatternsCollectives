import numpy as np


class Parameters:

    def __init__(self):
        
        self.n_jobs = 24
        

        # Parameters
        self.phi_max = np.pi
        self.dp = 0.05 # \Delta s
        self.w_0 = 0.2 * self.phi_max # \omega_0
        self.signal_max = 6
        self.w_star = self.signal_max * self.w_0 # \omega_n
        self.Phi_R_star = 0.2 * self.phi_max # \Phi_R, the refractory period
        self.q_value_constant = 3 # When q is fixed
        self.rho_w = 1 # \rho_w
        self.rho_t = self.rho_w


        # Matrices size
        self.n = int(self.phi_max / self.dp)
        self.index_diag = np.arange(self.n)
        self.index_sub_diag = np.arange(self.n - 1)


        # Parameters to vary
        self.n_step = 50
        self.xi_array = np.arange(0, 6, 1 / (2*self.n_step))
        self.S_array = np.linspace(0, self.signal_max, self.n)
        self.Phi_R_array = np.linspace(1, self.n, self.n) * self.dp
        # self.Phi_R_array_plot = np.linspace(self.phi_max/n_step, self.phi_max - self.phi_max/n_step, n_step)
        
        
        # Combined array
        self.xi_grid, self.S_grid, self.Phi_R_grid = np.meshgrid(self.xi_array, self.S_array, self.Phi_R_array, indexing='ij')
        self.combined_array = np.vstack((self.xi_grid.flatten(), self.S_grid.flatten(), self.Phi_R_grid.flatten())).T


        # Plots
        self.size_height_figure = 7
        self.figsize = (self.size_height_figure, self.size_height_figure-1)
        self.dpi = 300
        self.fontsize = 30
        self.fontsize_ticks = self.fontsize / 1.5
        self.alpha = 0.5
        self.linewidth = 4
