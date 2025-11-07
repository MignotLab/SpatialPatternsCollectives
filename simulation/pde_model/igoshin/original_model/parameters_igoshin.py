import cupy as cp
# import numpy as cp


class Parameters:


    def __init__(self):
        
        ### PLOTS ###
        self.size_height_figure = 7
        self.figsize = (self.size_height_figure, self.size_height_figure-1)
        self.dpi = 300
        self.fontsize = 30
        self.fontsize_ticks = self.fontsize / 1.5
        self.cmap = 'inferno_r'
        self.color_u = '#a559aa' # Purple
        self.color_v = 'grey'

        ### Parameters ###
        self.phi_mid = cp.pi
        self.w_0 = 0.2 * self.phi_mid
        self.w_n = 3 * self.w_0
        self.delta_phi_r = 0.2 * self.phi_mid
        self.q = 4
        self.v0 = 8
        self.dtype = cp.float32

        self.alpha_sigmoid = 100

        ### NUMERIC PARAMETERS ###
        self.lx = 100 # longueur domaine spatial en µm
        self.dx = 0.05
        self.x = cp.arange(0, self.lx, self.dx, dtype=self.dtype)
        self.nx = len(self.x)

        self.dp = 0.01 # d_phi
        self.p = cp.arange(0, self.phi_mid, self.dp, dtype=self.dtype) # phi
        self.np = len(self.p)

        self.rrep = cp.reshape(cp.repeat(self.p, self.nx), (self.np, self.nx))
        self.rrep = cp.concatenate((self.rrep, self.rrep), axis=0)

        self.sigma = 0.25 # CFL
        self.dt = self.sigma * min(self.dx, self.dp) / max(self.v0, self.w_0+self.w_n)

        self.save_frequency = 0.5 # in minutes
        self.save_frequency_kymo = 0.1 # in minutes
        self.start_time_save_kymo = 0
