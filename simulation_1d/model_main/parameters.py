import cupy as cp


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

        ### PARALLELIZATION ###
        self.n_jobs = 6
        self.dtype = cp.float32

        ### NUMERIC PARAMETERS ###
        self.lx = 100 # longueur domaine spatial en µm
        self.dx = 0.05
        self.x = cp.arange(0, self.lx, self.dx, dtype=self.dtype)
        self.nx = len(self.x)

        self.lr = 6 #longueur de l'horloge interne (min)
        self.dr = 0.05 #puis tester 0.03 avec alpha = 0.1
        self.r = cp.arange(0, self.lr, self.dr, dtype=self.dtype)
        self.nr = len(self.r)
        # self.rrep = cp.reshape(cp.repeat(self.r, 2*self.nx), (2, self.nr, self.nx))
        self.rrep_tmp = cp.reshape(cp.repeat(self.r, self.nx), (self.nr, self.nx))
        self.rrep = cp.array([self.rrep_tmp, self.rrep_tmp])
        self.v0 = 4 # µm/min
        self.vr = 1 #vitesse horloge (R_P = 5 minutes)

        self.sigma = 0.25 # cfl
        self.dt = self.sigma * min(self.dx, self.dr) / max(self.v0, self.vr)

        self.save_frequency = 0.5 # in minutes
        self.save_frequency_kymo = 0.1 # in minutes
        self.start_time_save_kymo = 0

        ### REVERSAL FUNCTIONS ###
        self.rp_max = 5 # min
        self.rr_max = 3 # 1 / min


