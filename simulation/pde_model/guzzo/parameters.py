import numpy as np


class Parameters:


    def __init__(self):

        ### PARALLELIZATION ###
        self.n_jobs = 12

        ### NUMERIC PARAMETERS ###
        self.lx = 100 # longueur domaine spatial en µm
        self.dx = 0.1
        self.x = np.arange(0, self.lx, self.dx)
        self.nx = len(self.x)

        self.lr = 15 #longueur de l'horloge interne (min)
        self.dr = 0.1 #puis tester 0.03 avec alpha = 0.1
        self.r = np.arange(0, self.lr, self.dr)
        self.nr = len(self.r)
        # self.rrep = np.reshape(np.repeat(self.r, 2*self.nx), (2, self.nr, self.nx))
        self.rrep_tmp = np.reshape(np.repeat(self.r, self.nx), (self.nr, self.nx))
        self.rrep = np.array([self.rrep_tmp, self.rrep_tmp])
        self.v0 = 3 # µm/min
        self.vr = 1 #vitesse horloge (R_P = 5 minutes)

        self.sigma = 0.25 # cfl
        self.dt = self.sigma * min(self.dx, self.dr) / max(self.v0, self.vr)

        self.save_frequency = 1
        self.save_frequency_kymo = 0.1
        self.start_time_save_kymo = 0

        ### REVERSAL FUNCTIONS ###
        self.rp_max = 5 # min
        self.rr_max = 10 # 1 / min


