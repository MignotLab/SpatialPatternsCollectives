import numpy as np
import pandas as pd
from tqdm import tqdm

import parameters


class Velocity:


    def __init__(self, track_id):
        
        # Import class parameters
        self.par = parameters.Parameters()

        # Class object
        self.velocity = np.zeros(len(track_id)) * np.nan

        # Build a boolean array to detect the change of trajectories
        self.cond_change_traj = track_id[1:] != track_id[:-1]
        self.cond_change_traj = np.concatenate((self.cond_change_traj,np.array([True])))


    def compute_velocity(self, x, y, t):
        """
        Compute the velocity on the trajectories
        
        """
        time_diff = t[1:] - t[:-1]
        time_diff[time_diff==0] = 1
        self.velocity[:-1] = np.linalg.norm(np.array([x[1:]-x[:-1], y[1:]-y[:-1]]), axis=0) / (time_diff * self.par.tbf)

        # Remove the velocity computed between trajectories
        self.velocity[self.cond_change_traj] = np.nan


    def compute_vt(self, x0, y0, x1, y1, xm, ym, xn, yn, main_pole, reversals):
        """
        Compute the target velocity of the main pole
        
        """
        # Initialize velocity target
        self.vt = np.ones((2,len(x0))) * np.nan

        # Compute the velocity
        v0 = np.array([x0-x1, y0-y1])
        vn = np.array([xn-xm, yn-ym])

        cond_pole_0 = main_pole == 0
        cond_pole_n = main_pole == self.par.n_nodes - 1

        self.vt[:, cond_pole_0] = v0[:, cond_pole_0]
        self.vt[:, cond_pole_n] = vn[:, cond_pole_n]

        norm_vt = np.linalg.norm(self.vt, axis=0)
        norm_vt[norm_vt==0] = 1
        self.vt = self.vt / norm_vt


    # def compute_vr(self, x0, y0, xn, yn, t, main_pole, reversals):
    #     """
    #     Compute the real velocity of the main pole
        
    #     """
    #     # Initialize velocity target
    #     self.vr = np.ones((2,len(x0))) * np.nan

    #     # Compute the velocity
    #     time_diff = t[1:] - t[:-1]
    #     time_diff[time_diff==0] = 1
    #     v0 = np.array([x0[1:]-x0[:-1], y0[1:]-y0[:-1]]) / time_diff
    #     v0 = np.concatenate((v0.T, np.array([v0[:,-1]]))).T
    #     vn = np.array([xn[1:]-xn[:-1], yn[1:]-yn[:-1]]) / time_diff
    #     vn = np.concatenate((vn.T, np.array([vn[:,-1]]))).T

    #     cond_pole_0 = main_pole == 0
    #     cond_pole_n = main_pole == self.par.n_nodes - 1
    #     cond_rev = reversals.astype(bool)

    #     self.vr[:, cond_pole_0 & ~self.cond_change_traj & ~cond_rev] = v0[:, cond_pole_0 & ~self.cond_change_traj & ~cond_rev]
    #     self.vr[:, cond_pole_n & ~self.cond_change_traj & ~cond_rev] = vn[:, cond_pole_n & ~self.cond_change_traj & ~cond_rev]


    def compute_vr(self, x, y, t, reversals):
        """
        Compute the real velocity of the centroid
        
        """
        # Initialize velocity target
        self.vr = np.ones((2, len(x))) * np.nan

        # Compute the velocity
        time_diff = t[1:] - t[:-1]
        time_diff[time_diff==0] = 1
        v0 = np.array([x[1:]-x[:-1], y[1:]-y[:-1]]) / time_diff
        v0 = np.concatenate((v0.T, np.array([v0[:, -1]]))).T

        cond_rev = reversals.astype(bool)

        self.vr[:, ~self.cond_change_traj & ~cond_rev] = v0[:, ~self.cond_change_traj & ~cond_rev]


    def compute_vr_s(self, xs, ys, reversals):
        """
        Compute the target velocity of the main pole
        
        """
        # Initialize velocity target
        self.vr_s = np.ones((2,len(xs))) * np.nan

        # Compute the velocity
        vs = np.array([xs[1:]-xs[:-1],ys[1:]-ys[:-1]])
        vs = np.concatenate((vs.T,np.array([vs[:,-1]]))).T

        cond_rev = reversals.astype(bool)

        self.vr_s[:, ~self.cond_change_traj & ~cond_rev] = vs[:, ~self.cond_change_traj & ~cond_rev]
