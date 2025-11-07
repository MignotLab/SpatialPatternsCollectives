import numpy as np
import warnings

class Velocity:
    """
    Forward-difference velocities:
      - scalar speed at index i uses positions i -> i+1
      - vector velocity vr[:, i] uses positions i -> i+1
      - NaN at the last frame of each trajectory (no forward step)
      - NaN where Δt == 0
    """

    def __init__(self, par, track_id):
        # Store parameters (expects par.tbf and par.n_nodes to exist)
        self.par = par

        # ---- Trajectory boundary markers ----
        # True at indices i where i -> i+1 crosses into a different trajectory,
        # and at the globally last index (no forward step available).
        self.cond_traj_end = np.concatenate((
            track_id[1:] != track_id[:-1],
            np.array([True])
        ))

        # True at the very first sample of each trajectory
        self.cond_traj_start = np.concatenate((
            np.array([True]),
            track_id[1:] != track_id[:-1]
        ))

        # Backward-compatibility name (some code may expect this)
        self.cond_change_traj = self.cond_traj_end.copy()

        # Allocate outputs
        self.velocity = np.full(len(track_id), np.nan)   # scalar speed
        self.vr       = np.full((2, len(track_id)), np.nan)  # vector velocity
        self.vr_s     = np.full((2, len(track_id)), np.nan)  # smoothed vector velocity
        self.vt       = None  # target direction (set in compute_vt)


    def compute_velocity(self, x, y, t):
        """
        Compute scalar speed with forward differences:
          velocity[i] = ||(x[i+1]-x[i], y[i+1]-y[i])|| / (Δt[i] * par.tbf)
        Notes:
          - NaN at the end of each trajectory and when Δt == 0
        """
        N = len(x)
        self.velocity[:] = np.nan

        if N < 2:
            warnings.warn("compute_velocity: insufficient data points (N < 2). Returning NaN array.")
            return

        dt = t[1:] - t[:-1]
        dx = x[1:] - x[:-1]
        dy = y[1:] - y[:-1]

        with np.errstate(divide='ignore', invalid='ignore'):
            speed = np.sqrt(dx*dx + dy*dy) / (dt * self.par.tbf)

        # Valid where not end-of-trajectory and dt != 0
        valid = (~self.cond_traj_end[:-1]) & (dt != 0)
        self.velocity[:-1][valid] = speed[valid]
        # All other indices remain NaN


    def compute_vt(self, x0, y0, x1, y1, xm, ym, xn, yn, main_pole):
        """
        Compute the target direction of the main pole.
        For a main pole at index 0: direction = [x0 - x1, y0 - y1]
        For a main pole at index n_nodes-1: direction = [xn - xm, yn - ym]
        The result is normalized to unit vectors where defined.
        """
        # Initialize with NaNs
        self.vt = np.full((2, len(x0)), np.nan)

        # Neighbor-based raw vectors
        v0 = np.array([x0 - x1, y0 - y1])   # for main_pole == 0
        vn = np.array([xn - xm, yn - ym])   # for main_pole == n_nodes - 1

        # Conditions on which end is the main pole
        cond_pole_0 = (main_pole == 0)
        cond_pole_n = (main_pole == self.par.n_nodes - 1)

        # Assign raw vectors
        self.vt[:, cond_pole_0] = v0[:, cond_pole_0]
        self.vt[:, cond_pole_n] = vn[:, cond_pole_n]

        # Normalize (avoid division by zero)
        norms = np.linalg.norm(self.vt, axis=0)
        zero_norm = (norms == 0) | np.isnan(norms)
        norms[zero_norm] = 1.0
        self.vt = self.vt / norms
        # Entries that were NaN remain NaN; zero vectors become [0,0] after division (kept as zeros)


    def compute_vr(self, x, y, t, align="backward"):
        """
        Compute centroid vector velocity with finite differences.

        align:
        - "forward":  vr[:, i] = (x[i+1]-x[i], y[i+1]-y[i]) / Δt[i]      (step i -> i+1)
                        → invalid at END of each trajectory
        - "backward": vr[:, i] = (x[i]-x[i-1], y[i]-y[i-1]) / Δt[i-1]    (step i-1 -> i)
                        → invalid at START of each trajectory

        Notes:
        - NaN where Δt == 0
        - Reinitializes self.vr at each call
        """
        import warnings
        N = len(x)
        self.vr = np.full((2, N), np.nan)

        if N < 2:
            warnings.warn("compute_vr: insufficient data points (N < 2). Returning NaN array.")
            return

        dt = t[1:] - t[:-1]
        dx = x[1:] - x[:-1]
        dy = y[1:] - y[:-1]

        if align == "forward":
            # valid for indices i = 0..N-2 that are NOT end-of-trajectory and dt != 0
            valid = (~self.cond_traj_end[:-1]) & (dt != 0)
            self.vr[0, :-1][valid] = dx[valid] / dt[valid]
            self.vr[1, :-1][valid] = dy[valid] / dt[valid]

        elif align == "backward":
            # valid for indices i = 1..N-1 that are NOT start-of-trajectory and dt != 0
            # (use cond_traj_start because backward diff uses (i-1)->i)
            valid = (~self.cond_traj_start[1:]) & (dt != 0)
            self.vr[0, 1:][valid] = dx[valid] / dt[valid]
            self.vr[1, 1:][valid] = dy[valid] / dt[valid]

        else:
            raise ValueError("compute_vr: 'align' must be 'forward' or 'backward'")


    def compute_vr_s(self, xs, ys, t, align="backward"):
        """
        Compute smoothed centroid vector velocity with finite differences.

        align:
        - "forward":  vr_s[:, i] = (xs[i+1]-xs[i], ys[i+1]-ys[i]) / Δt[i]     (step i -> i+1)
                        → invalid at END of each trajectory
        - "backward": vr_s[:, i] = (xs[i]-xs[i-1], ys[i]-ys[i-1]) / Δt[i-1]   (step i-1 -> i)
                        → invalid at START of each trajectory

        Notes:
        - NaN where Δt == 0
        - Reinitializes self.vr_s at each call
        """
        import warnings
        N = len(xs)
        self.vr_s = np.full((2, N), np.nan)

        if N < 2:
            warnings.warn("compute_vr_s: insufficient data points (N < 2). Returning NaN array.")
            return

        dt = t[1:] - t[:-1]
        dx = xs[1:] - xs[:-1]
        dy = ys[1:] - ys[:-1]

        if align == "forward":
            # valid for indices i = 0..N-2 that are NOT end-of-trajectory and dt != 0
            valid = (~self.cond_traj_end[:-1]) & (dt != 0)
            self.vr_s[0, :-1][valid] = dx[valid] / dt[valid]
            self.vr_s[1, :-1][valid] = dy[valid] / dt[valid]

        elif align == "backward":
            # valid for indices i = 1..N-1 that are NOT start-of-trajectory and dt != 0
            valid = (~self.cond_traj_start[1:]) & (dt != 0)
            self.vr_s[0, 1:][valid] = dx[valid] / dt[valid]
            self.vr_s[1, 1:][valid] = dy[valid] / dt[valid]

        else:
            raise ValueError("compute_vr_s: 'align' must be 'forward' or 'backward'")
