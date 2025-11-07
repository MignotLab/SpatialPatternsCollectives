import numpy as np
import pandas as pd
from tqdm import tqdm
import os

from . import tools
from . import smoothed_trajectories

class ReversalsDetection:


    def __init__(self, par, df, end_min_size_smoothed_um, start_min_size_smoothed_um, step_min_size_smooth_um):
        
        # Import class parameters and tools
        self.par = par
        self.tools = tools.Tools()

        self.min_size_smoothed_um = end_min_size_smoothed_um
        self.min_size_smoothed = self.min_size_smoothed_um / self.par.scale

        # Smoothed the trajectories
        self.smo = smoothed_trajectories.SmoothedTrajectories(par=par, 
                                                              df=df, 
                                                              end_min_size_smoothed_um=end_min_size_smoothed_um, 
                                                              start_min_size_smoothed_um=start_min_size_smoothed_um, 
                                                              step_min_size_smooth_um=step_min_size_smooth_um)
        self.smo.smooth_small_displacement()
        self.smo.fill_smoothed_trajectories()
        self.df = self.smo.df

        # Class object
        self.rev = []
        self.rev_memory = []
        self.tbr = []

        for i in range(self.smo.n_traj):
            self.rev.append(np.zeros(len(self.smo.x[i])).astype(bool))
            self.rev_memory.append(np.zeros(len(self.smo.x[i])).astype(bool))
            self.tbr.append(np.ones(len(self.smo.x[i])) * np.nan)


    # def compute_trajectory_length(self):
    #     """
    #     Compute the length of each trajetories
        
    #     """
    #     print('COMPUTE THE LENGTH OF THE TRAJECTORIES')
    #     __, counts = np.unique(self.df.loc[:,self.par.track_id_column].to_numpy(), return_counts=True)
    #     counts = np.repeat(counts,repeats=counts,axis=0)
    #     self.df.loc[:,'traj_length'] = counts


    def compute_trajectory_length(self, col_traj_length: str, inplace: bool = False) -> None | pd.DataFrame:
        """
        Compute the length of trajectories in a tracking dataframe and add a specific column.

        Parameters:
        -----------
        col_traj_length : str
            The column name where the computed trajectory lengths will be stored.
        inplace : bool, optional
            If True, the operation modifies the input dataframe directly. 
            If False, a modified copy of the dataframe is returned (default is False).

        Returns:
        --------
        pandas.DataFrame or None
            - If `inplace` is False, returns a modified copy of the dataframe with the new column added.
            - If `inplace` is True, modifies the dataframe directly and returns None.
        """
        # Compute trajectory lengths
        traj_lengths = self.df[self.par.track_id_column].value_counts()

        # Map each ID to its trajectory length
        if inplace:
            self.df[col_traj_length] = self.df[self.par.track_id_column].map(traj_lengths)
        else:
            df_copy = self.df.copy()
            df_copy[col_traj_length] = df_copy[self.par.track_id_column].map(traj_lengths)
            return df_copy
            

    def reversals_detection(self):
        """
        Detect the reversals as the acute angle on the smoothed trajectories
        
        """
        print("DETECTION ON THE REVERSALS AND COMPUTATION OF THE TBR")
        time_fru_memory = round(self.par.frustration_time_memory / self.par.tbf)

        for traj in tqdm(range(self.smo.n_traj)):
        
            if len(self.smo.x_s[traj]) > 4:
                ### REVERSALS ###
                # Boolean array for reversals on the smoothed trajectories
                cond_rev = self.smo.ang_f[traj] < self.par.angle_rev
                # Condition for the last detected reversal
                indices_rev = np.where(cond_rev)[0]

                # Remove first and last reversals to close from the beginning or the end
                # of the trajectories that are not filter by the smoothing
                if len(indices_rev) > 0:
                    ind_first_rev = indices_rev[0]
                    length_first_rev_first_point = np.sum(self.smo.leng_f[traj][:ind_first_rev])

                    if length_first_rev_first_point < self.min_size_smoothed:
                        cond_rev[ind_first_rev] = False

                    index_last_rev = indices_rev[-1]
                    lenght_last_rev_last_point = np.sum(self.smo.leng_f[traj][index_last_rev-1:])
                    
                    if lenght_last_rev_last_point < self.min_size_smoothed:
                        cond_rev[index_last_rev] = False

                self.rev[traj] = cond_rev

                # Condition to know the time around a reversal
                cond_rev_memory = cond_rev.copy()
                id_rev_traj = np.where(cond_rev)[0]
                for idx in id_rev_traj:
                    if idx > time_fru_memory:
                        start = int(idx - time_fru_memory)
                        cond_rev_memory[start:idx] = True
                    else:
                        cond_rev_memory[0:idx] = True
                    if len(cond_rev_memory) - idx > time_fru_memory:
                        end = int(idx + time_fru_memory)
                        cond_rev_memory[idx+1:end+1] = True
                    else:
                        cond_rev_memory[idx+1:] = True
                self.rev_memory[traj] = cond_rev_memory

                # Boolean array for the reversals on the detected trajectories
                rev_time_tmp = self.smo.t[traj][cond_rev]

                ### TBR ###
                # Compute the tbr from the smoothed trajectories
                if len(np.where(cond_rev)[0]) > 1:

                    tbr_tmp = rev_time_tmp[1:] - rev_time_tmp[:-1]
                    tbr_tmp = np.concatenate((np.array([np.nan]),tbr_tmp))

                    # Fill tbr list
                    self.tbr[traj][cond_rev] = tbr_tmp

        # Add the reversals column into the inital dataframe df
        self.df.loc[:,self.par.rev_column] = np.concatenate(self.rev).astype(int)
        self.df.loc[:,self.par.rev_memory_column] = np.concatenate(self.rev_memory).astype(int)
        self.df.loc[:,self.par.tbr_column] = np.concatenate(self.tbr)
        self.compute_trajectory_length(col_traj_length=self.par.traj_length_column, inplace=True)