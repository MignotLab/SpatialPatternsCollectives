import numpy as np
import pandas as pd
from tqdm import tqdm
import os

import parameters
import tools
import smoothed_trajectories

class ReversalsDetection:


    def __init__(self, min_size_smoothed_um):
        
        # Import class parameters and tools
        self.par = parameters.Parameters()
        self.tools = tools.Tools()

        # Smoothed the trajectories
        self.smo = smoothed_trajectories.SmoothedTrajectories(min_size_smoothed_um)
        self.smo.smooth_small_displacement()
        self.smo.fill_smoothed_trajectories()

        # Class object
        self.rev = []
        self.rev_memory = []
        self.tbr = []

        for i in range(self.smo.n_traj):
            self.rev.append(np.zeros(len(self.smo.x[i])).astype(bool))
            self.rev_memory.append(np.zeros(len(self.smo.x[i])).astype(bool))
            self.tbr.append(np.ones(len(self.smo.x[i])) * np.nan)
            

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

                    if length_first_rev_first_point < self.par.min_size_smoothed:
                        cond_rev[ind_first_rev] = False

                    index_last_rev = indices_rev[-1]
                    lenght_last_rev_last_point = np.sum(self.smo.leng_f[traj][index_last_rev-1:])
                    
                    if lenght_last_rev_last_point < self.par.min_size_smoothed:
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
        self.smo.df.loc[:,'reversals'] = np.concatenate(self.rev).astype(int)
        self.smo.df.loc[:,'reversals_memory'] = np.concatenate(self.rev_memory).astype(int)
        self.smo.df.loc[:,'tbr'] = np.concatenate(self.tbr)

        # # Write dataframe into the csv file
        # # Check whether the specified path exists or not
        # isExist = os.path.exists(self.par.path_reversals)
        # if not isExist:
        #     # Create a new directory because it does not exist
        #     os.makedirs(self.par.path_reversals)
        #     print("The new directory is created!")
        
        # self.smo.df.to_csv(self.par.path_reversals+'data_rev'+self.par.end_name_file+'.csv',index=False)
