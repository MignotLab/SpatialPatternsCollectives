import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial import KDTree as kdtree
from joblib import Parallel, delayed

import parameters
import tools
import velocities_computation


class Corrections:


    def __init__(self):
        
        self.par = parameters.Parameters()
        self.tool = tools.Tools()

        # Read csv tracking file
        self.df = pd.read_csv(self.par.path_tracking)

        if self.par.track_mate_csv:
            self.df = self.df.iloc[3:,2:].astype(float)

        self.df.sort_values(by=[self.par.track_id_column,self.par.t_column],ignore_index=True,inplace=True)
        self.id_frames = np.unique(self.df.loc[:,self.par.t_column].values)
        # Increase computation
        # self.df = self.df.loc[:10000,:]

        self.vel = velocities_computation.Velocity(track_id=self.df.loc[:,self.par.track_id_column].values)
        # Compute the velocities
        t = self.df.loc[:,self.par.t_column].values
        x_all = self.df.loc[:,self.par.x_column].values
        y_all = self.df.loc[:,self.par.y_column].values
        self.vel.compute_velocity(x=x_all,y=y_all,t=t)
        # Add the reversals column into the inital dataframe df
        self.df.loc[:,'velocity'] = self.vel.velocity

        self.index_traj, self.count = np.unique(self.df.loc[:,self.par.track_id_column], return_counts=True)
        self.columns_x, self.columns_y = self.tool.gen_coord_str(n=self.par.n_nodes, xy=False)  


    def break_long_displacement(self):
        """
        When too long displacement are detected break the trajectory to
        create a new one
        
        """
        # Initialize the beginning and the end of the actual trajectory
        start_traj_id = 0
        end_traj_id = np.where(np.isnan(self.vel.velocity[start_traj_id:]))[0][0]

        print("BREAK TRAJECTORIES WHEN BAD DETECTIONS")
        for i in tqdm(range(len(self.vel.velocity)-1)):

            if np.isnan(self.vel.velocity[i]):
                start_traj_id = i + 1
                end_traj_id = start_traj_id + np.where(np.isnan(self.vel.velocity[start_traj_id:]))[0][0]

            if self.vel.velocity[i] > self.par.vmax / self.par.scale * self.par.tbf:

                self.df.loc[start_traj_id:i,self.par.track_id_column] = np.ones(int(i-start_traj_id+1)) + np.max(self.df.loc[:,self.par.track_id_column])
                self.vel.velocity[i] = np.nan                    
                start_traj_id = i + 1


    def reorder_pole(self):
        """
        Reorder the pole of the bacteria in the dataframe in the case 
        where several point are detected along the bacteria shape for 
        each cell
        
        """
        if self.par.n_nodes > 1:
            # I only keep the poles coordinates
            print('REORDER POLES')
            for id_traj in tqdm(self.index_traj):

                cond_traj = self.df.loc[:, self.par.track_id_column] == id_traj
                len_traj = len(self.df.loc[cond_traj])
                # Extract the coordinates as numpy array
                x_array = self.df.loc[cond_traj, self.columns_x].values
                y_array = self.df.loc[cond_traj, self.columns_y].values

                for i in range(len_traj-1):

                    # Compute the difference between consecutive coordinates
                    dist_normal = np.sqrt((x_array[i, 0] - x_array[i+1, 0])**2 + (y_array[i, 0] - y_array[i+1, 0])**2)
                    dist_inverse = np.sqrt((x_array[i, 0] - x_array[i+1, -1])**2 + (y_array[i, 0] - y_array[i+1, -1])**2)
                
                    if dist_inverse < dist_normal:

                        x_array[i+1:, :] = np.flip(x_array[i+1:, :], axis=1)
                        y_array[i+1:, :] = np.flip(y_array[i+1:, :], axis=1)

                self.df.loc[cond_traj, self.columns_x] = x_array.copy()
                self.df.loc[cond_traj, self.columns_y] = y_array.copy()


    def compute_trajectory_length(self):
        """
        Compute the length of each trajetories
        
        """
        print('START COMPUTE THE LENGTH OF THE TRAJECTORIES')
        __, counts = np.unique(self.df.loc[:,self.par.track_id_column].values, return_counts=True)
        counts = np.repeat(counts,repeats=counts,axis=0)
        self.df.loc[:,'traj_length'] = counts
        print('END COMPUTE THE LENGTH OF THE TRAJECTORIES')


    # def fill_time_holes(self):
    #     """
    #     Fill the hole inside trajectories where missed a time point
        
    #     """
    #     track_id = self.df.loc[:,self.par.track_id_column].values
    #     cond_change_traj = track_id[1:] != track_id[:-1]
    #     # cond_change_traj = np.concatenate((cond_change_traj,np.array([True])))
    #     times = self.df.loc[:,self.par.t_column].values
    #     cond_diff_time = times[1:] != times[:-1] + 1
    #     diff_time = times[1:] - times[:-1]
        
    #     extract_1 = self.df.loc[:len(self.df)-2].loc[~cond_change_traj&cond_diff_time]
    #     extract_2 = self.df.loc[1:].loc[~cond_change_traj&cond_diff_time]
    #     df_tmp = pd.DataFrame((extract_1.values+extract_2.values)/2,columns=self.df.columns)
    #     # df_tmp.loc[:,['theta','smoothTheta','phi']] = extract_1.loc[:,['theta','smoothTheta','phi']].values
    #     indices = np.where(~cond_change_traj&cond_diff_time)[0] + 1
    #     array_final = np.insert(arr=self.df.values, obj=indices, values=df_tmp.values, axis=0)
    #     df_final = pd.DataFrame(array_final,columns=self.df.columns)

