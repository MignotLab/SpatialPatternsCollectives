# %%
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

import parameters
import velocities_computation
import tools

class SmoothedTrajectoriesTypeError(Exception):
    pass

class SmoothedTrajectories:


    def __init__(self, min_size_smoothed_um):
        
        # Import class parameters and tools
        self.par = parameters.Parameters()
        self.tool = tools.Tools()

        # Smooth parameter in pixel
        self.min_size_smoothed = min_size_smoothed_um / self.par.scale

        # Read csv tracking file
        self.df = pd.read_csv(self.par.path_tracking)

        if self.par.track_mate_csv:
            self.df = self.df.iloc[3:, 2:].astype(float)
            self.df = self.df.loc[:, [self.par.track_id_column, self.par.t_column, self.par.x_column, self.par.y_column]]
            self.fill_trajectory_holes()

        self.df.sort_values(by=[self.par.track_id_column,self.par.t_column],ignore_index=True,inplace=True)
        self.id_frames = np.unique(self.df.loc[:, self.par.t_column].values)
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
        
        # Class object
        self.index_traj, self.count = np.unique(self.df.loc[:,self.par.track_id_column], return_counts=True)
        self.new_index_traj = np.repeat(np.arange(1,len(self.index_traj)+1), self.count)
        self.n_traj = len(self.index_traj)
        # Track
        self.x = []
        self.y = []
        self.t = []
        # Smooth
        self.x_s = []
        self.y_s = []
        self.t_s = []
        self.ang_s = []
        self.leng_s = []
        # Smooth fill
        self.x_f = []
        self.y_f = []
        self.t_f = []
        self.ang_f = []
        self.leng_f = []


    def fill_trajectory_holes(self):
        """
        In case the tracking sometimes do not detect a bacteria at a specific frame, interpolate the new position
        
        """
        print('FILL THE HOLES')
        self.df_detection_cor = self.df.copy()
        id_array = self.df_detection.loc[:, self.par.track_id_column].values
        frame_array = self.df_detection.loc[:, self.par.t_column].values
        # Build a boolean array to detect the change of trajectories
        cond_change_traj = id_array[1:] != id_array[:-1]
        # Array to know the step time between two consecutive event
        time_diff = frame_array[1:] - frame_array[:-1]
        # During change of traj it could be have time_diff != 1 but we don't want to take it into account
        time_diff[cond_change_traj] = 1
        # Add the last point of the array manually
        time_diff = np.concatenate((time_diff, np.array([1])))
        # Condition to know where correction have to be done
        cond_time_diff_1 = time_diff > 1
        # Condition to know which is the following point after the point which need a correction
        cond_time_diff_2 = np.roll(cond_time_diff_1, shift=1)
        columns_coord = [self.par.x_column, self.par.y_column]
        # Iteration over the time difference between consecutive bcterium
        max_diff = int(np.max(time_diff))

        self.columns_name = [self.par.track_id_column, self.par.t_column, self.par.x_column, self.par.y_column]
        for i in tqdm(range(2, max_diff+1)):
            cond_time_diff_i = time_diff == i

            # Fill the dataframe with the missing timestep
            for j in range(1, i):
                ### Creates new rows for the missing coordinates which are not present
                # The id is the same than at in time 1
                array_id = id_array[cond_time_diff_i]
                # # We hypothesize that the len of the skel is also the same
                # array_len_skel = skel_array[cond_time_diff_i]
                # The value of the frame is increase by j
                array_frame_i = frame_array[cond_time_diff_i] + j
                # Build the new coordinates
                coords_time_1 = self.df_detection.loc[cond_time_diff_i, columns_coord].values
                coords_time_2 = self.df_detection.loc[np.roll(cond_time_diff_i, shift=1), columns_coord].values
                # Create a vector in the direction of the two points
                vector_coord = coords_time_2 - coords_time_1
                array_coord = coords_time_1 + j * vector_coord / i
                # Build final array
                array_final = np.concatenate((np.array([array_id]), np.array([array_frame_i]), array_coord.T), axis=0).T
                self.df_detection_cor = pd.concat([self.df_detection_cor, pd.DataFrame(array_final, columns=self.columns_name)], axis=0, ignore_index=True)

        self.df = self.df_detection_cor.dropna()
        self.df.sort_values(by=['id', 'frame'], ignore_index=True, inplace=True)


    def coord_in_list(self):
        """
        Read the csv file and extract trajectories into lists
        
        """
        print("DATAFRAME X, Y and T TO LIST")
        for id_traj in tqdm(self.index_traj):

            cond_traj = self.df.loc[:, self.par.track_id_column] == id_traj
            self.t.append(self.df.loc[cond_traj, self.par.t_column].values)
            self.x.append(self.df.loc[cond_traj, self.par.x_column].values)
            self.y.append(self.df.loc[cond_traj, self.par.y_column].values)


    def smooth_small_displacement(self):
        """
        Smooth the trajectory depending on the parameter smooth_size from x and y coordinates
        
        """
        # Transform first the dataframe t, x and y into lists
        self.coord_in_list()
        self.x_s = self.x.copy()
        self.y_s = self.y.copy()
        self.t_s = self.t.copy()

        # Smooth over to small displacements
        print("SMOOTH TRAJECTORIES")
        for j in tqdm(range(len(self.x_s))):

            x_s_tmp = self.x_s[j].copy()
            y_s_tmp = self.y_s[j].copy()
            try:
                t_s_tmp = self.t_s[j].copy()
            except:
                print(y_s_tmp)
                print(t_s_tmp)
                raise SmoothedTrajectoriesTypeError()
            
            for iteration in range(self.par.iteration):

                i = 1
                xi = [x_s_tmp[0]]
                yi = [y_s_tmp[0]]
                ti = [t_s_tmp[0]]

                leng_tmp = np.sqrt((x_s_tmp[1:] - x_s_tmp[:-1])**2 + (y_s_tmp[1:] - y_s_tmp[:-1])**2)
                v1 = np.array([x_s_tmp[1:-1] - x_s_tmp[ :-2], y_s_tmp[1:-1] - y_s_tmp[ :-2]])
                v2 = np.array([x_s_tmp[2:  ] - x_s_tmp[1:-1], y_s_tmp[2:  ] - y_s_tmp[1:-1]])
                ang_tmp = np.abs(self.tool.py_ang(-v1, v2))

                while i < len(x_s_tmp) - 2:

                    # All conditions
                    low_ang = (ang_tmp[i-1] <= np.pi/2) and (ang_tmp[i] <= np.pi/2)
                    high_ang = (ang_tmp[i-1] > np.pi/2) and (ang_tmp[i] > np.pi/2)
                    high_low_ang = (ang_tmp[i-1] > np.pi/2) and (ang_tmp[i] <= np.pi/2)
                    low_high_ang = (ang_tmp[i-1] <= np.pi/2) and (ang_tmp[i] > np.pi/2)

                    if leng_tmp[i] < self.min_size_smoothed:
                        
                        # Compute new ccordinates
                        if low_ang or high_ang:

                            xy = np.array([(x_s_tmp[i+1]+x_s_tmp[i])/2, (y_s_tmp[i+1]+y_s_tmp[i])/2])
                            ti.append((t_s_tmp[i] + t_s_tmp[i+1]) / 2)

                        elif high_low_ang:

                            xy = np.array([x_s_tmp[i+1], y_s_tmp[i+1]])
                            ti.append(t_s_tmp[i+1])

                        elif low_high_ang:

                            xy = np.array([x_s_tmp[i], y_s_tmp[i]])
                            ti.append(t_s_tmp[i])

                        # Store new coordinate
                        xi.append(xy[0])
                        yi.append(xy[1])

                        # Increase i
                        i += 2
                    
                    else:
                        
                        # Store new coordinate
                        xy = np.array([x_s_tmp[i], y_s_tmp[i]])
                        xi.append(xy[0])
                        yi.append(xy[1])
                        ti.append(t_s_tmp[i])

                        # Increase i
                        i += 1

                # Store the last coordinates
                if i == len(x_s_tmp)-2:

                    xy1 = np.array([x_s_tmp[i], y_s_tmp[i]])
                    xy2 = np.array([x_s_tmp[i+1], y_s_tmp[i+1]])
                    xi.extend([xy1[0], xy2[0]])
                    yi.extend([xy1[1], xy2[1]])
                    ti.extend([t_s_tmp[i], t_s_tmp[i+1]])

                elif i == len(x_s_tmp)-1:

                    xy = np.array([x_s_tmp[i], y_s_tmp[i]])
                    xi.append(xy[0])
                    yi.append(xy[1])
                    ti.append(t_s_tmp[i])

                x_s_tmp = np.array(xi)
                y_s_tmp = np.array(yi)
                t_s_tmp = np.array(ti)

            leng_tmp = np.sqrt((x_s_tmp[1:] - x_s_tmp[:-1])**2 + (y_s_tmp[1:] - y_s_tmp[:-1])**2)
            v1 = np.array([x_s_tmp[1:-1] - x_s_tmp[ :-2], y_s_tmp[1:-1] - y_s_tmp[ :-2]])
            v2 = np.array([x_s_tmp[2:  ] - x_s_tmp[1:-1], y_s_tmp[2:  ] - y_s_tmp[1:-1]])
            # Compute an angle in [0,pi]
            ang_tmp = np.abs(self.tool.py_ang(-v1, v2))
            # Apply a correction for null vector which generate an angle equal to 0 whereas its not a reversal
            cond_null_vector = (np.linalg.norm(v1,axis=0) == 0) | (np.linalg.norm(v2,axis=0) == 0)
            ang_tmp[cond_null_vector] = np.pi

            self.x_s[j] = x_s_tmp
            self.y_s[j] = y_s_tmp
            self.t_s[j] = t_s_tmp
            self.leng_s.append(leng_tmp)
            self.ang_s.append(ang_tmp)


    def fill_smoothed_trajectories(self):
        """
        Fill the "holes" in the smoothed trajectories due to the suppression of 
        several points to apply the smooth. Then the number of points of the smoothed
        trajectories will be the same than the initial trajectories
        
        """
        # Reinitialize the array to empty list
        self.x_f = []
        self.y_f = []
        self.t_f = []
        self.ang_f = []
        self.leng_f = []
        
        print("FILL THE SMOOTHED TRAJECTORIES TO MATCH WITH THE TIME FRAMES")
        for traj in tqdm(range(self.n_traj)):

            x_f_tmp = np.zeros(len(self.x[traj]))
            y_f_tmp = np.zeros(len(self.y[traj]))
            ang_f_tmp = np.ones(len(self.x[traj])) * np.pi

            t_f_tmp = np.round(self.t_s[traj]).astype(int)
            size_time_hole = t_f_tmp[1:] - t_f_tmp[:-1]
            id_ang = t_f_tmp[1:-1] - t_f_tmp[0]
            ang_f_tmp[id_ang] = self.ang_s[traj].copy()

            count = -1
            for i in range(len(size_time_hole)):
                
                count += 1
                x_f_tmp[count] = self.x_s[traj][i]
                y_f_tmp[count] = self.y_s[traj][i]
                step_x = (self.x_s[traj][i+1] - self.x_s[traj][i]) / size_time_hole[i]
                step_y = (self.y_s[traj][i+1] - self.y_s[traj][i]) / size_time_hole[i]

                for j in range(1, size_time_hole[i]):
                    
                    count += 1
                    x_f_tmp[count] = x_f_tmp[count-1] + step_x
                    y_f_tmp[count] = y_f_tmp[count-1] + step_y

            # Last coordinate of the trajectory
            x_f_tmp[count+1] = self.x_s[traj][-1]
            y_f_tmp[count+1] = self.y_s[traj][-1]
            # Lenght of the smoothed trajectories
            leng_f_tmp = np.sqrt((x_f_tmp[1:] - x_f_tmp[:-1])**2 + (y_f_tmp[1:] - y_f_tmp[:-1])**2)

            self.x_f.append(x_f_tmp)
            self.y_f.append(y_f_tmp)
            self.ang_f.append(ang_f_tmp)
            self.leng_f.append(leng_f_tmp)

        self.df.loc[:,self.par.x_column+'s'] = np.concatenate(self.x_f)
        self.df.loc[:,self.par.y_column+'s'] = np.concatenate(self.y_f)