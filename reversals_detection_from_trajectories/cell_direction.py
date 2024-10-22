import numpy as np
import pandas as pd
import os

import parameters
import tools


class CellDirection:


    def __init__(self, end_name_file):
        
        # Import class
        self.par = parameters.Parameters()
        self.tool = tools.Tools()
        self.end_name_file = end_name_file

        # Read csv tracking file
        self.df = pd.read_csv(self.par.path_reversals+'data_rev'+self.end_name_file+'.csv')
        self.df.sort_values(by=[self.par.track_id_column,self.par.t_column],ignore_index=True,inplace=True)

        # Generate tuple for angle name in the dataframe
        self.angle_column = []
        for i in range(self.par.n_nodes):
            
            self.angle_column += ["ang"+str(i)]


    def nodes_directions_unique_node(self, angles, angle_unit='degree'):
        """
        Compute the direction of the cells when only one node is provide
        
        """
        coord = self.df.loc[:,[self.par.x_column, self.par.y_column_middle]].values
        coord_s = self.df.loc[:,(self.par.x_column+'s', self.par.y_column_middle+'s')].values

        # Compute the vector between the center node between t and t+1
        vect_track = np.column_stack((coord_s[:,0][1:] - coord_s[:,0][:-1], coord_s[:,1][1:] - coord_s[:,1][:-1]))
        # Add a row at the end to match with the previous vectors
        vect_track = np.concatenate((vect_track, np.array([vect_track[-1]])), axis=0)

        if angle_unit == 'degree':
            angs = angles * np.pi / 180
        else:
            angs = angles.copy()

        vect_forward = np.column_stack((np.cos(angs),np.sin(angs)))
        vect_backward = np.column_stack((np.cos(angs+np.pi),np.sin(angs+np.pi)))
        coord_pole_forward = coord + vect_forward
        coord_pole_backward = coord + vect_backward

        scalar_product_forward = np.sum(vect_forward * vect_track, axis=1)
        scalar_product_backward = np.sum(vect_backward * vect_track, axis=1)
        cond_main_pole_forward = scalar_product_forward > scalar_product_backward

        # Fill dataframe with new columns named main pole
        self.df.loc[:,["x_pole","y_pole"]] = np.nan
        self.df.loc[cond_main_pole_forward,["x_pole","y_pole"]] = coord_pole_forward[cond_main_pole_forward,:]
        self.df.loc[~cond_main_pole_forward,["x_pole","y_pole"]] = coord_pole_backward[~cond_main_pole_forward,:]

        # Add nan value at the end of each trajectories
        array_track_id = self.df.loc[:,self.par.track_id_column].values
        cond_ends = array_track_id[1:] - array_track_id[:-1] != 0
        cond_ends = np.concatenate((cond_ends,np.array([True])))
        self.df.loc[cond_ends,["x_pole","y_pole"]] = np.nan

        # Write dataframe into the csv file
        # Check whether the specified path exists or not
        isExist = os.path.exists(self.par.path_directions)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(self.par.path_directions)
            print("The new directory is created!")
            
        self.df.to_csv(self.par.path_directions+'data_dir'+self.end_name_file+'.csv',index=False)


    def nodes_directions(self):
        """
        Compute the direction of each node by detecting the direction of
        the cell from the smoothed trajectories
        
        """
        # Extract coordinates
        x_columns_name, y_columns_name = self.tool.gen_coord_str(n=self.par.n_nodes, xy=False)
        x_columns = self.df.loc[:, x_columns_name].values
        y_columns = self.df.loc[:, y_columns_name].values
        x_columns_s = self.df.loc[:, self.par.x_column+'s'].values
        y_columns_s = self.df.loc[:, self.par.y_column+'s'].values

        # Compute the vector between the center and the two adjacent nodes
        vect_bact_half_0 = np.array([x_columns[:, int(self.par.n_nodes/2-1)] - x_columns[:, int(self.par.n_nodes/2)], y_columns[:, int(self.par.n_nodes/2-1)] - y_columns[:, int(self.par.n_nodes/2)]])
        vect_bact_half_n = np.array([x_columns[:, int(self.par.n_nodes/2+1)] - x_columns[:, int(self.par.n_nodes/2)], y_columns[:, int(self.par.n_nodes/2+1)] - y_columns[:, int(self.par.n_nodes/2)]])
        
        # Compute the vector between the center node between t and t+1
        vect_track = np.array([x_columns_s[1:] - x_columns_s[:-1], y_columns_s[1:] - y_columns_s[:-1]])
        # Add a row at the end to match with the previous vectors
        vect_track = np.concatenate((vect_track.T, np.array([vect_track[:, -1]])), axis=0).T

        # Scalar product between the two vectors
        norm_half_0 = np.linalg.norm(vect_bact_half_0, axis=0)
        norm_half_n = np.linalg.norm(vect_bact_half_n, axis=0)
        norm_vect_track = np.linalg.norm(vect_track, axis=0)
        # Condition for superposition between two nodes half and half+-1
        cond_norm_0 = (norm_half_0 == 0) | (norm_half_n == 0) | (norm_vect_track == 0)
        # Avoid division by 0
        norm_half_0[norm_half_0==0] = 1
        norm_half_n[norm_half_n==0] = 1
        scalar_product_half_0 = np.sum(vect_bact_half_0 * vect_track / norm_half_0, axis=0)
        scalar_product_half_n = np.sum(vect_bact_half_n * vect_track / norm_half_n, axis=0)

        # Condition for pole n
        cond_main_pole_n = scalar_product_half_n > scalar_product_half_0

        # Fill dataframe with new columns named main pole
        self.df.loc[:, "main_pole"] = 0
        self.df.loc[:, "x_main_pole"] = self.df.loc[:, "x0"].copy()
        self.df.loc[:, "x_second_pole"] = self.df.loc[:, "x1"].copy()
        self.df.loc[:, "y_main_pole"] = self.df.loc[:, "y0"].copy()
        self.df.loc[:, "y_second_pole"] = self.df.loc[:, "y1"].copy()

        self.df.loc[cond_main_pole_n, "main_pole"] = self.par.n_nodes - 1
        self.df.loc[cond_main_pole_n, "x_main_pole"] = self.df.loc[cond_main_pole_n, "x"+str(self.par.n_nodes - 1)].copy()
        self.df.loc[cond_main_pole_n, "x_second_pole"] = self.df.loc[cond_main_pole_n, "x"+str(self.par.n_nodes - 2)].copy()
        self.df.loc[cond_main_pole_n, "y_main_pole"] = self.df.loc[cond_main_pole_n, "y"+str(self.par.n_nodes - 1)].copy()
        self.df.loc[cond_main_pole_n, "y_second_pole"] = self.df.loc[cond_main_pole_n, "y"+str(self.par.n_nodes - 2)].copy()

        self.df.loc[cond_norm_0, "main_pole"] = np.nan
        self.df.loc[cond_norm_0, "x_main_pole"] = np.nan
        self.df.loc[cond_norm_0, "x_second_pole"] = np.nan
        self.df.loc[cond_norm_0, "y_main_pole"] = np.nan
        self.df.loc[cond_norm_0, "y_second_pole"] = np.nan

        # Add nan value at the end of each trajectories
        array_track_id = self.df.loc[:, self.par.track_id_column].values
        cond_ends = array_track_id[1:] - array_track_id[:-1] != 0
        cond_ends = np.concatenate((cond_ends, np.array([True])))
        self.df.loc[cond_ends, "main_pole"] = np.nan

        # Add news columns for the angles in x and y
        dir_x_0 = x_columns[:, :-1] - x_columns[:, 1:]
        dir_x_n = -dir_x_0[:, :].copy()
        dir_y_0 = y_columns[:, :-1] - y_columns[:, 1:]
        dir_y_n = -dir_y_0[:, :].copy()
        dir_x_0 = np.concatenate((np.array([dir_x_0[:,0]]), dir_x_0.T), axis=0).T
        dir_y_0 = np.concatenate((np.array([dir_y_0[:,0]]), dir_y_0.T), axis=0).T
        dir_x_n = np.concatenate((dir_x_n.T, np.array([dir_x_n[:,-1]])), axis=0).T
        dir_y_n = np.concatenate((dir_y_n.T, np.array([dir_y_n[:,-1]])), axis=0).T

        # Compute angle of each segment seprate by each nodes
        angles_array = np.ones(dir_x_0.shape) * np.nan
        cond_main_pole_0 = self.df.loc[:, 'main_pole'].values == 0
        cond_main_pole_n = self.df.loc[:, 'main_pole'].values == self.par.n_nodes - 1
        angles_array[cond_main_pole_0, :] = np.arctan2(dir_y_0, dir_x_0)[cond_main_pole_0, :]
        angles_array[cond_main_pole_n, :] = np.arctan2(dir_y_n, dir_x_n)[cond_main_pole_n, :]

        # Fill the angles in the dataframe
        self.df.loc[:, self.angle_column] = np.round(angles_array, 3)
        self.df.sort_values(by=[self.par.track_id_column, self.par.t_column], ignore_index=True, inplace=True)
        
        # # Write dataframe into the csv file
        # isExist = os.path.exists(self.par.path_directions)
        # if not isExist:
        #     # Create a new directory because it does not exist
        #     os.makedirs(self.par.path_directions)
        #     print("The new directory is created!")
        
        # self.df.to_csv(self.par.path_directions+'data_dir'+self.par.end_name_file+'.csv',index=False)