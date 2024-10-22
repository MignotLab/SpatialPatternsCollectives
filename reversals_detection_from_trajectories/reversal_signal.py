import numpy as np
import pandas as pd
from scipy.spatial import KDTree as kdtree
from tqdm import tqdm
import os
from joblib import Parallel, delayed

import parameters
import tools
import velocities_computation


class ReversalSignal:


    def __init__(self, end_name_file):
        
        # Import class
        self.par = parameters.Parameters()
        self.tool = tools.Tools()
        self.end_name_file = end_name_file

        # Copy the dataframe in cell_direction
        # For testing
        # self.df = pd.read_csv(self.par.path_directions+'data_dir'+self.end_name_file+'_3_frames.csv')
        self.df = pd.read_csv(self.par.path_directions+'data_dir'+self.end_name_file+'.csv')
        self.df.sort_values(by=[self.par.track_id_column,self.par.t_column],ignore_index=True,inplace=True)
        self.n_traj = np.max(self.df.loc[:,self.par.track_id_column].values)
        self.frame_indices = np.unique(self.df.loc[:,self.par.t_column].values)

        #Import class
        self.vel = velocities_computation.Velocity(track_id=self.df.loc[:,self.par.track_id_column].values)

        # Nodes coordinates
        x_columns_name, y_columns_name = self.tool.gen_coord_str(n=self.par.n_nodes, xy=False)
        self.t = self.df.loc[:, self.par.t_column].values
        self.coords_x = self.df.loc[:, x_columns_name].values
        self.coords_y = self.df.loc[:, y_columns_name].values
        self.coords_xs = self.df.loc[:, self.par.x_column+'s'].values
        self.coords_ys = self.df.loc[:, self.par.y_column+'s'].values
        self.main_pole = self.df.loc[:, 'main_pole'].values
        self.reversals = self.df.loc[:, 'reversals'].values

        # Class objects
        self.local_frustration = np.ones(len(self.df)) * np.nan
        self.local_frustration_s = np.ones(len(self.df)) * np.nan
        self.time_cumul = round(self.par.frustration_time_memory / self.par.tbf)
        self.frustration_memory = np.ones((self.time_cumul)) * np.nan
        self.frustration_memory_s = np.ones((self.time_cumul)) * np.nan
        self.cumul_frustration = np.ones(len(self.df)) * np.nan
        self.cumul_frustration_s = np.ones(len(self.df)) * np.nan

        # Correction do to the rate to keep a signal between 0 and 1
        tmp = np.zeros(self.time_cumul)
        # Construct correction array
        for i in range(len(tmp)):
            
            tmp = np.roll(tmp, shift=1)
            tmp *= np.exp(- self.par.cumul_frustration_decreasing_rate * self.par.tbf)
            tmp[0] = 1

        self.correction = np.sum(tmp)
        self.exp_factor = np.exp(-self.par.cumul_frustration_decreasing_rate*np.arange(self.time_cumul))


    def compute_local_frustration(self, method='initial',save=False):
        """
        Compute the frustration of the cells before a reversal
        
        """
        # Target velocity computation
        self.vel.compute_vt(x0=self.coords_x[:,0],
                            y0=self.coords_y[:,0],
                            x1=self.coords_x[:,1],
                            y1=self.coords_y[:,1],
                            xm=self.coords_x[:,-2],
                            ym=self.coords_y[:,-2],
                            xn=self.coords_x[:,-1],
                            yn=self.coords_y[:,-1],
                            main_pole=self.main_pole,
                            reversals=self.reversals)

        # # Real velocity computation
        # self.vel.compute_vr(x0=self.coords_x[:, 0],
        #                     y0=self.coords_y[:, 0],
        #                     xn=self.coords_x[:, -1],
        #                     yn=self.coords_y[:, -1],
        #                     t=self.t,
        #                     main_pole=self.main_pole,
        #                     reversals=self.reversals)
        
        # Real velocity computation
        self.vel.compute_vr(x=self.df.loc[:, self.par.x_column].values,
                            y=self.df.loc[:, self.par.y_column].values,
                            t=self.t,
                            reversals=self.reversals)

        # Real smoothed velocity computation
        self.vel.compute_vr_s(xs=self.coords_xs,
                              ys=self.coords_ys,
                              reversals=self.reversals)

        # Compute the scalar products and put it at -1 in the case there is a nan
        # v_mean = 3.7 #np.nanmean(np.linalg.norm(self.vel.vr,axis=0))
        v_mean = np.nanmean(np.linalg.norm(self.vel.vr,axis=0))
        sp_mix = np.sum(self.vel.vr * (v_mean * self.vel.vt), axis=0)
        sp_mix_s = np.sum(self.vel.vr_s * (v_mean * self.vel.vt), axis=0)

        if method == 'initial':
            self.local_frustration = 1 - sp_mix / v_mean**2
            self.local_frustration_s = 1 - sp_mix_s / v_mean**2

        if method == 'michele':
            sp_vr = np.sum(self.vel.vr * self.vel.vr, axis=0)
            sp_vr_s = np.sum(self.vel.vr_s * self.vel.vr_s, axis=0)
            sp_vt = np.sum((v_mean * self.vel.vt) * (v_mean * self.vel.vt), axis=0)
            # Avoid division by 0
            sp_vr[sp_vr==0] = 1
            sp_vr_s[sp_vr_s==0] = 1
            sp_vt[sp_vt==0] = 1
            self.local_frustration = 1 - sp_mix / np.maximum(sp_vr,sp_vt)
            self.local_frustration_s = 1 - sp_mix_s / np.maximum(sp_vr_s,sp_vt)

        self.df.loc[:,'local_frustration'] = np.round(self.local_frustration,2)
        self.df.loc[:,'local_frustration_s'] = np.round(self.local_frustration_s,2)

        if save:
            # Write dataframe into the csv file
            # Check whether the specified path exists or not
            isExist = os.path.exists(self.par.path_reversal_signal)
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(self.par.path_reversal_signal)
                print("The new directory is created!")
                
            self.df.to_csv(self.par.path_reversal_signal+'data_rev_sig'+self.end_name_file+'.csv',index=False)


    def compute_cumul_frustration(self, method='initial', save_local=False):
        """
        Compute the frustration of the cells before a reversal with the michelle measure
        
        """
        print("COMPUTE FRUSTRATION")
        # Compute local frustration
        self.compute_local_frustration(method=method,save=save_local)
        count = 0
        track_indices = np.unique(self.df.loc[:,self.par.track_id_column].values)

        for i, track_index in enumerate(tqdm(track_indices)):
            cond_traj = self.df.loc[:,self.par.track_id_column] == track_index
            local_frustration_traj = self.local_frustration[cond_traj]
            local_frustration_traj_s = self.local_frustration_s[cond_traj]
            self.frustration_memory = local_frustration_traj[0] * self.exp_factor
            self.frustration_memory_s = local_frustration_traj_s[0] * self.exp_factor

            for j in range(len(local_frustration_traj)):
                # Update the cumul of the frustration
                self.cumul_frustration[count] = np.nansum(self.frustration_memory)
                self.cumul_frustration_s[count] = np.nansum(self.frustration_memory_s)
                count += 1
                # Rool the frustration memory array to remove the older frustration
                self.frustration_memory = np.roll(self.frustration_memory, shift=1, axis=0)
                self.frustration_memory_s = np.roll(self.frustration_memory_s, shift=1, axis=0)
                # Decrease exponentially the old frustrations
                self.frustration_memory[:] *= np.exp(-self.par.cumul_frustration_decreasing_rate * self.par.tbf)
                self.frustration_memory_s[:] *= np.exp(-self.par.cumul_frustration_decreasing_rate * self.par.tbf)
                # Add the new frustration as the first element
                self.frustration_memory[0] = local_frustration_traj[j]
                self.frustration_memory_s[0] = local_frustration_traj_s[j]

            
        self.df.loc[:,'cumul_frustration'] = np.round(self.cumul_frustration,2) / self.correction
        self.df.loc[:,'cumul_frustration_s'] = np.round(self.cumul_frustration_s,2) / self.correction

        # Save the dataframe
        isExist = os.path.exists(self.par.path_reversal_signal)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(self.par.path_reversal_signal)
            print("The new directory is created!")

        # print("SAVE...")
        # self.df.to_csv(self.par.path_reversal_signal+'data_rev_sig'+self.par.end_name_file+'.csv',index=False)
        # print("FINISH")


    # def compute_polarity_main(self, frame):
    #     """
    #     Compute the polarity signal
        
    #     """
    #     cond_time = self.df.loc[:,self.par.t_column].values == frame
    #     x_array = self.df.loc[cond_time,self.tool.gen_coord_str(n=self.par.n_nodes,xy=False)[0]].values.T
    #     y_array = self.df.loc[cond_time,self.tool.gen_coord_str(n=self.par.n_nodes,xy=False)[1]].values.T
    #     dir_array = self.df.loc[cond_time,self.tool.gen_string_numbered(n=self.par.n_nodes,str_name="ang")].values.T

    #     coord = np.column_stack((x_array.flatten(), y_array.flatten()))
    #     tree = kdtree(coord)
    #     dist, ind = tree.query(coord, k=self.par.kn)
    #     self.dist_flat[:], self.ind_flat[:] = np.concatenate(self.dist), np.concatenate(self.ind)


    # def compute_polarity(self):
    #     """
    #     Compute the polarity signal
        
    #     """
    #     print("COMPUTE THE POLARITY")
    #     dfs = Parallel(n_jobs=self.par.n_jobs)(delayed(self.compute_polarity_main)(t) for t in tqdm(self.frame_indices))
    #     df_polarity = pd.concat(dfs)


    def compute_number_of_neighbors(self):
        """
        Make a list of the id od the neighbors of each cell and also count the number of different neighbours
        The condition to be a neighbors is that at least one node of a different cell should be at a distance less than
        the width of the bacterium
        
        """ 
        print('COMPUTE NUMBER OF NEIGHBOURS')
        column_x, column_y = self.tool.gen_coord_str(self.par.n_nodes, xy=False)
        neighbour_max_distance = self.par.width / self.par.scale * 1.2

        for frame in tqdm(self.frame_indices):
            cond_time = self.df.loc[:, 'frame'] == frame
            n_bact = self.df.loc[cond_time, column_x].shape[0]
            coord = np.column_stack((self.df.loc[cond_time, column_x].values.T.flatten(), self.df.loc[cond_time, column_y].values.T.flatten()))
            tree = kdtree(coord)
            dist, ind = tree.query(coord, k=self.par.kn)
            # dist_flat, ind_flat = np.concatenate(dist), np.concatenate(ind)

            id_node, id_bact = np.divmod(ind, n_bact)
            array_same_bact = np.tile((np.ones((self.par.kn, n_bact)) * np.arange(n_bact)).T.astype(int), (self.par.n_nodes,1))
            cond_same_bact = id_bact == array_same_bact
            # Compute the number of neighbours
            cond_dist =  dist < neighbour_max_distance

            count_n = id_bact.reshape((n_bact, int(self.par.n_nodes * self.par.kn)), order='F')
            cond_dist = cond_dist.reshape((n_bact, int(self.par.n_nodes * self.par.kn)), order='F')
            cond_same_bact = cond_same_bact.reshape((n_bact, int(self.par.n_nodes * self.par.kn)), order='F')

            cond_neighbours = ~cond_same_bact & cond_dist

            # # Compute the difference of bact index
            cond_neighbours = np.take_along_axis(cond_neighbours, count_n.argsort(axis=1), axis=1)
            count_n = np.sort(count_n, axis=1)

            count_n_smart = count_n.copy()
            count_n_smart[cond_neighbours] -= (n_bact+1)
            count_n_smart = np.sort(count_n_smart)
            count_n_smart = count_n_smart[:, 1:] - count_n_smart[:, :-1]
            count_n_smart[count_n_smart != 0] = 1

            count_n = count_n[:, 1:] - count_n[:, :-1]
            count_n[count_n != 0] = 1

            self.df.loc[cond_time, 'n_neighbours'] = np.sum(count_n_smart, axis=1) - np.sum(count_n, axis=1)

    def compute_number_of_neighbors_and_polarity_for_loop(self):
        """
        Make a list of the id od the neighbors of each cell and also count the number of different neighbours
        The condition to be a neighbors is that at least one node of a different cell should be at a distance less than
        the width of the bacterium
        
        """ 
        print('COMPUTE THE NUMBER OF NEIGHBOURS AND THE MEAN POLARITY')
        column_x, column_y = self.tool.gen_coord_str(self.par.n_nodes, xy=False)
        column_angles = self.tool.gen_string_numbered(n=self.par.n_nodes, str_name="ang")
        neighbour_max_distance = self.par.width / self.par.scale * 1.2

        # Compute the neighbours id
        array_number_of_neighbours = []
        list_id_neighbors = []
        mean_polarity_bact_neighbours = []
        for frame in tqdm(self.frame_indices):
            cond_time = self.df.loc[:, 'frame'] == frame
            n_bact = self.df.loc[cond_time, column_x].shape[0]
            coord = np.column_stack((self.df.loc[cond_time, column_x].values.T.flatten(), self.df.loc[cond_time, column_y].values.T.flatten()))
            angles = self.df.loc[cond_time, column_angles].values
            tree = kdtree(coord)
            dist, ind = tree.query(coord, k=self.par.kn)
            # dist_flat, ind_flat = np.concatenate(dist), np.concatenate(ind)
            id_node, id_bact = np.divmod(ind, n_bact)
            # array_same_bact = np.tile((np.ones((self.par.kn, n_bact)) * np.arange(n_bact)).T.astype(int), (self.par.n_nodes,1))
            # cond_different_bact = id_bact != array_same_bact
            # Compute the number of neighbours
            cond_dist =  dist < neighbour_max_distance
            # cond_global = cond_different_bact & cond_dist

            for index, (idx_neigbhours, cond_dist_index) in enumerate(zip(id_bact, cond_dist)):
                index_neighbours = np.unique(idx_neigbhours[cond_dist_index])
                array_number_of_neighbours.append(len(index_neighbours) - 1) # Remove itself which is take into account here
                mean_angle_bact_neigbours_i = np.mean(angles[index_neighbours, :], axis=1)
                if len(index_neighbours) - 1 > 0:
                    mean_polarity_bact_neighbours.append(np.mean(np.cos(mean_angle_bact_neigbours_i[0] - mean_angle_bact_neigbours_i[1:])))
                    list_id_neighbors.append(np.array_str(index_neighbours[1:]))
                else:
                    mean_polarity_bact_neighbours.append(1) # No neighbours polarity equal to 1
                    list_id_neighbors.append(np.nan)


            # count_n = id_bact.reshape((n_bact, int(self.par.n_nodes * self.par.kn)), order='F')
            # cond_dist = cond_dist.reshape((n_bact, int(self.par.n_nodes * self.par.kn)), order='F')
            # cond_same_bact = cond_same_bact.reshape((n_bact, int(self.par.n_nodes * self.par.kn)), order='F')

            # cond_neighbours = ~cond_same_bact & cond_dist

            # # # Compute the difference of bact index
            # cond_neighbours = np.take_along_axis(cond_neighbours, count_n.argsort(axis=1), axis=1)
            # count_n = np.sort(count_n, axis=1)

            # count_n_smart = count_n.copy()
            # count_n_smart[cond_neighbours] -= (n_bact+1)
            # count_n_smart = np.sort(count_n_smart)
            # count_n_smart = count_n_smart[:, 1:] - count_n_smart[:, :-1]
            # count_n_smart[count_n_smart != 0] = 1

            # count_n = count_n[:, 1:] - count_n[:, :-1]
            # count_n[count_n != 0] = 1

        self.df.loc[:, 'n_neighbours'] = array_number_of_neighbours
        self.df.loc[:, 'id_neighbours'] = list_id_neighbors
        self.df.loc[:, 'mean_polarity'] = mean_polarity_bact_neighbours


    def first_occurrence(self, row):
        """
        Get a boolean array indicating the first occurrences of elements in a given row.

        Parameters:
        - row (numpy.ndarray): One-dimensional array representing a row in the 2D array.

        Returns:
        - numpy.ndarray: Boolean array of the same length as the input row, 
        where True indicates the first occurrence of each unique element.
        """
        # Get unique elements and their indices of the first occurrences
        unique_elements, first_occurrence_indices = np.unique(row, return_index=True)
        
        # Create a boolean array with False, then set True at indices of the first occurrences
        result = np.full_like(row, fill_value=False, dtype=bool)
        result[first_occurrence_indices] = True
        
        return result

    def compute_polarity_and_nb_neighbors(self):
        """
        Compute the mean polarity and the number of neighbors for each bacteria.
        
        """

        print('COMPUTE NUMBER OF NEIGHBOURS')
        column_x, column_y = self.tool.gen_coord_str(self.par.n_nodes, xy=False)
        column_angles = self.tool.gen_string_numbered(n=self.par.n_nodes, str_name="ang")
        neighbor_max_distance = self.par.width / self.par.scale * 1.2

        for frame in tqdm(self.frame_indices):

            cond_time = self.df.loc[:, 'frame'] == frame
            n_bact = self.df.loc[cond_time, column_x].shape[0]
            coord = np.column_stack((self.df.loc[cond_time, column_x].values.T.flatten(), self.df.loc[cond_time, column_y].values.T.flatten()))
            angles = self.df.loc[cond_time, column_angles].values
            mean_angles = self.tool.mean_angle(angles=angles, axis=1)
            # Warning use tile and no repeat here !!!
            mean_angles_repeat = np.tile(mean_angles, self.par.n_nodes)
            tree = kdtree(coord)
            dist, ind = tree.query(coord, k=self.par.kn)
            id_node, id_bact = np.divmod(ind, n_bact)
            array_same_bact = np.tile((np.ones((self.par.kn, n_bact)) * np.arange(n_bact)).T.astype(int), (self.par.n_nodes,1))
            cond_same_bact = id_bact == array_same_bact
            cond_dist =  dist > neighbor_max_distance
            id_bact_reshape = id_bact.reshape((n_bact, int(self.par.n_nodes * self.par.kn)), order='F')

            # MEAN POLARITY COMPUTATION
            # Find the index of the bacteria among the id of nodes
            neighbours_angles = mean_angles_repeat[ind]
            bacteria_nodes_angles = np.reshape(np.repeat(mean_angles_repeat, self.par.kn), (int(n_bact*self.par.n_nodes), self.par.kn))
            polarity = np.cos(bacteria_nodes_angles - neighbours_angles)
            # polarity[cond_same_bact | cond_dist] = np.nan
            # Reshape to mean polarity of a same bact
            polarity = polarity.reshape((n_bact, int(self.par.n_nodes*self.par.kn)), order='F')
            # Put all the non intersting bact id to nan and reshape to compute the occurence
            id_bact_with_cond = np.where(cond_same_bact | cond_dist, np.nan, id_bact).reshape((n_bact, int(self.par.n_nodes * self.par.kn)), order='F')
            # Condition to avoid to count one neighbor several time
            cond_same_multiple_same_nei = np.apply_along_axis(self.first_occurrence, axis=1, arr=id_bact_with_cond)
            # Compute the number of True - 1 to note take into account the unique nan
            neighbors_count = np.sum(cond_same_multiple_same_nei, axis=1) - 1
            polarity[~cond_same_multiple_same_nei] = np.nan
            # We do not take into account the orientation of itself in the mean polarity (1:)
            polarity = np.nanmean(polarity[:, 1:], axis=1)
            # Put polarity one to bacteria without neighbors
            polarity[neighbors_count==0] = 1
            self.df.loc[cond_time, 'mean_angle'] = mean_angles
            self.df.loc[cond_time, 'n_neighbours'] = neighbors_count
            self.df.loc[cond_time, 'mean_polarity'] = polarity


    def nodes_to_neighbours_euclidian_direction(self, x_nodes, y_nodes, ind):
        """
        Compute the direction between the nodes and their k-neighbours
        
        """
        # Flatten the array of the neighbours indices
        ind_flat = np.concatenate(ind)

        # Define the coordinate of each bacterium self.k times (for each neighbour)
        x, y = np.repeat(x_nodes, self.par.kn), np.repeat(y_nodes, self.par.kn)

        # Compute the direction between the bacterium and their k-neighbour
        x_dir = x_nodes[ind_flat] - x
        y_dir = y_nodes[ind_flat] - y

        # Normalise the previous directions
        norm_dir = np.linalg.norm(np.array([x_dir, y_dir]), axis=0)

        # Direction is null vector for superposed neighbours
        norm_dir[norm_dir == 0] = np.inf

        return np.reshape(x_dir / norm_dir, ind.shape), np.reshape(y_dir / norm_dir, ind.shape)
    

    def compute_polarity_and_nb_neighbors_angle_view(self):
        """
        Compute the mean polarity and the number of neighbors for each bacteria.
        
        """

        print('COMPUTE NUMBER OF NEIGHBOURS')
        neighbor_max_distance = self.par.width / self.par.scale * 1.2
        x_main_pole = self.df.loc[:, 'x_main_pole'].values
        y_main_pole = self.df.loc[:, 'y_main_pole'].values
        x_second_pole = self.df.loc[:, 'x_second_pole'].values
        y_second_pole = self.df.loc[:, 'y_second_pole'].values
        all_angles = np.arctan2(y_main_pole - y_second_pole, x_main_pole - x_second_pole)
        cond_not_nan = ~np.isnan(self.df.loc[:, 'x_main_pole'].values)

        for frame in tqdm(self.frame_indices):

            cond_time = self.df.loc[:, 'frame'] == frame
            conditions = cond_time & cond_not_nan
            n_bact = self.df.loc[conditions, 'x0'].shape
            coord = np.column_stack((self.df.loc[conditions, 'x_main_pole'].values, self.df.loc[conditions, 'y_main_pole'].values))
            # Some poles cannot be detected because of tracking problems
            angles = all_angles[conditions]
            # Warning use tile and no repeat here !!!
            tree = kdtree(coord)
            dist, ind = tree.query(coord, k=self.par.kn)
            _, id_bact = np.divmod(ind, n_bact)
            cond_dist =  dist > neighbor_max_distance

            # MEAN POLARITY COMPUTATION
            nodes_to_neighbors_direction_x, nodes_to_neighbors_direction_y = self.nodes_to_neighbours_euclidian_direction(coord[:, 0], coord[:, 1], ind)
            scalar_product = nodes_to_neighbors_direction_x * np.cos(angles[:, np.newaxis]) + nodes_to_neighbors_direction_y * np.sin(angles[:, np.newaxis])
            cond_not_in_angle_view = scalar_product < np.cos(self.par.angle_view)

            neighbours_angles = angles[ind]
            polarity = np.cos(angles[:, np.newaxis] - neighbours_angles)
            polarity[cond_dist | cond_not_in_angle_view] = np.nan
            # First column is polarity with itself
            polarity[:, 0] = np.nan
            neighbors_count = np.sum((~cond_dist & ~cond_not_in_angle_view)[:, 1:], axis=1)
            # We do not take into account the orientation of itself in the mean polarity (1:)
            polarity = np.nanmean(polarity, axis=1)
            # Put polarity one to bacteria without neighbors
            polarity[neighbors_count==0] = 1
            self.df.loc[conditions, 'n_neighbours_igoshin'] = neighbors_count
            self.df.loc[conditions, 'mean_polarity_igoshin'] = polarity