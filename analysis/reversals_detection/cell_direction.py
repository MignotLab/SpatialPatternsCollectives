import numpy as np
import pandas as pd
import os
from multiprocessing import Pool, cpu_count

from . import tools


def _reorder_single_trajectory_poles(args):
    """
    Reorder the poles of a single bacterium trajectory to maintain consistent orientation.

    For a given trajectory (i.e., all frames corresponding to one track ID), this function
    checks the orientation between consecutive skeleton frames. If the orientation appears 
    flipped (based on Euclidean distance between first node of current frame and last node 
    of next frame), it reverses the direction of all subsequent node sequences.

    Parameters
    ----------
    args : tuple
        A tuple containing:
            - track_id : any
                The unique identifier of the trajectory.
            - df_traj : pd.DataFrame
                Sub-dataframe corresponding to one trajectory (one track ID).
            - x_cols : list[str]
                List of column names for x-coordinates of skeleton nodes.
            - y_cols : list[str]
                List of column names for y-coordinates of skeleton nodes.

    Returns
    -------
    pd.DataFrame
        A corrected copy of the trajectory dataframe with reordered nodes if needed.
    """
    id_traj, df_traj, x_cols, y_cols = args

    x_array = df_traj[x_cols].to_numpy()
    y_array = df_traj[y_cols].to_numpy()
    len_traj = len(df_traj)

    for i in range(len_traj - 1):
        dist_normal = np.sqrt((x_array[i, 0] - x_array[i + 1, 0])**2 + (y_array[i, 0] - y_array[i + 1, 0])**2)
        dist_inverse = np.sqrt((x_array[i, 0] - x_array[i + 1, -1])**2 + (y_array[i, 0] - y_array[i + 1, -1])**2)

        if dist_inverse < dist_normal:
            x_array[i + 1:, :] = np.flip(x_array[i + 1:, :], axis=1)
            y_array[i + 1:, :] = np.flip(y_array[i + 1:, :], axis=1)

    # Remet dans un DataFrame identique à df_traj, avec colonnes mises à jour
    df_traj_corrected = df_traj.copy()
    df_traj_corrected.loc[:, x_cols] = x_array
    df_traj_corrected.loc[:, y_cols] = y_array

    return df_traj_corrected


class CellDirection:


    def __init__(self, par, df, end_filename):
        
        # Import class
        self.par = par
        self.tool = tools.Tools()
        self.end_filename = end_filename

        # Read csv tracking file
        self.df = df
        self.df.sort_values(by=[self.par.track_id_column,self.par.t_column],ignore_index=True,inplace=True)

        # Generate tuple for angle name in the dataframe
        self.angle_column = []
        for i in range(self.par.n_nodes):
            
            self.angle_column += ["ang"+str(i)]


    def nodes_directions_unique_node(self, angles, angle_unit='degree'):
        """
        Compute the direction of the cells when only one node is provide
        
        """
        coord = self.df.loc[:,[self.par.x_column, self.par.y_column_middle]].to_numpy()
        coord_s = self.df.loc[:,(self.par.x_column+'s', self.par.y_column_middle+'s')].to_numpy()

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
        array_track_id = self.df.loc[:,self.par.track_id_column].to_numpy()
        cond_ends = array_track_id[1:] - array_track_id[:-1] != 0
        cond_ends = np.concatenate((cond_ends,np.array([True])))
        self.df.loc[cond_ends,["x_pole","y_pole"]] = np.nan


    def _reorder_pole(self):
        """
        Ensure consistent pole orientation for each bacterium across its trajectory.

        This method checks, for each trajectory in the tracking dataframe, whether the 
        sequence of skeleton nodes is oriented consistently over time. If an inversion 
        is detected (i.e., the distance from one frame to the next is shorter when comparing 
        to the opposite pole), the subsequent nodes are flipped to preserve consistent orientation.

        This operation is applied independently to each trajectory (track ID) and is 
        parallelized across all trajectories for improved performance.

        Modifies
        --------
        self.df_tracking : pd.DataFrame
            The tracking dataframe is updated in-place with corrected node coordinates.
        """
        if self.par.n_nodes <= 1:
            return  # Nothing to reorder

        x_columns_name, y_columns_name = self.tool.gen_coord_str(n=self.par.n_nodes, xy=False)
        x_cols = x_columns_name
        y_cols = y_columns_name
        id_col = self.par.track_id_column
        t_col = self.par.t_column

        # Ensure that nodes are ordered in time before grouping by trajectory
        self.df.sort_values(
            by=[id_col, t_col],
            ignore_index=True,
            inplace=True
        )
        
        # Liste des trajectoires à traiter
        grouped = self.df.groupby(id_col)

        # Préparation des arguments pour le pool
        args = [(track_id, group.copy(), x_cols, y_cols) for track_id, group in grouped]

        # Traitement en parallèle
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(_reorder_single_trajectory_poles, args)

        # Réassemblage
        self.df = pd.concat(results, ignore_index=True)


    def nodes_directions(self):
        """
        Compute the direction of each node by detecting the direction of
        the cell from the smoothed trajectories
        
        """
        # Reorder poles to ensure proper orientation of the bacteria in the tracking data
        print('REORDER POLES (parallel)')
        self._reorder_pole()

        print('COMPUTE NODES DIRECTION')
        # Extract coordinates
        x_columns_name, y_columns_name = self.tool.gen_coord_str(n=self.par.n_nodes, xy=False)
        x_columns = self.df.loc[:, x_columns_name].to_numpy()
        y_columns = self.df.loc[:, y_columns_name].to_numpy()
        x_columns_s = self.df.loc[:, self.par.x_column+'s'].to_numpy()
        y_columns_s = self.df.loc[:, self.par.y_column+'s'].to_numpy()

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
        array_track_id = self.df.loc[:, self.par.track_id_column].to_numpy()
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
        cond_main_pole_0 = self.df.loc[:, 'main_pole'].to_numpy() == 0
        cond_main_pole_n = self.df.loc[:, 'main_pole'].to_numpy() == self.par.n_nodes - 1
        angles_array[cond_main_pole_0, :] = np.arctan2(dir_y_0, dir_x_0)[cond_main_pole_0, :]
        angles_array[cond_main_pole_n, :] = np.arctan2(dir_y_n, dir_x_n)[cond_main_pole_n, :]

        # Fill the angles in the dataframe
        self.df.loc[:, self.angle_column] = np.round(angles_array, 3)
        self.df.sort_values(by=[self.par.track_id_column, self.par.t_column], ignore_index=True, inplace=True)