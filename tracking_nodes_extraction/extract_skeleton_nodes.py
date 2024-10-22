import numpy as np 
import pandas as pd 
from skimage.measure import regionprops_table
from skimage.morphology import skeletonize
from skan.csr import Skeleton as skan_skel
import cv2
from joblib import Parallel, delayed
from tqdm import tqdm
import os
import re
import matplotlib.pyplot as plt

from parameters import Parameters
from tools import Tools
par = Parameters()


_n_skel_point = par.n_nodes


def calculate_cumulative_distances(coords):
    # Calculate distances between consecutive points
    deltas = np.diff(coords, axis=0)
    distances = np.sqrt((deltas ** 2).sum(axis=1))
    # Calculate cumulative distances
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

    return cumulative_distances


def interpolate_coordinates(coords, distances, target_distances):
    # Interpolate x and y coordinates separately
    x_coords = np.interp(target_distances, distances, coords[:, 0])
    y_coords = np.interp(target_distances, distances, coords[:, 1])

    return np.vstack((x_coords, y_coords)).T


def extract_equidistant_coordinates(coords, num_points):
    # Calculate cumulative distances
    cumulative_distances = calculate_cumulative_distances(coords)
    len_skel = cumulative_distances[-1]
    # Generate target distances
    total_distance = cumulative_distances[-1]
    target_distances = np.linspace(0, total_distance, num_points)
    # Interpolate coordinates
    equidistant_coords = interpolate_coordinates(coords, cumulative_distances, target_distances)

    return equidistant_coords, len_skel


def get_points(regionmask):
    """
    Extra_properties function for regionprops
    
    """
    skel = skeletonize(regionmask, method='lee')
    try:
        coords_skel = skan_skel(skel).path_coordinates(0)
        if len(coords_skel) < _n_skel_point:
            return np.full(_n_skel_point*2 + 2, np.nan)
    except:
        return np.full(_n_skel_point*2 + 2, np.nan)

    n_paths = skan_skel(skel).n_paths

    equidistant_coords, len_skel = extract_equidistant_coordinates(coords_skel, _n_skel_point)

    # Put to nan if the object is less than the minimal size
    if len_skel < par.min_size_bacteria / par.scale:
        return np.full(_n_skel_point*2 + 2, np.nan)

    return np.concatenate((equidistant_coords.flatten(), np.array([len_skel, n_paths])))


def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')


def list_files_in_directory(directory_path):
    files = [filename for filename in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, filename))]
    return sorted(files, key=extract_number)


class ExtractSkeletonNodes:
    """ Class for detection of centroid and points along the skeleton of object and tracking """


    def __init__(self, indices_frame_df, n_im, n_jobs):

        self.par = Parameters()
        self.tool = Tools()
        self.indices_frame_df = indices_frame_df
        self.n_im = n_im
        self.n_jobs = n_jobs
        # Column renamed
        self.seg_id_column = 'id_seg'
        self.track_id_column = 'id'
        self.t_column = 'frame'
        self.x_centroid_column = 'x_centroid'
        self.y_centroid_column = 'y_centroid'
        self.xy_nodes_columns = self.tool.gen_coord_str(n=_n_skel_point, xy=True)
        self.x_nodes_columns, self.y_nodes_columns = self.tool.gen_coord_str(n=_n_skel_point, xy=False)
        self.area_column = 'area'
        self.len_skel_column = 'len_skel'
        self.n_paths_column = 'n_paths'
        self.traj_length_column = 'traj_length'
        # Folder of the image sequence
        self.columns_name_init = [self.seg_id_column, self.x_centroid_column, self.y_centroid_column, self.area_column]
        self.columns_name_init += self.xy_nodes_columns
        self.columns_name_init += [self.len_skel_column, self.n_paths_column, self.t_column]

        self.columns_name_reorder = [self.seg_id_column, self.track_id_column, self.t_column] + self.xy_nodes_columns + [self.x_centroid_column, self.y_centroid_column, self.len_skel_column, self.area_column, self.n_paths_column]

        # Import the DataFrame
        self.df = pd.read_csv(self.par.name_folder_csv + self.par.name_file_csv)
        if self.par.track_mate_csv:
            self.df = self.df.iloc[3:,2:].astype(float).reset_index(drop=True)

        # Always sort the values in the same order, x and y inverted is normal
        self.df = self.df.rename(columns={self.par.seg_id_column: self.seg_id_column, self.par.area_column: self.area_column, self.par.track_id_column: self.track_id_column, self.par.x_centroid_column: self.y_centroid_column, self.par.y_centroid_column: self.x_centroid_column})
        

    def detection(self, label_image, frame_id):
        """
        Extraction of centroids and skeleton points of labeled object in the image

        Parameters
        ----------
            label_image : (M, N[, P]) ndarray
                Labeled input image. Labels with value 0 are ignored.

        Returns
        -------
             df : pandas DataFrame
                 The Dataframe of tracking trajectories.
                 Columns: [id_seg, id, frame, x_c, y_c, x_0, y_0, ..., x_n, y_n]
        
        """
        df_skel_nodes = pd.DataFrame(
                        regionprops_table(label_image=label_image,
                                        intensity_image=label_image,
                                        properties=['bbox', 'intensity_max', 'centroid', 'area'],
                                        extra_properties=(get_points,)
                                        )
                        )

        # Find the coordinates in the all image
        tmp = self.tool.gen_string_numbered(n=2*_n_skel_point, str_name="get_points-")
        str_x = tmp[::2]
        str_y = tmp[1::2]
        df_skel_nodes.loc[:, str_x] += df_skel_nodes.loc[:, 'bbox-0'].values[:, None]
        df_skel_nodes.loc[:, str_y] += df_skel_nodes.loc[:, 'bbox-1'].values[:, None]

        # Add frame id column
        df_skel_nodes.loc[:, self.t_column] = np.ones(len(df_skel_nodes)) * frame_id
        # Remove the four first columns 
        # (bbox: (min_row, min_col, max_row, max_col)) provide by regionsprops
        df_skel_nodes = df_skel_nodes.iloc[:, 4:]
        # Give a new name for the df columns
        df_skel_nodes.columns = self.columns_name_init
        # Add the track_id column of df in df_tracks_frame, x and y are inverted to match with the skeleton nodes
        cond_frame_df = self.df.loc[:, self.par.t_column] == frame_id
        df_skel_nodes = df_skel_nodes.merge(self.df.loc[cond_frame_df, (self.seg_id_column, self.track_id_column)], how='right', on=self.seg_id_column)
        # Reorder the columns
        df_skel_nodes = df_skel_nodes.loc[:, self.columns_name_reorder]

        return df_skel_nodes


    def start(self):
        """
        Parallelization of the detection function
        
        """
        file_names = list_files_in_directory(self.par.name_folder_seg_movie)
        # Parallelization with joblib
        print("DETECTION")
        dfs = Parallel(n_jobs=self.n_jobs)(
            delayed(self.detection)(
                cv2.imread(self.par.name_folder_seg_movie+file_name, cv2.IMREAD_UNCHANGED), i
            ) for file_name, i in tqdm(zip(file_names, self.indices_frame_df), total=len(file_names))
        )
        df = pd.concat(dfs)
        df.sort_values(by=[self.seg_id_column, self.t_column], ignore_index=True, inplace=True)
        # Absolutly needed to remove nans before reorder pole
        # Reorder pole could not work as well as it did if too much missed frames are present between two detection
        df = df.dropna().reset_index(drop=True)
        columns_to_rescaled = [self.x_centroid_column, self.y_centroid_column] + self.xy_nodes_columns + [self.len_skel_column]
        df.loc[:, columns_to_rescaled] *= self.par.scale
        df.loc[:, self.area_column] *= self.par.scale ** 2
        # Reorder the poles before the correction, IMPORTANT !
        index_traj = np.unique(self.df.loc[:, self.track_id_column], return_counts=False)
        self.reorder_pole(df, index_traj)
        df.sort_values(by=[self.track_id_column, self.t_column], ignore_index=True, inplace=True)
 
        return df

    
    def save(self, df, path):
        """
        Save dataframe as csv file
        
        """
        df.to_csv(path, header=True, index=False)



    def compute_trajectory_length(self, df):
        """
        Compute the length of the trajetories and add a column in the dataframe
        
        """
        df_copy = df.copy()
        __, counts = np.unique(df_copy.loc[:, self.track_id_column].values, return_counts=True)
        counts = np.repeat(counts, repeats=counts, axis=0)
        df_copy.loc[:, self.traj_length_column] = counts

        return df_copy


    def compute_velocity(self, track_id, t, x, y):
        """
        Compute the velocity on the trajectories
        
        """
        # Class object
        velocity = np.zeros(len(track_id)) * np.nan

        # Build a boolean array to detect the change of trajectories
        cond_change_traj = track_id[1:] != track_id[:-1]
        cond_change_traj = np.concatenate((cond_change_traj, np.array([True])))

        time_diff = t[1:] - t[:-1]
        time_diff[time_diff==0] = 1
        velocity[:-1] = np.linalg.norm(np.array([x[1:]-x[:-1], y[1:]-y[:-1]]), axis=0) / (time_diff * self.par.tbf)

        # Remove the velocity computed between trajectories
        velocity[cond_change_traj] = np.nan

        return velocity, cond_change_traj


    def reorder_pole(self, df, index_traj):
        """
        Reorder the pole of the bacteria in the dataframe in the case 
        where several point are detected along the bacteria shape for 
        each cell
        
        """
        df_tmp = df.copy()
        if _n_skel_point > 1:
            # I only keep the poles coordinates
            print('REORDER POLES')
            for id_traj in tqdm(index_traj):

                cond_traj = df_tmp.loc[:, self.track_id_column] == id_traj
                len_traj = len(df.loc[cond_traj])
                # Extract the coordinates as numpy array
                x_array = df_tmp.loc[cond_traj, self.x_nodes_columns].values
                y_array = df_tmp.loc[cond_traj, self.y_nodes_columns].values

                for i in range(len_traj-1):

                    # Compute the difference between consecutive coordinates
                    dist_normal = np.sqrt((x_array[i, 0] - x_array[i+1, 0])**2 + (y_array[i, 0] - y_array[i+1, 0])**2)
                    dist_inverse = np.sqrt((x_array[i, 0] - x_array[i+1, -1])**2 + (y_array[i, 0] - y_array[i+1, -1])**2)
                
                    if dist_inverse < dist_normal:

                        x_array[i+1:, :] = np.flip(x_array[i+1:, :], axis=1)
                        y_array[i+1:, :] = np.flip(y_array[i+1:, :], axis=1)

                df_tmp.loc[cond_traj, self.x_nodes_columns] = x_array.copy()
                df_tmp.loc[cond_traj, self.y_nodes_columns] = y_array.copy()

        return df_tmp


    def correction_detection(self, df):
        """
        Allow to (if needed) add point in trajectories with "holes"
        When you use the Kalman filter on TrackMate the movie of the tracking could
        detect a cell at t-1 not at t but find the cell at t+1.
        Also if at some point the velocity is to high, then this point will be remove
        and the one after also until a mean velocity less than max_vel is reach.
        If the number of iteration are reach before, then this trajectory will be remove.
        
        """
        self.x_nodes_columns
        print("LOAD AND SORT FILE...")
        df_copy = df.copy()
        nb_traj_init = len(np.unique(df_copy.loc[:, self.track_id_column]))
        df_copy.sort_values(by=[self.track_id_column, self.t_column], ignore_index=True, inplace=True)

        print("REMOVE NAN TRACK_ID")
        cond_nan_track_id = np.isnan(df_copy.loc[:, self.track_id_column])
        df_copy = df_copy.loc[~cond_nan_track_id, :]

        # print("REMOVE TRAJECTORIES OF SMALL OBJECT")
        # self.remove_small_object()

        print('REMOVE HIGH SPEED POINT')
        for i in tqdm(range(self.par.frame_gap+1)):
            df_copy.sort_values(by=[self.track_id_column, self.t_column], ignore_index=True, inplace=True)
            x = df_copy.loc[:, self.x_nodes_columns[int(_n_skel_point/2)+1]].values
            y = df_copy.loc[:, self.y_nodes_columns[int(_n_skel_point/2)+1]].values
            t = df_copy.loc[:, self.t_column].values
            track_id = df_copy.loc[:, self.track_id_column].values
            velocity, cond_change_traj = self.compute_velocity(track_id=track_id, t=t, x=x, y=y)
            if i == 0:
                plt.hist(velocity, bins=100, label=np.round(np.nanmean(velocity),3))
                plt.title('velocity histogram before correction')
                plt.legend()
            cond_normal_velocity = velocity < self.par.max_velocity
            df_copy  = df_copy.loc[cond_normal_velocity | cond_change_traj, :]

        print("REMOVE ALL NANS")
        df_copy = df_copy.dropna().reset_index(drop=True)

        print('REMOVE TRAJECTORIES WITH A HIGH NUMBER OF CONSECUTIVES HOLES')
        df_copy.sort_values(by=[self.track_id_column, self.t_column], ignore_index=True, inplace=True)
        id_array = df_copy.loc[:, self.track_id_column].values
        frame_array = df_copy.loc[:, self.t_column].values

        # Build a boolean array to detect the change of trajectories
        cond_change_traj = id_array[1:] != id_array[:-1]
        # Array to know the step time between two consecutive event
        time_diff = frame_array[1:] - frame_array[:-1]
        # During change of traj it could be have time_diff != 1 but we don't want to take it into account
        time_diff[cond_change_traj] = 1
        # Add the last point of the array manually
        time_diff = np.concatenate((time_diff, np.array([1])))
        max_diff = int(np.max(time_diff))
        cond_time_diff = np.zeros(len(time_diff), dtype=bool)

        for i in tqdm(range(self.par.frame_gap+2, max_diff+1)):
            cond_time_diff_i = time_diff == i
            cond_time_diff = cond_time_diff | cond_time_diff_i
        remove_indices = np.unique(df_copy.loc[cond_time_diff, self.track_id_column])
        cond_remove_indices = np.zeros(len(time_diff), dtype=bool)

        for traj_id in tqdm(remove_indices):
            cond_traj = df_copy.loc[:, self.track_id_column].values == traj_id
            cond_remove_indices = cond_remove_indices | cond_traj
        df_copy = df_copy.loc[~cond_remove_indices].reset_index(drop=True)

        print('FILL THE HOLES')
        df_copy.sort_values(by=[self.track_id_column, self.t_column], ignore_index=True, inplace=True)
        df_copy_cor = df_copy.loc[:, self.columns_name_reorder].copy()

        id_array = df_copy.loc[:, self.track_id_column].values
        frame_array = df_copy.loc[:, self.t_column].values
        id_seg_array = df_copy.loc[:, self.seg_id_column].values
        n_paths_array = df_copy.loc[:, self.n_paths_column].values

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
        columns_coord = self.tool.gen_coord_str(n=_n_skel_point, xy=True) + [self.x_centroid_column, self.y_centroid_column]
        # Iteration over the time difference between consecutive bcterium
        max_diff = int(np.max(time_diff))

        for i in tqdm(range(2, self.par.frame_gap+2)):
            cond_time_diff_i = time_diff == i

            # Fill the dataframe with the missing timestep
            for j in range(1, i):
                ### Creates new rows for the missing coordinates which are not present
                # The id_seg is the same than at in time 1
                array_id_seg = id_seg_array[cond_time_diff_i]
                # The id is the same than at in time 1
                array_id = id_array[cond_time_diff_i]
                # The path let's say it is the same
                array_n_paths = n_paths_array[cond_time_diff_i]
                # The value of the frame is increase by j
                array_frame_i = frame_array[cond_time_diff_i] + j
                # Build the new coordinates
                coords_time_1 = df_copy.loc[cond_time_diff_i, columns_coord].values
                coords_time_2 = df_copy.loc[np.roll(cond_time_diff_i, shift=1), columns_coord].values
                # Create a vector in the direction of the two points
                vector_coord = coords_time_2 - coords_time_1
                array_coord = coords_time_1 + j * vector_coord / i
                # The lenght of the skel and the area are the mean between the two time point
                len_skel_time_1 = df_copy.loc[cond_time_diff_i, self.len_skel_column].values
                len_skel_time_2 = df_copy.loc[np.roll(cond_time_diff_i, shift=1), self.len_skel_column].values
                area_time_1 = df_copy.loc[cond_time_diff_i, self.area_column].values
                area_time_2 = df_copy.loc[np.roll(cond_time_diff_i, shift=1), self.area_column].values
                array_len_skel = (len_skel_time_2 + len_skel_time_1) / 2
                array_area = (area_time_2 + area_time_1) / 2
                # Build final array (the order is important and should be the same than in self.columns_name_reorder)
                array_final = np.concatenate((np.array([array_id_seg]), 
                                              np.array([array_id]), 
                                              np.array([array_frame_i]), 
                                              array_coord.T, 
                                              np.array([array_len_skel]), 
                                              np.array([array_area]), 
                                              np.array([array_n_paths])), 
                                              axis=0).T

                df_copy_cor = pd.concat([df_copy_cor, pd.DataFrame(array_final, columns=self.columns_name_reorder)], axis=0, ignore_index=True)

        df_copy_cor = df_copy_cor.dropna()
        df_copy_cor.sort_values(by=[self.track_id_column, self.t_column], ignore_index=True, inplace=True)

        print('COMPUTE VELOCITIES')
        # Recompute the velocity and add it as a column in the dataframe
        x = df_copy_cor.loc[:, self.x_nodes_columns[int(_n_skel_point/2)+1]].values
        y = df_copy_cor.loc[:, self.y_nodes_columns[int(_n_skel_point/2)+1]].values
        t = df_copy_cor.loc[:, self.t_column].values
        track_id = df_copy_cor.loc[:, self.track_id_column].values
        velocity, cond_change_traj = self.compute_velocity(track_id=track_id, t=t, x=x, y=y)
        df_copy_cor.loc[:, 'velocity'] = velocity

        plt.hist(velocity, bins=100, label=np.round(np.nanmean(velocity),3))
        plt.title('velocity histogram after correction')
        plt.legend()

        print('START COMPUTE THE LENGTH OF THE TRAJECTORIES')
        df_copy_cor = self.compute_trajectory_length(df_copy_cor)
        print('END COMPUTE THE LENGTH OF THE TRAJECTORIES')

        print("CONVERT SOME COLUMNS AS INTEGER")
        df_copy_cor[[self.track_id_column, self.seg_id_column, self.t_column, self.n_paths_column]] = df_copy_cor[[self.track_id_column, self.seg_id_column, self.t_column, self.n_paths_column]].astype(int)

        # See if the things are good corrected
        id_array = df_copy_cor.loc[:, self.track_id_column].values
        frame_array = df_copy_cor.loc[:, self.t_column].values
        cond_change_traj = id_array[1:] != id_array[:-1]
        time_diff = frame_array[1:] - frame_array[:-1]
        time_diff[cond_change_traj] = 1
        print('Maximal time difference inside trajectories after correction: ', np.max(time_diff))
        print('Minimal time difference inside trajectories after correction: ', np.min(time_diff))
        print('Number of minimal difference: ', len(np.where(time_diff==0)[0]))
        nb_traj_end = len(np.unique(df_copy_cor.loc[:, self.track_id_column]))
        print('Number of trajectories before correction:', nb_traj_init)
        print('Number of trajectories after correction:', nb_traj_end)

        return df_copy_cor