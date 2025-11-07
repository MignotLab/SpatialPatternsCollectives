import numpy as np
import pandas as pd
from scipy.spatial import KDTree as kdtree
from tqdm import tqdm
import os
from multiprocessing import Pool, cpu_count
import gc

from . import tools
from . import velocities_computation


def _process_single_track_frustration(args):
    """
    Process the cumulative frustration computation for a single trajectory.

    Parameters
    ----------
    args : tuple
        Contains:
        - track_index : any
        - df_traj : pd.DataFrame
        - local_frustration : np.ndarray
        - local_frustration_s : np.ndarray
        - exp_factor : np.ndarray
        - decreasing_rate : float
        - tbf : float
        - correction : float
        - track_id_column : str

    Returns
    -------
    pd.DataFrame
        DataFrame with cumulative frustration columns for the given trajectory.
    """
    (track_index, df_traj, local_frustration, 
     local_frustration_s, exp_factor, decreasing_rate, 
     tbf, correction, track_id_column, cumul_frustration_column) = args

    cond_traj = df_traj[track_id_column] == track_index
    indices = df_traj.index
    traj_frustration = local_frustration[indices]
    traj_frustration_s = local_frustration_s[indices]

    memory = traj_frustration[0] * exp_factor
    memory_s = traj_frustration_s[0] * exp_factor

    cumul = np.zeros(len(traj_frustration))
    cumul_s = np.zeros(len(traj_frustration_s))

    for j in range(len(traj_frustration)):
        cumul[j] = np.nansum(memory)
        cumul_s[j] = np.nansum(memory_s)

        # Roll memory
        memory = np.roll(memory, shift=1)
        memory_s = np.roll(memory_s, shift=1)

        # Decay memory
        decay = np.exp(-decreasing_rate * tbf)
        memory *= decay
        memory_s *= decay

        # Add new value
        memory[0] = traj_frustration[j]
        memory_s[0] = traj_frustration_s[j]

    df_traj[cumul_frustration_column] = cumul/ correction
    df_traj[cumul_frustration_column + '_s'] = cumul_s / correction

    return df_traj[[cumul_frustration_column, cumul_frustration_column + '_s']]


def _first_occurrence(row):
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


def _nodes_to_neighbours_euclidian_direction(x_nodes, y_nodes, ind, kn):
    """
    Compute the direction vectors between each node and its k neighbors.

    Parameters
    ----------
    x_nodes : np.ndarray
    y_nodes : np.ndarray
    ind : np.ndarray
    kn : int

    Returns
    -------
    tuple of np.ndarray
        Normalized direction vectors (x_dir, y_dir), both shaped like `ind`.
    """
    ind_flat = np.concatenate(ind)
    x = np.repeat(x_nodes, kn)
    y = np.repeat(y_nodes, kn)

    x_dir = x_nodes[ind_flat] - x
    y_dir = y_nodes[ind_flat] - y

    norm_dir = np.sqrt(x_dir**2 + y_dir**2)
    norm_dir[norm_dir == 0] = np.inf

    return np.reshape(x_dir / norm_dir, ind.shape), np.reshape(y_dir / norm_dir, ind.shape)


def _process_single_frame_polarity(args):
    """
    Compute mean angle, polarity, number of neighbors, and number of negative-polarity neighbors
    for all bacteria in a single frame, based on their skeletal nodes.

    This function assumes each bacterium is represented by a fixed number of nodes (e.g., 11 nodes per cell),
    each with a position (x, y) and a local angle. The function computes the mean angle of each bacterium,
    and evaluates the local polarity of each node with respect to its spatial neighbors using cosine similarity
    between angles. Neighbors that are either too far or belong to the same bacterium are excluded.

    For each bacterium, the function returns:
    - its mean orientation angle,
    - the number of unique neighboring bacteria (excluding itself and distant nodes),
    - the average polarity (cosine similarity) between its own direction and the directions of valid neighbors,
    - the number of distinct neighboring bacteria that exhibit at least one local node orientation
      opposite to the reference bacterium (i.e., polarity < 0).

    Note: The count of negative-polarity neighbors is based on local comparisons at the node level.
    A bacterium is considered "oppositely oriented" if **any of its nodes** has an angle differing
    by more than 90 degrees (cos(Δθ) < 0) from the mean direction of the reference bacterium.

    Parameters
    ----------
    args : tuple
        Tuple containing the following elements:
        - frame : int
            Frame index being processed.
        - df_frame : pd.DataFrame
            Subset of the full detection dataframe corresponding to the current frame.
        - column_x : list of str
            Column names for x coordinates of the skeleton nodes.
        - column_y : list of str
            Column names for y coordinates of the skeleton nodes.
        - column_angles : list of str
            Column names for angles at each node.
        - neighbor_max_distance : float
            Maximum allowed distance (in pixels) between nodes to be considered neighbors.
        - kn : int
            Number of nearest neighbors per node to query with the KDTree.
        - n_nodes : int
            Number of skeleton nodes per bacterium.
        - tool : object
            Utility object that must include the method `mean_angle(angles, axis=1)`.

    Returns
    -------
    tuple
        A tuple containing:
        - frame : int
            Frame index.
        - index : pd.Index
            Index of the dataframe rows corresponding to the current frame.
        - mean_angles : np.ndarray
            Mean orientation angles (in radians) per bacterium.
        - neighbors_count : np.ndarray
            Number of unique neighboring bacteria (excluding self and distant nodes).
        - polarity_mean : np.ndarray
            Average polarity per bacterium, defined as the mean cosine similarity between the 
            bacterium's own mean orientation and the mean orientations of its neighboring bacteria.
        - negative_neighbors_count : np.ndarray
             For each bacterium, the number of unique neighboring bacteria whose mean orientation 
            is opposite (i.e., cosine similarity < 0) to its own mean orientation.
    """
    frame, df_frame, column_x, column_y, column_angles, neighbor_max_distance, kn, n_nodes, tool = args

    if df_frame.empty:
        return frame, df_frame.index, None, None, None

    n_bact = df_frame[column_x].shape[0]
    coord = np.column_stack((
        df_frame[column_x].to_numpy().T.flatten(),
        df_frame[column_y].to_numpy().T.flatten()
    ))

    angles = df_frame[column_angles].to_numpy()
    mean_angles = tool.mean_angle(angles=angles, axis=1)
    mean_angles_repeat = np.tile(mean_angles, n_nodes)

    tree = kdtree(coord)
    dist, ind = tree.query(coord, k=kn)
    id_node, id_bact = np.divmod(ind, n_bact)

    array_same_bact = np.tile((np.ones((kn, n_bact)) * np.arange(n_bact)).T.astype(int), (n_nodes, 1))
    cond_same_bact = id_bact == array_same_bact
    cond_dist = dist > neighbor_max_distance

    neighbours_angles = mean_angles_repeat[ind]
    bacteria_nodes_angles = np.reshape(np.repeat(mean_angles_repeat, kn), (n_bact * n_nodes, kn))
    polarity = np.cos(bacteria_nodes_angles - neighbours_angles)
    polarity = polarity.reshape((n_bact, n_nodes * kn), order='F')

    id_bact_with_cond = np.where(cond_same_bact | cond_dist, np.nan, id_bact).reshape((n_bact, n_nodes * kn), order='F')
    cond_unique = np.apply_along_axis(_first_occurrence, axis=1, arr=id_bact_with_cond)

    neighbors_count = np.sum(cond_unique, axis=1) - 1
    polarity[~cond_unique] = np.nan

    # Exclude the self-comparison column
    polarity_sub = polarity[:, 1:]
    # Detect rows where all values are NaN
    rows_empty = np.all(np.isnan(polarity_sub), axis=1)
    # Initialize polarity_mean with NaNs
    polarity_mean = np.full(polarity.shape[0], np.nan)
    # Compute mean polarity only for rows with valid neighbors
    rows_valid = ~rows_empty
    polarity_mean[rows_valid] = np.nanmean(polarity_sub[rows_valid], axis=1)
    # Assign polarity = 1 for bacteria with no valid neighbors
    polarity_mean[rows_empty & (neighbors_count == 0)] = 1

    # Vectorized count of unique negative polarity neighbors per bacterium
    mask_negative_valid = (polarity < 0) & cond_unique & ~np.isnan(id_bact_with_cond)
    id_bact_valid = np.where(mask_negative_valid, id_bact_with_cond, -1).astype(int)
    row_ids = np.repeat(np.arange(n_bact), id_bact_valid.shape[1])
    pair_ids = row_ids * n_bact + id_bact_valid.ravel()
    valid_pair_ids = pair_ids[id_bact_valid.ravel() != -1]
    unique_pairs = np.unique(valid_pair_ids)
    unique_bact_ids = unique_pairs // n_bact
    negative_neighbors_count = np.bincount(unique_bact_ids, minlength=n_bact)

    gc.collect()

    return frame, df_frame.index, mean_angles, neighbors_count, polarity_mean, negative_neighbors_count


def _process_single_frame_polarity_angle_view(args):
    """
    Compute mean polarity and number of neighbors for all bacteria in a single frame.

    Parameters
    ----------
    args : tuple
        (frame, df_frame, neighbor_max_distance, kn, angle_view)

    Returns
    -------
    tuple
        (frame, boolean index array (valid cells), neighbors_count array, polarity array)
    """
    frame, df_frame, neighbor_max_distance, kn, angle_view = args

    # Exclure les lignes sans pole détecté
    cond_valid = ~np.isnan(df_frame['x_main_pole'])
    if not np.any(cond_valid):
        return frame, cond_valid, None, None

    df_valid = df_frame[cond_valid]

    # Coordonnées et angles
    x_main = df_valid['x_main_pole'].to_numpy()
    y_main = df_valid['y_main_pole'].to_numpy()
    x_second = df_valid['x_second_pole'].to_numpy()
    y_second = df_valid['y_second_pole'].to_numpy()
    angles = np.arctan2(y_main - y_second, x_main - x_second)

    coord = np.column_stack((x_main, y_main))
    tree = kdtree(coord)
    dist, ind = tree.query(coord, k=kn)
    cond_dist = dist > neighbor_max_distance

    x_dir, y_dir = _nodes_to_neighbours_euclidian_direction(x_main, y_main, ind, kn)
    scalar_product = x_dir * np.cos(angles[:, np.newaxis]) + y_dir * np.sin(angles[:, np.newaxis])
    cond_not_in_angle_view = scalar_product < np.cos(angle_view)

    neighbours_angles = angles[ind]
    polarity = np.cos(angles[:, np.newaxis] - neighbours_angles)
    polarity[cond_dist | cond_not_in_angle_view] = np.nan
    polarity[:, 0] = np.nan

    neighbors_count = np.sum((~cond_dist & ~cond_not_in_angle_view)[:, 1:], axis=1)

    # Exclude the self-comparison column
    polarity_sub = polarity[:, 1:]
    # Detect rows where all values are NaN
    rows_empty = np.all(np.isnan(polarity_sub), axis=1)
    # Initialize polarity_mean with NaNs
    polarity_mean = np.full(polarity.shape[0], np.nan)
    # Compute mean polarity only for rows with valid neighbors
    rows_valid = ~rows_empty
    polarity_mean[rows_valid] = np.nanmean(polarity_sub[rows_valid], axis=1)
    # Assign polarity = 1 for bacteria with no valid neighbors
    polarity_mean[rows_empty & (neighbors_count == 0)] = 1

    gc.collect()
    return frame, df_valid.index, neighbors_count, polarity_mean


class ReversalSignal:


    def __init__(self, par, df, end_filename):
        
        # Import class
        self.par = par
        self.tool = tools.Tools()
        self.end_filename = end_filename

        # Copy the dataframe in cell_direction
        self.df = df
        self.df.sort_values(by=[self.par.track_id_column,self.par.t_column],ignore_index=True,inplace=True)
        self.n_traj = np.max(self.df.loc[:,self.par.track_id_column].to_numpy())
        self.frame_indices = np.unique(self.df.loc[:,self.par.t_column].to_numpy())

        #Import class
        self.vel = velocities_computation.Velocity(par=par, track_id=self.df.loc[:,self.par.track_id_column].to_numpy())

        # Nodes coordinates
        x_columns_name, y_columns_name = self.tool.gen_coord_str(n=self.par.n_nodes, xy=False)
        self.t = self.df.loc[:, self.par.t_column].to_numpy()
        self.coords_x = self.df.loc[:, x_columns_name].to_numpy()
        self.coords_y = self.df.loc[:, y_columns_name].to_numpy()
        self.coords_xs = self.df.loc[:, self.par.x_column+'s'].to_numpy()
        self.coords_ys = self.df.loc[:, self.par.y_column+'s'].to_numpy()
        self.main_pole = self.df.loc[:, 'main_pole'].to_numpy()
        self.reversals = self.df.loc[:, self.par.rev_column].to_numpy()

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


    def compute_local_frustration(self, method='initial'):
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
                            main_pole=self.main_pole)

        # Real velocity computation
        self.vel.compute_vr(x=self.df.loc[:, "x_main_pole"].to_numpy(),
                            y=self.df.loc[:, "y_main_pole"].to_numpy(),
                            t=self.t,
                            align="backward")
        # Mask velocities at reversal frames:
        # With forward differences, vr[:, i] corresponds to the step i -> i+1.
        # A head flip at frame r will blow up vr[:, r], so we set it to NaN.
        rev_mask = self.reversals.astype(bool)
        # If vr is stored on the Velocity instance (as in the class we wrote), index it there:
        self.vel.vr[:, rev_mask] = np.nan

        # Real smoothed velocity computation
        self.vel.compute_vr_s(xs=self.coords_xs,
                              ys=self.coords_ys,
                              t=self.t,
                              align="backward")
        self.vel.vr_s[:, rev_mask] = np.nan

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

        self.df.loc[:,self.par.local_frustration_column] = self.local_frustration
        self.df.loc[:,self.par.local_frustration_column + '_s'] = self.local_frustration_s


    def compute_cumul_frustration(self, method='initial', n_jobs=None):
        """
        Compute the cumulative frustration of the cells before a reversal using the selected method.
        This is done independently for each trajectory and parallelized for performance.

        Parameters
        ----------
        method : str
            The method used for local frustration computation ('initial', 'neighbor', etc.)
        n_jobs : int or None
            Number of parallel processes. If None, use all available CPUs.

        Modifies
        --------
        self.df : pd.DataFrame
            Adds columns self.par.cumul_frustration_column and self.par.cumul_frustration_column + '_s'.
        """
        print("COMPUTE FRUSTRATION (parallel)")

        if n_jobs is None:
            n_jobs = cpu_count()

        # Compute local frustration first
        self.compute_local_frustration(method=method)

        # Prepare arguments by trajectory
        track_indices = self.df[self.par.track_id_column].unique()
        args = [
            (
                track_index,
                self.df[self.df[self.par.track_id_column] == track_index].copy(),
                self.local_frustration,
                self.local_frustration_s,
                self.exp_factor,
                self.par.cumul_frustration_decreasing_rate,
                self.par.tbf,
                self.correction,
                self.par.track_id_column,
                self.par.cumul_frustration_column
            )
            for track_index in track_indices
        ]

        # Parallel processing
        with Pool(processes=n_jobs) as pool:
            results = pool.map(_process_single_track_frustration, args)

        # Merge partial results into the main DataFrame
        df_cumul = pd.concat(results, ignore_index=False)
        self.df.loc[df_cumul.index, self.par.cumul_frustration_column] = df_cumul[self.par.cumul_frustration_column]
        self.df.loc[df_cumul.index, self.par.cumul_frustration_column + '_s'] = df_cumul[self.par.cumul_frustration_column + '_s']


    def compute_polarity_and_nb_neighbors(self, n_jobs=None):
        """
        Compute the mean polarity and the number of neighbors for each bacterium based on all surrounding neighbors.

        This method evaluates, for each bacterium at a given frame, the angular alignment with its spatial neighbors.
        All neighbors within a fixed distance threshold are considered, regardless of orientation or direction.

        Parameters
        ----------
        n_jobs : int or None
            Number of parallel processes. If None, use all available CPUs.

        Modifies
        --------
        self.df : pd.DataFrame
            Adds or updates the columns:
            - 'mean_angle': average orientation of each bacterium
            - self.par.n_neighbours_column: number of valid neighbors within distance
            - self.par.mean_polarity_column: average cosine similarity to neighbors' directions
        """
        print("COMPUTE POLARITY AND NUMBER OF NEIGHBOURS (parallel standard version)")

        if n_jobs is None:
            n_jobs = cpu_count()

        column_x, column_y = self.tool.gen_coord_str(self.par.n_nodes, xy=False)
        column_angles = self.tool.gen_string_numbered(n=self.par.n_nodes, str_name="ang")
        neighbor_max_distance = self.par.width / self.par.scale * 1.2

        args = [
            (
                frame,
                self.df[self.df["frame"] == frame].copy(),
                column_x,
                column_y,
                column_angles,
                neighbor_max_distance,
                self.par.kn,
                self.par.n_nodes,
                self.tool
            )
            for frame in self.frame_indices
        ]

        with Pool(processes=n_jobs) as pool:
            results = pool.map(_process_single_frame_polarity, args)

        for frame, cond_time, mean_angles, neighbors_count, polarity_mean, negative_neighbors_count in results:
            if mean_angles is not None:
                self.df.loc[cond_time, 'mean_angle'] = mean_angles
                self.df.loc[cond_time, self.par.n_neighbours_column] = neighbors_count
                self.df.loc[cond_time, self.par.mean_polarity_column] = polarity_mean
                self.df.loc[cond_time, self.par.n_neg_neighbours_column] = negative_neighbors_count


    def compute_polarity_and_nb_neighbors_angle_view(self, n_jobs=None):
        """
        Compute the mean polarity and the number of neighbors in the directional field of view of each bacterium.

        This method evaluates the polarity using only the neighbors that are located within the angular 
        forward-facing view of each bacterium (defined by its main pole). It mimics directional perception 
        as in certain biological models (e.g., Igoshin et al.).

        Parameters
        ----------
        n_jobs : int or None
            Number of parallel processes. If None, use all available CPUs.

        Modifies
        --------
        self.df : pd.DataFrame
            Adds or updates the columns:
            - self.par.n_neighbours_igoshin_column: number of neighbors in the directional field of view
            - self.par.mean_polarity_igoshin_column: polarity computed with only forward-facing neighbors
        """
        print("COMPUTE POLARITY AND NUMBER OF NEIGHBOURS (parallel angle view version)")

        if n_jobs is None:
            n_jobs = cpu_count()

        neighbor_max_distance = self.par.width / self.par.scale * 1.2

        args = [
            (
                frame,
                self.df[self.df["frame"] == frame].copy(),  # Ne garder que la frame courante
                neighbor_max_distance,
                self.par.kn,
                self.par.angle_view
            )
            for frame in self.frame_indices
        ]

        with Pool(processes=n_jobs) as pool:
            results = pool.map(_process_single_frame_polarity_angle_view, args)

        for frame, cond_valid, neighbors_count, polarity_mean in results:
            if neighbors_count is not None:
                self.df.loc[cond_valid, self.par.n_neighbours_igoshin_column] = neighbors_count
                self.df.loc[cond_valid, self.par.mean_polarity_igoshin_column] = polarity_mean