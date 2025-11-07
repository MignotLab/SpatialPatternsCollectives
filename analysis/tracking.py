"""
Author: Jean-Baptiste Saulnier
Date: 2024-11-21

This script provides a tracking implementation for bacteria using a Gaussian-based overlap method 
for associating cells across frames. It includes functions for calculating Gaussian overlaps between 
nodes, reordering poles of bacteria shapes, and correcting tracks based on velocity thresholds.

"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.io import imread
from scipy.spatial import KDTree as kdtree
from skimage.measure import label
from typing import Sequence
from multiprocessing import Pool, cpu_count
import warnings
import os


def _warn(msg: str) -> None:
    """Emit a warning with the right stacklevel so tu la vois au bon endroit."""
    warnings.warn(msg, RuntimeWarning, stacklevel=2)


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


class Tracker:
    """
    A class to perform tracking of bacteria based on detected positions in each frame.
    
    Attributes:
    -----------
    df_tracking: pandas.DataFrame
        A DataFrame containing the detected positions of bacteria in each frame.
    columns_dict: dict
        A dictionary that maps the column names for the x and y coordinates, 
        track IDs, segment IDs, and time column.
    max_velocity: float
        The maximum velocity threshold to determine when a new track ID is assigned.
    n_nodes: int
        The number of nodes (points) used to describe each bacterium in the tracking process.
    kn: int
        The number of nearest neighbors used to identify corresponding bacteria between frames.
    frames: numpy.ndarray
        An array of frame indices (excluding the last frame).
    """
    def __init__(self, df_detection, columns_dict, max_velocity):
        """
        Initialize the Tracker with detection data, column mapping, and parameters.

        Parameters
        ----------
        df_detection : pd.DataFrame
            Input detections, one row per bacterium per frame.
        columns_dict : dict
            Dictionary of column names (e.g., x/y node coordinates, track ID, time).
        max_velocity : float
            Maximum allowed velocity (px/frame). Above this, a new track ID is created.
        """
        self.columns_dict = columns_dict
        self.max_velocity = max_velocity
        self.n_nodes = len(self.columns_dict["x_nodes_columns"])
        self.kn = 3  # number of nearest neighbors to query in KDTree

        df = df_detection.copy()

        # 1) Normalize frame numbers so that tracking always starts at t=0
        t_col = self.columns_dict["t_column"]
        frame_min = df[t_col].min()
        df[t_col] = df[t_col] - frame_min

        # 2) Create a new column for track IDs, initialized to NaN
        id_col = self.columns_dict["track_id_column"]
        df[id_col] = np.nan

        # 3) Drop rows with NaN coordinates (these cannot be used in KDTree)
        nodes_cols = self.columns_dict["x_nodes_columns"] + self.columns_dict["y_nodes_columns"]
        df = df.dropna(subset=nodes_cols).reset_index(drop=True)

        # 4) Assign initial track IDs to all bacteria in the first frame (t=0)
        t0 = int(df[t_col].min())  # should be 0 now
        mask_t0 = (df[t_col] == t0)
        if df.loc[mask_t0, id_col].isna().any():
            n0 = int(mask_t0.sum())
            df.loc[mask_t0, id_col] = np.arange(n0, dtype=np.int64)

        # 5) Store the cleaned dataframe and the list of frames to process
        self.df_tracking = df
        self.frames = np.unique(self.df_tracking[t_col])
        if len(self.frames) <= 1:
            raise ValueError(
                f"[Tracker.__init__] Cannot perform tracking: only {len(self.frames)} frame(s) found."
            )
        # Exclude the last frame (no 'next' frame to match against)
        self.frames = self.frames[:-1]

        # 6) Sanity check logging
        print(
            f"[Tracker.__init__] t0={t0}  "
            f"init_ids={int(mask_t0.sum())}  "
            f"NaN_t0={int(self.df_tracking.loc[mask_t0, id_col].isna().sum())}"
        )


    def compute_gaussian_overlap(self, dist, std=2):
        """
        Compute a Gaussian similarity score between skeleton points of a bacterium and its neighbors.

        This function computes a soft overlap (or similarity) score between each point 
        sampled along the skeleton of a bacterium and the corresponding points of its neighbors.
        The similarity is based on a Gaussian function of the pairwise distance, which 
        emphasizes spatial proximity and smoothly downweights more distant points.

        Mathematically, the overlap is computed as:
            overlap = exp(-dist² / (4 * std²))

        This formulation is inspired by radial basis functions and kernel density estimation, 
        and is commonly used in object tracking to model probabilistic associations.

        Parameters
        ----------
        dist : np.ndarray
            A (nb_bact * nb_nodes, nb_neighbors) array of Euclidean distances 
            between bacterium nodes and neighboring nodes.

        std : float, optional
            Standard deviation of the Gaussian kernel (default is 2). Controls 
            the sensitivity to distance: smaller values yield sharper decay.

        Returns
        -------
        np.ndarray
            An array of shape (nb_bact * nb_nodes, nb_neighbors) containing 
            Gaussian similarity scores for each distance.
        """
        overlaps = np.exp(-dist**2 / (4 * std ** 2))  # Compute the Gaussian overlap based on the distance
        return overlaps


    def compute_track_indices_gaussian_overlap(self, frame):
        """
        Compute the track indices for each bacterium by associating its nodes in the current frame with those in the next frame.

        Parameters:
        -----------
        frame : int
            The current frame number for which the track indices are to be computed.
        """
        # Extract the coordinates of the bacteria in the current and next frame
        cond_frame_actual = self.df_tracking.loc[:, self.columns_dict["t_column"]] == frame
        cond_frame_next = self.df_tracking.loc[:, self.columns_dict["t_column"]] == frame + 1

        x_actual = self.df_tracking.loc[cond_frame_actual, self.columns_dict["x_nodes_columns"]].to_numpy()
        y_actual = self.df_tracking.loc[cond_frame_actual, self.columns_dict["y_nodes_columns"]].to_numpy()
        x_next = self.df_tracking.loc[cond_frame_next, self.columns_dict["x_nodes_columns"]].to_numpy()
        y_next = self.df_tracking.loc[cond_frame_next, self.columns_dict["y_nodes_columns"]].to_numpy()

        coords_actual = np.column_stack((x_actual.flatten(), y_actual.flatten()))
        coords_next = np.column_stack((x_next.flatten(), y_next.flatten()))

        # Use KDTree for efficient nearest neighbor search
        tree = kdtree(coords_actual)  # KDTree for actual frame
        dist, ind = tree.query(coords_next, k=self.kn)  # Query nearest neighbors in the next frame

        nb_bact = len(x_next)  # Number of bacteria in the next frame
        id_bact, _ = np.divmod(ind, self.n_nodes)  # Extract the bacterium and node indices

        gaussian_overlap = self.compute_gaussian_overlap(dist)  # Calculate Gaussian overlap between points
        sort_index = np.argsort(-gaussian_overlap, axis=1)  # Sort overlaps in descending order
        gaussian_overlap_sorted = np.take_along_axis(gaussian_overlap, sort_index, axis=1)  # Sort the overlaps
        id_bact_sorted = np.take_along_axis(id_bact, sort_index, axis=1)  # Sort the bacterium indices

        # Reshape the data into a more suitable format for further processing
        gaussian_overlap_sorted_reshaped = gaussian_overlap_sorted.reshape((nb_bact, int(self.n_nodes*self.kn)))
        id_bact_sorted_reshaped = id_bact_sorted.reshape((nb_bact, int(self.n_nodes*self.kn)))

        # Prepare a DataFrame for further processing
        df_tmp = pd.DataFrame({
            'probability': gaussian_overlap_sorted_reshaped.flatten(),
            'target_id': id_bact_sorted_reshaped.flatten(),
            'cell_id': np.repeat(np.arange(gaussian_overlap_sorted_reshaped.shape[0]), gaussian_overlap_sorted_reshaped.shape[1])
        })

        # Group by 'cell_id' and 'target_id' to sum the probabilities for each target
        summed = df_tmp.groupby(['cell_id', 'target_id'], as_index=False).sum()

        # Sort by 'cell_id' and 'probability' in descending order
        sorted_summed = summed.sort_values(by=['cell_id', 'probability'], ascending=[True, False])

        # Extract the top two target IDs for each bacterium
        results = (sorted_summed.groupby('cell_id', as_index=False)
                   .agg(top_target_id=('target_id', lambda x: x.iloc[0] if len(x) > 0 else np.nan),
                        second_target_id=('target_id', lambda x: x.iloc[1] if len(x) > 1 else np.nan),
                        top_prob=('probability', lambda x: x.iloc[0] if len(x) > 0 else np.nan),
                        second_prob=('probability', lambda x: x.iloc[1] if len(x) > 1 else np.nan)))

        # ... dans compute_track_indices_gaussian_overlap(...), après avoir calculé `results`:

        # Sanity: lignes attendues = nb_bact (une par bact. du frame suivant)
        nb_bact = int(len(self.df_tracking.loc[cond_frame_next]))
        if len(results) != nb_bact:
            _warn(f"[frame={frame}] results rows ({len(results)}) != nb_bact_next ({nb_bact}). "
                "Possible upstream grouping/shape issue.")

        # 1) NaN générés par la RÉSOLUTION DE DOUBLONS ?
        #    -> dans ta boucle, tu mets NaN à tout le monde sauf le meilleur
        nan_after_dedup = results['top_target_id'].isna()
        n_nan_after_dedup = int(nan_after_dedup.sum())

        # Pour quantifier ceux créés spécifiquement par le dédoublonnage, on refait rapidement un check:
        # (si tu veux un comptage plus fin, enregistre top_target_id AVANT le dédoublonnage)
        dup_flag = results.get('top_target_duplicate', pd.Series(False, index=results.index))
        n_dup_rows = int(dup_flag.sum())
        if n_dup_rows > 0:
            _warn(f"[frame={frame}] duplicates detected: {n_dup_rows} rows in conflict; "
                f"NaN after dedup: {n_nan_after_dedup}")

        # 2) AUCUN CANDIDAT ?
        #    Si la table `summed` ne contient pas le cell_id -> pas de candidat => NaN
        #    On calcule le set des cell_id couverts par `summed`
        try:
            cell_ids_all = set(range(nb_bact))
            cell_ids_with_candidate = set(summed['cell_id'].unique())
            cell_ids_without_cand = sorted(cell_ids_all - cell_ids_with_candidate)
            if len(cell_ids_without_cand) > 0:
                _warn(f"[frame={frame}] no-candidate cell_ids (t+1): {cell_ids_without_cand[:10]} "
                    f"(total={len(cell_ids_without_cand)})")
        except Exception as e:
            _warn(f"[frame={frame}] unable to assess no-candidate cell_ids: {e!r}")

        # 3) REMAP VERS UN ID ACTUEL NaN ?
        col_id = self.columns_dict["track_id_column"]
        col_t  = self.columns_dict["t_column"]

        # Important : s’assurer que t0 a bien des IDs (sinon NaN se propagent)
        t0 = self.df_tracking[col_t].min()
        mask_t0 = (self.df_tracking[col_t] == t0)
        if self.df_tracking.loc[mask_t0, col_id].isna().any():
            n0 = int(mask_t0.sum())
            _warn(f"[init ids] first frame (t={t0}) contains NaN IDs; initializing 0..{n0-1}")
            self.df_tracking.loc[mask_t0, col_id] = np.arange(n0, dtype=np.int64)

        indices_actual = self.df_tracking.loc[cond_frame_actual, col_id].to_numpy()
        n_nan_indices_actual = int(np.isnan(indices_actual).sum())
        if n_nan_indices_actual > 0:
            _warn(f"[frame={frame}] indices_actual (t) contains {n_nan_indices_actual} NaN IDs "
                f"over {indices_actual.size} cells → remap may produce NaN.")

        # 4) Calcul des IDs à écrire dans le frame suivant, en DECOMPOSANT les cas
        top_target_id = results['top_target_id'].to_numpy(dtype=float)

        # a) cas remappables (top_target_id non-NaN)
        mask_remap = ~np.isnan(top_target_id)
        mapped = np.full_like(top_target_id, np.nan, dtype=float)
        if mask_remap.any():
            src_idx = top_target_id[mask_remap].astype(int)
            # garde-fou: src_idx doit être dans [0, len(indices_actual)-1]
            bad_bounds = (src_idx < 0) | (src_idx >= indices_actual.size)
            if bad_bounds.any():
                _warn(f"[frame={frame}] remap indices out-of-bounds: {int(bad_bounds.sum())} / {src_idx.size} "
                    f"(min={src_idx.min()}, max={src_idx.max()}, size={indices_actual.size})")
                # on clip pour inspecter (on continue l’analyse)
                src_idx = np.clip(src_idx, 0, max(0, indices_actual.size - 1))

            mapped_vals = indices_actual[src_idx]
            # b) parmi les remappables, lesquels tombent sur un NaN (ID courant manquant) ?
            remap_to_nan = np.isnan(mapped_vals)
            if remap_to_nan.any():
                ex = np.flatnonzero(remap_to_nan)[:10]
                _warn(f"[frame={frame}] remap→NaN: {int(remap_to_nan.sum())} cases (ex indices: {ex.tolist()})")
            mapped[mask_remap] = mapped_vals

        # c) ce qui RESTE NaN après remap (conflits résolus en NaN ou remap→NaN) doit recevoir un NOUVEL ID
        still_nan = np.isnan(mapped)
        n_still_nan = int(still_nan.sum())
        if n_still_nan > 0:
            current_max = self.df_tracking[col_id].max()
            current_max = int(current_max) if np.isfinite(current_max) else -1
            new_ids = np.arange(current_max + 1, current_max + 1 + n_still_nan, dtype=np.int64)
            mapped[still_nan] = new_ids
            _warn(f"[frame={frame}] assigned {n_still_nan} new IDs to unresolved targets "
                f"(conflict or remap→NaN).")

        # 5) Écrit (aucun cast int ici pour laisser le debug en place en amont)
        self.df_tracking.loc[cond_frame_next, col_id] = mapped

        # 6) Check immédiat
        n_nan_written = int(self.df_tracking.loc[cond_frame_next, col_id].isna().sum())
        if n_nan_written > 0:
            _warn(f"[frame={frame}] WARNING: {n_nan_written} NaN IDs persisted in t+1 after assignment "
                f"(this should be 0). Investigate earlier logs.")
    
        # Handle duplicate target IDs
        results['top_target_duplicate'] = results['top_target_id'].duplicated(keep=False)
        for target_id in results.loc[results['top_target_duplicate'], 'top_target_id'].unique():
            mask = (results['top_target_id'] == target_id)
            indices = results.index[mask]
            max_prob_index = results.loc[indices, 'top_prob'].idxmax()
            results.loc[mask, 'top_target_id'] = np.nan
            results.at[max_prob_index, 'top_target_id'] = target_id

        # Update the track IDs in the DataFrame
        indices_actual = self.df_tracking.loc[cond_frame_actual, self.columns_dict["track_id_column"]].to_numpy()
        top_target_id = results.loc[:, 'top_target_id'].to_numpy()
        cond_nan = np.isnan(top_target_id)
        target_indices = indices_actual[top_target_id[~cond_nan].astype(int)]

        index_max = np.nanmax(self.df_tracking.loc[:, self.columns_dict["track_id_column"]])
        if np.isnan(index_max):
            index_max = -1
        target_indices_nan = np.arange(index_max + 1, index_max + 1 + np.sum(cond_nan))
        top_target_id[~cond_nan] = target_indices
        top_target_id[cond_nan] = target_indices_nan

        # Assign the updated track IDs to the next frame
        self.df_tracking.loc[cond_frame_next, self.columns_dict["track_id_column"]] = top_target_id


    def tracking_corrections(self):
        """
        Post-process trajectories based on velocity constraints.

        If the instantaneous velocity between two successive frames exceeds the allowed maximum,
        the trajectory is split, and a new track_id is assigned starting from the point of the jump.
        This process is repeated until no invalid jump remains in any trajectory.
        """
        df = self.df_tracking
        cols = self.columns_dict
        t_col = cols["t_column"]
        track_col = cols["track_id_column"]
        center_node_x = cols["x_nodes_columns"][int(self.n_nodes / 2)]
        center_node_y = cols["y_nodes_columns"][int(self.n_nodes / 2)]

        # Sort to ensure proper frame order within trajectories
        df.sort_values(by=[track_col, t_col], ignore_index=True, inplace=True)

        track_ids = df[track_col].unique()
        max_id = int(np.nanmax(track_ids)) + 1

        for track_id in track_ids:
            if np.isnan(track_id):
                continue

            traj = df[df[track_col] == track_id].copy()
            if len(traj) <= 1:
                continue

            x = traj[center_node_x].to_numpy()
            y = traj[center_node_y].to_numpy()
            t = traj[t_col].to_numpy()

            v = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / np.maximum(np.diff(t), 1)
            v = np.concatenate([v, [0]])  # To align with row count

            # Trouver tous les sauts
            jumps = np.where(v > self.max_velocity)[0]

            if len(jumps) == 0:
                continue

            # On va casser la trajectoire en plusieurs sous-trajectoires
            split_points = [0] + (jumps + 1).tolist() + [len(traj)]
            for i in range(len(split_points) - 1):
                i0, i1 = split_points[i], split_points[i + 1]
                idxs = traj.index[i0:i1]
                df.loc[idxs, track_col] = max_id
                max_id += 1

        self.df_tracking = df


    def reorder_pole(self):
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
        if self.n_nodes <= 1:
            return  # Nothing to reorder

        x_cols = self.columns_dict["x_nodes_columns"]
        y_cols = self.columns_dict["y_nodes_columns"]
        id_col = self.columns_dict["track_id_column"]
        t_col = self.columns_dict["t_column"]

        # Ensure that nodes are ordered in time before grouping by trajectory
        self.df_tracking.sort_values(
            by=[id_col, t_col],
            ignore_index=True,
            inplace=True
        )
        
        # Liste des trajectoires à traiter
        grouped = self.df_tracking.groupby(id_col)

        # Préparation des arguments pour le pool
        args = [(track_id, group.copy(), x_cols, y_cols) for track_id, group in grouped]

        # Traitement en parallèle
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(_reorder_single_trajectory_poles, args)

        # Réassemblage
        self.df_tracking = pd.concat(results, ignore_index=True)


    def track(self):
        """
        Main function for tracking bacterial trajectories across frames.

        This function orchestrates the tracking process by:
        - Iterating through each frame and computing track indices.
        - Reordering the poles of bacteria to ensure consistency in their orientation.
        - Applying corrections to track IDs based on the velocity of the bacterial movement.
        
        It modifies the `self.df_tracking` dataframe to include updated tracking information.
        """
        print("TRACKING")
        # Iterate over each frame and compute the track indices (the IDs of bacteria in each frame)
        for frame in tqdm(self.frames):
            self.compute_track_indices_gaussian_overlap(frame)
            # self.df_tracking = self.compute_track_indices_mask_overlap(frame) # really slow

        print('REORDER POLES')
        # Reorder poles to ensure proper orientation of the bacteria in the tracking data
        self.reorder_pole()

        print('TRACK_ID CORRECTION')
        # Apply corrections to the track IDs by considering the velocity of bacteria movement
        self.tracking_corrections()

        # Return the dataframe containing the updated tracking information
        return self.df_tracking
    


