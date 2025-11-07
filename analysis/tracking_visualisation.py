import os
import re
import numpy as np
import pandas as pd
from typing import List, Optional, Union
from joblib import Parallel, delayed
from tqdm import tqdm
import napari
import colorcet as cc


def extract_number(filename: str) -> Union[int, float]:
    """Extract the first number found in a string for sorting purposes."""
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')


def list_files_in_directory(directory_path: str) -> List[str]:
    """List and numerically sort files in a given directory."""
    files = [
        filename for filename in os.listdir(directory_path)
        if os.path.isfile(os.path.join(directory_path, filename))
    ]
    return sorted(files, key=extract_number)


class VisualizeTracks:
    """
    Class for visualizing tracks and masks in a Napari viewer.
    Handles color assignment using Glasbey palette and consistent track ID coloring.
    """

    def __init__(
        self,
        track_df: pd.DataFrame,
        id_col: str,
        t_col: str,
        x_col: str,
        y_col: str,
        point_size: int = 20,
        x_head_col: Optional[str] = None,
        y_head_col: Optional[str] = None,
        stack_masks: Optional[np.ndarray] = None,
        stack_dia: Optional[np.ndarray] = None,
        step_ims: Optional[int] = None,
        specific_id_col: Optional[str] = None,
        specific_id: Optional[int] = None
        ) -> None:
        """
        Initialize the VisualizeTracks object.

        Parameters:
            track_df (pd.DataFrame): DataFrame containing tracking data.
            id_col (str): Column name for object ID.
            t_col (str): Column name for time/frame.
            x_col (str): Column name for x-coordinate.
            y_col (str): Column name for y-coordinate.
            point_size (int): Size of the points in Napari viewer.
            x_head_col (Optional[str]): Optional column for x coordinate of arrow heads.
            y_head_col (Optional[str]): Optional column for y coordinate of arrow heads.
            stack_masks (Optional[np.ndarray]): Optional 3D or 4D mask array.
            stack_dia (Optional[np.ndarray]): Optional diameter array (same shape as masks).
            step_ims (Optional[int]): Optional step between timepoints in mask stack.
        """
        self.track_df = track_df.copy()
        self.id_column = id_col
        self.t_column = t_col
        self.col_x = x_col
        self.col_y = y_col
        self.point_size = point_size
        self.col_x_head = x_head_col
        self.col_y_head = y_head_col
        self.stack_masks = stack_masks
        self.stack_dia = stack_dia
        self.step_ims = step_ims
        self.specific_id_col = specific_id_col
        self.specific_id = specific_id

        # Glasbey palette (256 RGB colors)
        self.glasbey = np.array([
            tuple(int(c[i:i+2], 16)/255 for i in (0, 2, 4))
            for c in [hex[1:] for hex in cc.glasbey_light] # type: ignore[assignment]
        ], dtype=np.float32)
        # Glasbey est de forme (256, 3), donc on ajoute une couleur noire au début pour l'index -1
        self.glasbey_extended = np.vstack((np.zeros((1, 3)), self.glasbey))  # shape (257, 3)
        self.color_specific = np.array([[1.0, 1.0, 1.0]])  # White for specific ID
        self.size_point_specific = int(point_size * 3)

    def generate_colored_mask_stack(self, n_jobs: int = 4) -> Optional[np.ndarray]:
        """
        Generate a (T, H, W, 3) RGB stack where each object is colored by its track_id.

        Parameters
        ----------
        n_jobs : int
            Number of parallel workers (-1 = all cores).

        Returns
        -------
        Optional[np.ndarray]
            RGB stack of shape (T, H, W, 3), or None if stack_masks is not set.
        """
        if self.stack_masks is None:
            return None

        T, H, W = self.stack_masks.shape

        def process_frame(t: int) -> np.ndarray:
            mask = self.stack_masks[t] # type: ignore[assignment]
            rgb = np.zeros((H, W, 3), dtype=np.float32)

            df_t = self.track_df[self.track_df[self.t_column] == t]
            if df_t.empty or mask.max() == 0:
                return rgb

            seg_to_track_t = df_t.drop_duplicates('id_seg').set_index('id_seg')[self.id_column].astype(int).to_dict()
            max_seg_id = mask.max()
            lut = np.zeros(max_seg_id + 1, dtype=np.int32)
            for seg_id, track_id in seg_to_track_t.items():
                if seg_id <= max_seg_id:
                    lut[seg_id] = track_id

            track_id_image = lut[mask]
            track_id_image[mask == 0] = -1  # background

            rgb = self.glasbey_extended[(track_id_image + 1) % 257]
            return rgb

        results = Parallel(n_jobs=n_jobs)(
            delayed(process_frame)(t) for t in tqdm(range(T), desc="Generating RGB frames")
        )

        rgb_stack = np.stack(results, axis=0) # type: ignore[assignment]

        return rgb_stack

    def visualize_tracks(self, scale: float = 1.0) -> None:
        """
        Display tracks in Napari viewer:
        - RGB mask from segmentation with consistent track coloring.
        - Background (if available).
        - Centroids and heads as colored points.

        Parameters
        ----------
        scale : float
            Scale factor for rescaling coordinates.
        """
        stack_rgb = self.generate_colored_mask_stack()

        track_df2 = self.track_df.copy()
        if self.col_x_head and self.col_y_head:
            track_df2.loc[:, [self.col_x, self.col_y, self.col_x_head, self.col_y_head]] /= scale
        else:
            track_df2.loc[:, [self.col_x, self.col_y]] /= scale

        v = napari.Viewer()

        if self.stack_dia is not None:
            v.add_image(self.stack_dia, name="dia_images", colormap='gray')

        if stack_rgb is not None:
            v.add_image(stack_rgb, name="colored_mask")

        track_ids = self.track_df[self.id_column].astype(int).to_numpy()
        colors = self.glasbey_extended[(track_ids + 1) % 257]
        v.add_points(
            track_df2[[self.t_column, self.col_x, self.col_y]].to_numpy(),
            size=self.point_size,
            face_color=colors, # type: ignore[assignment]
            name="centroids"
        )

        if self.col_x_head and self.col_y_head:
            v.add_points(
                track_df2[[self.t_column, self.col_x_head, self.col_y_head]].to_numpy(),
                size=int(self.point_size / 1.5),
                face_color='white',
                name="heads"
            )

        v.add_tracks(
            track_df2[[self.id_column, self.t_column, self.col_x, self.col_y]],
            name="tracks"
        )

        # Add the specific ID in white and larger
        if self.specific_id_col is not None and self.specific_id is not None:
            # Add the tracks id (not id_seg) of the specific id
            track_df_specific = track_df2[track_df2[self.specific_id_col] == self.specific_id]
            if not track_df_specific.empty:
                v.add_tracks(
                    track_df_specific[[self.id_column, self.t_column, self.col_x, self.col_y]],
                    name=f"tracks_specific_id_{self.specific_id}",
                )
            # Add the centroids of the specific ID in white and larger
            mask_specific = (track_df2[self.specific_id_col] == self.specific_id)
            if mask_specific.any():
                v.add_points(
                    track_df2.loc[mask_specific, [self.t_column, self.col_x, self.col_y]].to_numpy(),
                    size=self.size_point_specific,
                    face_color=self.color_specific, # type: ignore[assignment]
                    name=f"specific_id_{self.specific_id}"
                )

        napari.run()
