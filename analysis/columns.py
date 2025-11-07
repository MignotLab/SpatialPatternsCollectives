"""
Author: Jean-Baptiste Saulnier
Date: 2024-11-21

"""
import utils as tl
from typing import Any, cast


def define_columns(n_nodes: int) -> dict[str, Any]:
    """
    Generate column names for consistent use in dataframes.

    Parameters
    ----------
    n_nodes : int
        Number of nodes used for skeleton analysis.

    Returns
    -------
    dict
        A dictionary of column names.
    """
    seg_id_column = 'id_seg'
    track_id_column = 'id'
    t_column = 'frame'
    x_centroid_column = 'x_centroid'
    y_centroid_column = 'y_centroid'
    area_column = 'area'
    len_skel_column = 'len_skel'
    n_paths_column = 'n_paths'
    traj_length_column = 'traj_length'

    # Columns for nodes
    xy_nodes_columns = cast(list[str], tl.gen_coord_str(n=n_nodes, xy=True)) # cast to avoid type warning from pylance
    x_nodes_columns, y_nodes_columns = tl.gen_coord_str(n=n_nodes, xy=False)

    # Column orders
    columns_name = (
        [seg_id_column, t_column, x_centroid_column, y_centroid_column]
        + xy_nodes_columns
        + [len_skel_column, area_column, n_paths_column]
    )
    columns_name_non_ordered = (
        [x_centroid_column, y_centroid_column, seg_id_column, area_column]
        + xy_nodes_columns
        + [len_skel_column, n_paths_column, t_column]
    )

    return {
        "seg_id_column": seg_id_column,
        "track_id_column": track_id_column,
        "t_column": t_column,
        "x_centroid_column": x_centroid_column,
        "y_centroid_column": y_centroid_column,
        "area_column": area_column,
        "len_skel_column": len_skel_column,
        "n_paths_column": n_paths_column,
        "traj_length_column": traj_length_column,
        "xy_nodes_columns": xy_nodes_columns,
        "x_nodes_columns": x_nodes_columns,
        "y_nodes_columns": y_nodes_columns,
        "columns_name": columns_name,
        "columns_name_non_ordered": columns_name_non_ordered,
    }
