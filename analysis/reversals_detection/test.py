# %%
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -- Setup sys.path to import local module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.insert(0, parent_dir)

from analysis.reversals_detection import reversal_signal

def main():
    input_folder = '/Volumes/dock_ssd/Sync/tracking_for_michele/'
    input_filename = 'swarming_movie_3_tracking__DATA_REV_SIG__min_size_smoothed_um=1_um.csv'

    df = pd.read_csv(os.path.join(input_folder, input_filename))
    sig = reversal_signal.ReversalSignal(df=df, end_filename='test')
    sig.compute_polarity_and_nb_neighbors()

    return sig.df

if __name__ == "__main__":
    df = main()

# %%
import matplotlib.pyplot as plt
import numpy as np

def plot_frame_with_neighbor_coloring(df, frame, n_target_neighbors, neighbor_column="n_neg_neighbours"):
    """
    Visualize bacteria in a given frame, coloring based on number of neighbors.

    Parameters
    ----------
    df : pd.DataFrame
        The full dataframe with all frames and polarity/neighbor info.
    frame : int
        The frame to visualize.
    n_target_neighbors : int
        Number of neighbors to highlight in rose (others will be black).
    neighbor_column : str
        Name of the column to use for neighbor counting ("n_neighbours" or "n_neg_neighbours").
    """
    df_frame = df[df["frame"] == frame].copy()

    fig, ax = plt.subplots(figsize=(32, 32))
    ax.set_title(f"Frame {frame} — {neighbor_column} == {n_target_neighbors}")
    ax.set_aspect("equal")
    ax.set_xlim(0, 3200)
    ax.set_ylim(0, 3200)
    ax.invert_yaxis()

    for _, row in df_frame.iterrows():
        # Retrieve node coordinates
        xs = [row[f"x{i}"] for i in range(11)]
        ys = [row[f"y{i}"] for i in range(11)]

        # Is this a target cell?
        is_target = row[neighbor_column] == n_target_neighbors

        # Skeleton color
        color = "deeppink" if is_target else "black"

        # Draw the skeleton
        ax.plot(xs, ys, color=color, linewidth=5)

        # Draw direction vector if main_pole is defined
        if not np.isnan(row["main_pole"]):
            pole_idx = int(row["main_pole"])
            angle_col = f"ang{pole_idx}"
            if angle_col in row and not np.isnan(row[angle_col]):
                x_pole = row["x_main_pole"]
                y_pole = row["y_main_pole"]
                angle = row[angle_col]
                dx = np.cos(angle)
                dy = np.sin(angle)
                vec_color = "blue" if is_target else "red"
                ax.arrow(x_pole, y_pole, dx * 10, dy * 10,
                         head_width=10.0, head_length=15.0,
                         fc=vec_color, ec=vec_color, alpha=0.5, zorder=10)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    frame = 11
    n_target_neighbors = 5
    plot_frame_with_neighbor_coloring(df, frame, n_target_neighbors, neighbor_column="n_neg_neighbours")


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def extract_tbr_from_reversals(df, frame_interval_seconds=2):
    """
    Extract time between reversals (TBR) for each bacterium and return the full list.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least columns 'id', 'frame', and 'reversals' (0 or 1).
    frame_interval_seconds : float
        Duration of one frame in seconds.

    Returns
    -------
    list
        List of all TBR values in seconds (one list across all bacteria).
    """
    tbr_list = []

    for bact_id, df_bact in df.groupby("id"):
        reversal_frames = df_bact.loc[df_bact["reversals"] == 1, "frame"].values
        if len(reversal_frames) >= 2:
            diffs = np.diff(reversal_frames) * frame_interval_seconds / 60
            tbr_list.extend(diffs)

    return tbr_list

if __name__ == "__main__":
    input_folder = '/Volumes/dock_ssd/DATA/a_postdoc_2024/publication/article_myxo_2024/data_for_submitted_paper/non_formatted_data/fig4/frustration_reversals_correlation_simu_rippling/simu_rippling_1__1000_bacts__tbf=2_secondes__space_size=65.csv'
    df = pd.read_csv(input_folder)

    tbr_list = extract_tbr_from_reversals(df, frame_interval_seconds=2.0)
    
    plt.hist(tbr_list, bins=200, color="skyblue", edgecolor="black")
    plt.xlim(0,15)
    plt.xlabel("Time between reversals (s)")
    plt.ylabel("Count")
    plt.title("Distribution of TBR (simu data)")
    plt.tight_layout()
    plt.show()

# %%
import pandas as pd
import numpy as np

path = '/Volumes/dock_ssd/DATA/a_thesis_2019/codes/python/my_libraries_mx/scripts/output/tracking/reversal_analysis_paper_review_2025/swarming_movie_3_tracking/swarming_movie_3_tracking__DATA_REV_SIG__min_size_smoothed_um=1_um.csv'
df = pd.read_csv(path)

# %%
import tools
tool = tools.Tools()
id = 1
cond_id = df.loc[:, 'id'] == id
cond_rev = df.loc[:, 'reversals'] == 1
print(df.loc[cond_id & cond_rev, ('id', 'frame', 'local_frustration','cumul_frustration', 'reversals')])

print(df.loc[1052:1057, ('id', 'frame','local_frustration', 'local_frustration_s', 'cumul_frustration', 'reversals')])



n_nodes = 11
x_columns_name, y_columns_name = tool.gen_coord_str(n=n_nodes, xy=False)
track_id = df.loc[:, 'id'].to_numpy()
cond_change_traj = track_id[1:] != track_id[:-1]
cond_change_traj = np.concatenate((cond_change_traj,np.array([True])))

def compute_vt(x0, y0, x1, y1, xm, ym, xn, yn, main_pole):
    """
    Compute the target velocity of the main pole
    
    """
    # Initialize velocity target
    vt = np.ones((2,len(x0))) * np.nan

    # Compute the velocity
    v0 = np.array([x0-x1, y0-y1])
    vn = np.array([xn-xm, yn-ym])

    cond_pole_0 = main_pole == 0
    cond_pole_n = main_pole == n_nodes - 1

    vt[:, cond_pole_0] = v0[:, cond_pole_0]
    vt[:, cond_pole_n] = vn[:, cond_pole_n]

    norm_vt = np.linalg.norm(vt, axis=0)
    norm_vt[norm_vt==0] = 1
    vt = vt / norm_vt

    return vt

def compute_vr(x, y, t, reversals):
    """
    Compute the real velocity of the centroid
    
    """
    # Initialize velocity target
    vr = np.ones((2, len(x))) * np.nan

    # Compute the velocity
    time_diff = t[1:] - t[:-1]
    time_diff[time_diff==0] = 1
    v0 = np.array([x[1:]-x[:-1], y[1:]-y[:-1]]) / time_diff
    v0 = np.concatenate((v0.T, np.array([v0[:, -1]]))).T

    cond_rev = reversals.astype(bool)

    vr[:, ~cond_change_traj] = v0[:, ~cond_change_traj]

    return vr

def compute_local_frustration(df_init, method='michele'):
    """
    Compute the frustration of the cells before a reversal
    
    """
    df = df_init.copy()
    coords_x = df.loc[:, x_columns_name].to_numpy()
    coords_y = df.loc[:, y_columns_name].to_numpy()
    main_pole = df.loc[:, 'main_pole'].to_numpy()
    reversals = df.loc[:, 'reversals'].to_numpy()
    t = df.loc[:, 'frame'].to_numpy()
    x_column = 'x_centroid'
    y_column = 'y_centroid'
    # Target velocity computation
    vt = compute_vt(x0=coords_x[:,0],
                y0=coords_y[:,0],
                x1=coords_x[:,1],
                y1=coords_y[:,1],
                xm=coords_x[:,-2],
                ym=coords_y[:,-2],
                xn=coords_x[:,-1],
                yn=coords_y[:,-1],
                main_pole=main_pole)
    
    # Real velocity computation
    vr = compute_vr(x=df.loc[:, x_column].to_numpy(),
                y=df.loc[:, y_column].to_numpy(),
                t=t,
                reversals=reversals)

    # Compute the scalar products and put it at -1 in the case there is a nan
    # v_mean = 3.7 #np.nanmean(np.linalg.norm(self.vel.vr,axis=0))
    v_mean = np.nanmean(np.linalg.norm(vr,  axis=0))
    sp_mix = np.sum(vr * (v_mean * vt), axis=0)

    if method == 'initial':
        local_frustration = 1 - sp_mix / v_mean**2

    if method == 'michele':
        sp_vr = np.sum(vr * vr, axis=0)
        sp_vt = np.sum((v_mean * vt) * (v_mean * vt), axis=0)
        # Avoid division by 0
        sp_vr[sp_vr==0] = 1
        sp_vt[sp_vt==0] = 1
        local_frustration = 1 - sp_mix / np.maximum(sp_vr,sp_vt)

    df.loc[:, 'local_frustration_recomputed'] = local_frustration

    return df

df_new = compute_local_frustration(df, method='michele')
print(df_new.loc[1052:1057, ('id', 'frame','local_frustration', 'local_frustration_recomputed', 'cumul_frustration', 'reversals', 'main_pole')])