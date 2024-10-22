# %%
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 

# Définir la police sans-serif pour les graphiques
plt.rcParams['font.sans-serif'] = 'Arial'
# Utiliser la police sans-serif pour les graphiques
plt.rcParams['font.family'] = 'sans-serif'

# %%
df = pd.read_csv("data/data_rev_min_size_smoothed_um=0.5_um.csv")

# %%
import parameters
par = parameters.Parameters()
n_traj = len(np.unique(df.loc[:,par.track_id_column]))
_path_save = 'W:/jb/python/codes_for_plotting_figures_papers_2023/data_for_paper_supp/reversal_detection/jb_traj/'

def plot_trajectory(x_column,
                    y_column,
                    traj_id='random',
                    zoom=0.1,
                    min_length=1,
                    plot_traj=True,
                    plot_traj_point=False, 
                    plot_traj_s=False,
                    plot_traj_s_point=False,
                    plot_rev=True,
                    plot_rev_s=False,
                    start_and_end=None,
                    **kwargs):
        """
        Plot a trajectories with reversals and smoothed traj in option
        
        """
        s_rev = kwargs.get('s_rev', None)
        color_rev = kwargs.get('color_rev', None)
        linewidth = kwargs.get('linewidth', None)
        linewidth_s = kwargs.get('linewidth_s', None)
        s_point_traj = kwargs.get('s_point_traj', None)
        fontsize = kwargs.get('fontsize', None)
        fontsize_ticks = kwargs.get('fontsize_ticks', None)
        # Using parameters if there exist otherwise they take their default values
        if s_rev is None:
            s_rev = 50
        if color_rev is None:
            color_rev = 'violet'
        if linewidth is None:
            linewidth = 1
        if linewidth_s is None:
            linewidth_s = 1
        if s_point_traj is None:
            s_point_traj = 10
        if fontsize is None:
            fontsize = 30
        if fontsize_ticks is None:
            fontsize_ticks = fontsize / 1.5
        

        if traj_id == 'random':
            # Condition for the traj selection
            trajectory = np.random.randint(n_traj)
            cond_traj = df.loc[:, par.track_id_column] == trajectory

            while len(df.loc[cond_traj, par.track_id_column]) < min_length:
                trajectory = np.random.randint(n_traj)
                cond_traj = df.loc[:, par.track_id_column] == trajectory
        
        else:
            trajectory = traj_id
            cond_traj = df.loc[:, par.track_id_column] == trajectory

        print('traj_id = ', trajectory)
        
        # Coordinates and reversals array
        x = df.loc[cond_traj, x_column] * par.scale
        y = df.loc[cond_traj, y_column] * par.scale
        reversals = df.loc[cond_traj,'reversals'].astype(bool)

        # Coordiante of the smoothed trajectories
        x_s = df.loc[cond_traj, x_column+'s'] * par.scale
        y_s = df.loc[cond_traj, y_column+'s'] * par.scale

        if start_and_end:
            start = start_and_end[0]
            end = start_and_end[1]
            # If start and end parameters are different than 0 and -1
            x = x[start:end]
            y = y[start:end]
            reversals = reversals[start:end]
            x_s = x_s[start:end]
            y_s = y_s[start:end]

        x_min = np.min(x)
        x = x - x_min
        y_min = np.min(y)
        y = y - y_min
        x_s = x_s - x_min
        y_s = y_s - y_min

        ### PLOT ###
        width = (np.max(x) - np.min(x)) * zoom + 0.1
        height = (np.max(y) - np.min(y)) * zoom + 0.1
        
        fig, ax = plt.subplots(figsize=(width, height))

        if plot_traj:
            ax.plot(x, y, color='k', alpha=0.6, linewidth=linewidth, label="Initial trajectory")
            ax.scatter(x.values[0], y.values[0], s=s_rev, c='k', linewidths=3, marker="+", alpha=0.5, label="Start")

        if plot_traj_point:
            ax.scatter(x, y, color='k', alpha=0.5, s=s_point_traj, linewidths=0)

        if plot_traj_s:
            ax.plot(x_s, y_s, color='limegreen', alpha=0.8, linewidth=linewidth_s, label="Smoothed trajectory")
            # ax.scatter(x_s.values[0], y_s.values[0], s=s_rev, c='k', linewidths=3, marker="+", alpha=0.5)

        if plot_traj_s_point:
            ax.scatter(x_s, y_s, color='g', alpha=0.5, s=s_point_traj, linewidths=0)

        if plot_rev:
            ax.scatter(x[reversals], y[reversals], s=s_rev, c='r', marker="x", zorder=10, linewidths=3, alpha=0.5, label="Reversal")

        ax.set_xlabel('µm', fontsize=fontsize)
        ax.set_ylabel('µm', fontsize=fontsize)
        # ax.set_title(str(trajectory))
        ax.tick_params(axis='both',
                   which='major',
                   labelsize=fontsize_ticks)
        ax.legend(loc='upper right', handlelength=1, borderpad=0, frameon=False, fontsize=fontsize_ticks)

        fig.savefig(_path_save+'trajectory_smoothing_example.png', bbox_inches='tight', dpi=100)
        fig.savefig(_path_save+'trajectory_smoothing_example.svg', dpi=100)

# %%
plot_trajectory(x_column='y_centroid', 
                y_column='x_centroid',
                # traj_id='random',
                traj_id=9252,
                min_length=100,
                zoom=4,
                s_rev=200,
                linewidth=2,
                linewidth_s=2,
                plot_traj_s=True,
                start_and_end=[269, -1]
                )
# %%
