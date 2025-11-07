import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
from scipy import stats
import statistics
from scipy.stats import norm
from skimage import io
from tqdm import tqdm
import os

from . import tools

# Définir la police sans-serif pour les graphiques
plt.rcParams['font.sans-serif'] = 'Arial'
# Utiliser la police sans-serif pour les graphiques
plt.rcParams['font.family'] = 'sans-serif'


class Plots:


    def __init__(self, par, df, path_output, filename_no_ext, fontsize, end_filename):
        
        # Import class parameters and tools
        self.par = par
        self.tool = tools.Tools()
        self.path_output = path_output
        self.filename_no_ext = filename_no_ext
        self.fontsize = fontsize
        self.end_filename = end_filename

        # Named saving

        # Reversals dataframe
        self.df_rev = df

        # Number of trajectories
        self.n_traj = len(np.unique(self.df_rev.loc[:,self.par.track_id_column]))

        # Number of frames
        self.n_frames = int(np.max(self.df_rev.loc[:,self.par.t_column]) - np.min(self.df_rev.loc[:,self.par.t_column])) + 1

        # Mean number of bacteria
        tmp = np.zeros(self.n_frames)
        for i in range(self.n_frames):

            cond_frame = self.df_rev.loc[:,self.par.t_column] == i
            tmp[i] = len(self.df_rev.loc[cond_frame,self.par.t_column])
            
        self.mean_n_bact = np.mean(tmp)


    def plot_reversals(self, x_column,
                             y_column,
                             traj_id='random',
                             zoom=0.1,
                             min_length=1,
                             plot_traj=True,
                             plot_traj_point=False, size_traj_point=20,
                             plot_traj_s=False,
                             plot_traj_s_point=False,
                             plot_rev=True,
                             plot_rev_s=False):
        """
        Plot a trajectories with reversals and smoothed traj in option
        
        """

        if traj_id == 'random':

            # Condition for the traj selection
            trajectory = np.random.randint(self.n_traj)
            cond_traj = self.df_rev.loc[:, self.par.track_id_column] == trajectory

            while len(self.df_rev.loc[cond_traj,self.par.track_id_column]) < min_length:

                trajectory = np.random.randint(self.n_traj)
                cond_traj = self.df_rev.loc[:, self.par.track_id_column] == trajectory
        
        else:

            trajectory = traj_id
            cond_traj = self.df_rev.loc[:,self.par.track_id_column] == trajectory

        # Coordinates and reversals array
        x = self.df_rev.loc[cond_traj, x_column] * self.par.scale
        y = self.df_rev.loc[cond_traj, y_column] * self.par.scale
        reversals = self.df_rev.loc[cond_traj,self.par.rev_column].astype(bool)

        # Coordiante of the smoothed trajectories
        x_s = self.df_rev.loc[cond_traj, self.par.x_column+'s'] * self.par.scale
        y_s = self.df_rev.loc[cond_traj, self.par.y_column+'s'] * self.par.scale

        ### PLOT ###
        width = (np.max(x) - np.min(x)) * zoom + 0.1
        height = (np.max(y) - np.min(y)) * zoom + 0.1
        
        fig, ax = plt.subplots(figsize=(width,height))

        if plot_traj:
            
            ax.plot(x,y,color='k',alpha=0.5, label='id: '+str(trajectory))
            ax.scatter(x.to_numpy()[0],y.to_numpy()[0],s=25,c='r',linewidths=0)

        if plot_traj_point:

            ax.scatter(x,y,color='k',alpha=0.5,s=size_traj_point,linewidths=0)

        if plot_traj_s:

            ax.plot(x_s,y_s,color='g',alpha=0.5)
            ax.scatter(x_s.to_numpy()[0],y_s.to_numpy()[0],s=25,c='r',linewidths=0)

        if plot_traj_s_point:

            ax.scatter(x_s,y_s,color='g',alpha=0.5,s=10,linewidths=0)

        if plot_rev:
            ax.scatter(x[reversals],y[reversals],s=50,c='violet',linewidths=0)

        ax.set_xlabel(r'$\mu$m')
        ax.set_ylabel(r'$\mu$m')
        ax.set_title(str(trajectory))
        ax.legend(loc='upper right', fontsize=self.fontsize/1.5)


    def plot_tbr(self, min_lifetime, color, tbr_min=0, xmin=0, xmax=15, ymin=0, ymax=0.4, bin_width=10/60, save_plot=True, save_tbr_csv=True):
        """
        Plot the histogram of the time between reversals and optionally save the plot and the TBR data.

        Parameters
        ----------
        min_lifetime : float
            Minimum trajectory lifetime to be included in the histogram.
        color : str
            Color for the histogram bars.
        tbr_max_for_plot : float, optional
            Maximum time between reversals to display on the x-axis (default is 15).
        ticks_interval_tbr : float, optional
            Interval between x-axis ticks (default is 5).
        width_bin : float, optional
            Width of the histogram bins (default is 1/3).
        min_tbr : float, optional
            Minimum TBR value to be included in the histogram (default is 0).
        save_plot : bool, optional
            Whether to save the histogram plot (default is True).
        save_tbr_csv : bool, optional
            Whether to save the TBR data as a CSV file (default is True).
        """
        # Data
        cond_tbr = self.df_rev.loc[:, self.par.tbr_column] > tbr_min / self.par.tbf
        cond_length_traj = self.df_rev.loc[:, self.par.traj_length_column] >= min_lifetime
        mask = cond_tbr & cond_length_traj
        tbr_plot = self.df_rev.loc[mask, self.par.tbr_column].to_numpy() * self.par.tbf

        # Plot
        step = round((xmax-xmin) / bin_width) + 1
        bins_tbr = np.linspace(0, xmax, step)

        fig, ax = plt.subplots(figsize=(7, 7))
        # label_mean_tbr = 'Mean = ' + str(np.round(np.nanmean(tbr_plot), 1)) + ' min'
        hist, bins, __ = ax.hist(tbr_plot, 
                              bins=bins_tbr, 
                              color=color, 
                              alpha=0.7,
                              density=True, 
                              histtype='bar', 
                              ec=color, 
                              linewidth=0.5, 
                              range=[xmin, xmax],
                              label='Mean: '+str(np.round(np.mean(tbr_plot), 1)) + ' min')
        
        ax.set_xlabel("Time between reversals (min)", fontsize=self.fontsize)
        ax.set_ylabel("Density of events", fontsize=self.fontsize)

        ax.legend(loc='upper right', handlelength=1, borderpad=0, frameon=False, fontsize=self.fontsize / 1.5)

        ax.tick_params(axis='both',
                   which='major',
                   labelsize=self.fontsize / 1.5)
        ax.set_xticks(np.arange(xmin, xmax + 1, 5))
        ax.set_ylim(ymin, ymax) # Have the same y axis for all the plots

        if save_plot:
            full_path = os.path.join(self.path_output, 'plot')
            os.makedirs(full_path, exist_ok=True)
            filename_no_ext = self.filename_no_ext + '__' + self.end_filename + '__TBR'
            fig.savefig(os.path.join(full_path, filename_no_ext + '.png'), bbox_inches='tight', dpi=300)
            fig.savefig(os.path.join(full_path, filename_no_ext + '.svg'), dpi=300)

        if save_tbr_csv:
            df_tbr = pd.DataFrame({
                'x': self.df_rev.loc[:, self.par.x_column].to_numpy(),
                'y': self.df_rev.loc[:, self.par.x_column].to_numpy(),
                self.par.tbr_column: self.df_rev.loc[:, self.par.tbr_column].to_numpy() * self.par.tbf,
                self.par.traj_length_column: self.df_rev.loc[:, self.par.traj_length_column].to_numpy(),
            })
            filename = self.filename_no_ext + '__TBR__' + self.end_filename + '.csv'
            self.tool.save_df(df=df_tbr, path=self.path_output, filename=filename)


    def plot_velocity_distribution(self, max_vel=20, width_bin=1/6, save=True):
        """
        Plot the histogram of the velocities in µm/min
        
        """
        velocities = (self.df_rev.loc[:, "velocity"].to_numpy()) * self.par.scale

        # Parameters plot
        vel_max_for_plot = max_vel # in minutes
        bins_vel = np.arange(0,vel_max_for_plot+1,width_bin)

        # Plot
        fig, ax = plt.subplots(figsize=(8,6))
        label_mean_velocities = 'Mean = '+str(np.round(np.nanmean(velocities),1))+r' $\mu$m/min'
        n = ax.hist(velocities, bins=bins_vel, label=label_mean_velocities, density=True, alpha=0.4, histtype='bar', ec='black', color="royalblue")
        ax.set_xlabel(r'Velocities ($\mu$m/min)', fontsize=self.fontsize)
        ax.set_xlim(0, vel_max_for_plot)
        # ax.set_xticks(np.arange(0, vel_max_for_plot+1, width_bin*5))
        # ax.set_yticks(np.round(np.arange(0, np.max(n[0])+0.2, 0.2), 1))
        ax.tick_params(labelsize=self.fontsize)
        ax.legend(loc='best', fontsize=self.fontsize/1.5)

        if save:
            full_path = os.path.join(self.path_output, 'plot')
            os.makedirs(full_path, exist_ok=True)
            filename_no_ext = self.filename_no_ext + '__' + self.end_filename + '__VELOCITY'
            fig.savefig(os.path.join(full_path, filename_no_ext + '.png'), bbox_inches='tight', dpi=300)
            fig.savefig(os.path.join(full_path, filename_no_ext + '.svg'), dpi=300)


    def plot_nb_rev_traj_max_lifetime(self, width_bin, save):
        """
        Plot the number of reversals for the longest trajectories, i.e. trajectories with same duration than the movie
        
        """
        n_frames = (np.max(self.df_rev.loc[:,self.par.t_column]) - np.min(self.df_rev.loc[:,self.par.t_column]))
        cond_length_traj = self.df_rev.loc[:,self.par.traj_length_column] == n_frames
        df_tmp = self.df_rev.loc[cond_length_traj,(self.par.track_id_column,self.par.rev_column)]
        idx = np.unique(df_tmp.loc[:,self.par.track_id_column])
        nb_rev_array = np.zeros(len(idx))

        for count, index in enumerate(idx):

            cond_traj = df_tmp.loc[:,self.par.track_id_column] == index
            nb_rev = len(np.where(df_tmp.loc[cond_traj,self.par.rev_column])[0])
            nb_rev_array[count] = nb_rev

        # Parameters plot
        rev_max_for_plot = np.max(nb_rev_array) # in minutes
        bins_tbr = np.arange(0,rev_max_for_plot+1,width_bin)

        # Plot
        fig, ax = plt.subplots(figsize=(8,6))
        label_mean_rev = 'Mean = '+str(np.round(np.nanmean(nb_rev_array),2))+' reversals'
        n = ax.hist(nb_rev_array, bins=bins_tbr, label=label_mean_rev, density=False, alpha=0.4, histtype='bar', ec='black', color="royalblue")
        ax.set_xlabel("# Reversals", fontsize=self.fontsize)
        ax.set_ylabel("Count", fontsize=self.fontsize)
        ax.set_xlim(0, rev_max_for_plot)
        # ax.set_xticks(np.arange(0, rev_max_for_plot+1, 3))
        # ax.set_yticks(np.round(np.arange(0, np.max(n[0])+0.2, 0.2),1))
        ax.tick_params(labelsize=self.fontsize)
        ax.legend(loc='best',fontsize=self.fontsize/1.5)

        if save:
            os.makedirs(self.path_output, exist_ok=True)
            full_path = os.path.join(self.path_output, 'plot')
            filename_no_ext = self.filename_no_ext + '__' + self.end_filename + '__NB_REV_PER_TRAJ_MAX_LIFETIME'
            fig.savefig(os.path.join(full_path, filename_no_ext + '.png'), bbox_inches='tight', dpi=300)
            fig.savefig(os.path.join(full_path, filename_no_ext + '.svg'), dpi=300)


    def plot_frustration(self, frustration, reversals, reversals_memory, cumul=False, width_bins=0.2, xlim=None):
        """
        Plot the global and the frustration before a reversals
        
        """
        # Condition for all nan in thn_framessal
        cond_fru_rev = np.roll(reversals,-1)
        fru_rev = frustration[cond_fru_rev]
        if cumul:
            cond_rev_memory = reversals_memory.astype(bool)
            fru_glob = frustration[~cond_rev_memory]
        else:
            fru_glob = frustration[~cond_fru_rev]
        
        # Keep frustration only in the xlim intervall
        if xlim:
            # Conditions
            fru_glob[(fru_glob < xlim[0]) | (fru_glob > xlim[1])] = np.nan
            fru_rev[(fru_rev < xlim[0]) | (fru_rev > xlim[1])] = np.nan
            # Binning
            n_bin_glob = round((np.nanmax(fru_glob) - np.nanmin(fru_glob)) / width_bins)
            n_bin_rev = round((np.nanmax(fru_rev) - np.nanmin(fru_rev)) / width_bins)
        else:
            # Binnning
            n_bin_glob = round((np.nanmax(fru_glob) - np.nanmin(fru_glob)) / width_bins)
            n_bin_rev = round((np.nanmax(fru_rev) - np.nanmin(fru_rev)) / width_bins)

        # Remove NaN
        fru_glob = np.sort(fru_glob[~np.isnan(fru_glob)].to_numpy())
        fru_rev = np.sort(fru_rev[~np.isnan(fru_rev)].to_numpy())

        # P-values calculation
        # pvalue = stats.kstest(fru_glob, fru_rev)[1]


        # Plot
        fig, ax = plt.subplots(figsize=(8,6))
        hist_fru_glob = ax.hist(fru_glob,bins=n_bin_glob,label="Global frustration, mean="+str(round(np.mean(fru_glob),2)), density=True, alpha=0.7, histtype='bar', ec='grey',color="lightblue")
        hist_fru_rev = ax.hist(fru_rev,bins=n_bin_rev,label="Reversal frustration, mean="+str(round(np.mean(fru_rev),2)), density=True, alpha=0.7, histtype='bar', ec='grey',color="violet")
        max_glob = np.max(hist_fru_glob[0])
        max_rev = np.max(hist_fru_rev[0])
        ax.set_ylim(0,np.maximum(max_glob,max_rev)*1.5)
        ax.set_xlabel("Frustration",fontsize=self.fontsize)
        ax.set_ylabel("Density",fontsize=self.fontsize)
        if xlim:
            ax.set_xlim(xlim[0],xlim[1])
        ax.tick_params(labelsize=self.fontsize)
        ax.legend(loc='best',fontsize=self.fontsize/1.5)
        # ax.text(x=xlim[0]+np.abs(xlim[0]*0.1),y=np.maximum(max_glob,max_rev)*1.4,s="pvalue = "+"{:.1e}".format(pvalue),fontsize=fontsize/1.5)

        # Compute the Z-test between the two distribution
        mean_global = statistics.mean(fru_glob)
        sd_global = statistics.stdev(fru_glob)
        print(norm.pdf(fru_glob, mean_global, sd_global))
        mean_rev = statistics.mean(fru_rev)
        sd_rev = statistics.stdev(fru_rev)

        z_test = np.abs(mean_global - mean_rev) / np.sqrt(sd_global**2 + sd_rev**2)
        ax.text(x=xlim[0]+np.abs(xlim[0]*0.1),y=np.maximum(max_glob,max_rev)*1.4,s="Z-test = "+"{:.1e}".format(z_test),fontsize=self.fontsize/1.5)
        
        # Fit global
        fig, ax = plt.subplots(figsize=(8,6))
        hist_fru_glob = ax.hist(fru_glob,bins=n_bin_glob,label="Global frustration, mean="+str(round(np.mean(fru_glob),2)), density=True, alpha=0.7, histtype='bar', ec='grey',color="lightblue")
        ax.plot(fru_glob, norm.pdf(fru_glob, mean_global, sd_global))
        ax.set_ylim(0,np.maximum(max_glob,max_rev)*1.5)
        ax.set_xlabel("Frustration fit", fontsize=self.fontsize)
        ax.set_ylabel("Density", fontsize=self.fontsize)
        if xlim:
            ax.set_xlim(xlim[0],xlim[1])
        ax.tick_params(labelsize=self.fontsize)
        ax.legend(loc='best', fontsize=self.fontsize/1.5)

        # Fit rev
        fig, ax = plt.subplots(figsize=(8,6))
        hist_fru_rev = ax.hist(fru_rev,bins=n_bin_rev,label="Reversal frustration, mean="+str(round(np.mean(fru_rev),2)), density=True, alpha=0.7, histtype='bar', ec='grey',color="violet")
        ax.plot(fru_rev, norm.pdf(fru_rev, mean_rev, sd_rev))
        ax.set_ylim(0,np.maximum(max_glob,max_rev)*1.5)
        ax.set_xlabel("Frustration fit",fontsize=self.fontsize)
        ax.set_ylabel("Density",fontsize=self.fontsize)
        if xlim:
            ax.set_xlim(xlim[0],xlim[1])
        ax.tick_params(labelsize=self.fontsize)
        ax.legend(loc='best',fontsize=self.fontsize/1.5)
