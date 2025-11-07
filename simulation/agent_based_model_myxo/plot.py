import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Circle
from tqdm import tqdm
import gc
import os
# Définir la police sans-serif pour les graphiques
plt.rcParams['font.sans-serif'] = 'Arial'
# Utiliser la police sans-serif pour les graphiques
plt.rcParams['font.family'] = 'sans-serif'

import utils as tl


class Plot:
    """
    Class to handle plotting functionalities for a bacterial simulation.
    It includes methods for visualizing bacterial positions and directions, 
    along with optional visual elements such as focal adhesion points or an EPS grid.

    Attributes:
    - inst_par: Instance containing simulation parameters.
    - inst_gen: Instance handling data generation.
    - inst_dir: Instance for handling direction-related data.
    - inst_rev: Instance related to reversals.
    - inst_ecm: Instance managing extracellular polymeric substance (EPS) grid.
    - inst_move: Instance for managing bacterial movements.
    - inst_kym: Instance for generating kymographs.
    - inst_nei: Instance of the class managing neighbor relationships and distances between bacteria.
    - sample: Path to the directory for saving outputs.
    """
    def __init__(self, inst_par, inst_gen, inst_dir, inst_rev, inst_ecm, inst_move, inst_kym, inst_nei, inst_sig, sample):
        # Store the provided instances.
        self.par = inst_par
        self.gen = inst_gen
        self.dir = inst_dir
        self.rev = inst_rev
        self.ecm = inst_ecm
        self.move = inst_move
        self.kym = inst_kym
        self.nei = inst_nei
        self.sig = inst_sig
        self.sample = sample

        # Ensure the output directory exists.
        tl.initialize_directory_or_file(self.sample + '/')

        # Simulation parameters for plotting.
        self.n_bact = self.par.n_bact  # Number of bacteria.
        self.n_nodes = self.par.n_nodes  # Number of nodes per bacterium.
        self.d_n = self.par.d_n  # Distance between nodes.
        self.size = self.par.space_size  # Size of the simulation space.
        self.border = self.ecm.edges_width  # Border width for EPS visualization.
        self.r = self.ecm.r  # Length of pili in pixels.

        # Determine the point size for plotting based on simulation parameters.
        self.figsize = (32, 32)  # Simulation figure size for plotting.
        self.point_size = self.par.param_point_size * 100 / self.size * self.figsize[0] ** 2 * self.par.dpi_simu / 100

        # Assign random colors to bacteria for visual distinction.
        self.random_color = np.tile(np.random.rand(self.n_bact), (self.n_nodes, 1))

        # Base flashy colors
        self.flashy_colors = [
            "#FF0000", "#00FFFF", "#FFFF00", "#FF00FF", "#00FF00", "#FFA500",
            "#00CED1", "#FF1493", "#ADFF2F", "#1E90FF", "#FF4500", "#7CFC00",
            "#FF69B4", "#00FA9A", "#FFFFE0", "#FFD700"
        ]


    def plot_red_marker_top_right(self, fig, size=0.05):
        """
        Draw a solid red circular marker in the top-right corner of the figure,
        slightly inset so it does not touch the borders.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure object to which the marker will be added.

        size : float, optional
            Radius of the marker in normalized figure coordinates (0–1).
            Default is 0.02 (2% of the figure size).
        """
        # Offset to avoid touching the borders (20% extra margin)
        offset = size * 1.2

        # Top-right position in normalized figure coordinates
        x, y = offset, 1 - offset

        # Create and add the circle to the figure
        circle = Circle(
            (x, y),
            radius=size,
            transform=fig.transFigure,  # Use figure coordinates (0 to 1)
            facecolor='k',
            linewidth=0,        # No border width
            edgecolor=None,
            alpha=1,
            zorder=10
        )

        fig.patches.append(circle)


    def plotty(self, path, transition_stop=False):
        """
        Create and save a scatter plot visualizing bacterial positions.

        Parameters:
        - path: String, the file path where the plot will be saved.
        """
        # Initialize the figure and axis for plotting.
        fig, ax = plt.subplots(figsize=self.figsize)
        # ax = fig.add_subplot(111)

        # Map bacterial node directions to colors using their angles.
        angle_color = np.tile(np.arctan2(self.dir.nodes_direction[1, 0, :self.gen.first_index_prey_bact], 
                                         self.dir.nodes_direction[0, 0, :self.gen.first_index_prey_bact]), 
                              (self.par.n_nodes, 1))

        # Plot bacterial nodes as scatter points, colored by their direction angles.
        plt.scatter(self.gen.data[0, :, :self.gen.first_index_prey_bact], 
                    self.gen.data[1, :, :self.gen.first_index_prey_bact], 
                    s=self.point_size, 
                    linewidths=0.15, 
                    c=angle_color, 
                    edgecolor='k', 
                    cmap='hsv', 
                    vmin=-np.pi, 
                    vmax=np.pi, 
                    zorder=1, 
                    alpha=0.35)
        if transition_stop:
            self.plot_red_marker_top_right(fig=fig)
        
        if self.par.n_bact_prey > 0:
            cond_plot_prey = self.nei.cond_alive_bacteria[:, self.gen.first_index_prey_bact:]
            plt.scatter(self.gen.data[0, :, self.gen.first_index_prey_bact:][cond_plot_prey], 
                        self.gen.data[1, :, self.gen.first_index_prey_bact:][cond_plot_prey], 
                        s=self.point_size, 
                        linewidths=0., 
                        c='k', 
                        edgecolor='k', 
                        alpha=1)

        # Optionally plot focal adhesion points if enabled in parameters.
        if self.par.plot_position_focal_adhesion_point:
            plt.scatter(self.gen.data[0, self.move.position_focal_adhesion_point[0, :, :]], 
                        self.gen.data[1, self.move.position_focal_adhesion_point[1, :, :]], 
                        s=self.point_size / 5, 
                        c='k', 
                        zorder=1, 
                        alpha=0.35)

        # Optionally plot the EPS grid if enabled in parameters.
        if self.par.plot_ecm_grid:
            if self.par.n_bact_prey > 0:
                nb_layers = int((self.ecm.eps_grid_blur.shape[0] - self.ecm.prey_grid.shape[0]) / 2)
                if nb_layers > 0:
                    ecm_grid = np.pad(self.ecm.prey_grid, pad_width=nb_layers, mode='constant', constant_values=0).copy() + np.rot90(self.ecm.eps_grid_blur.copy())
                elif nb_layers < 0:
                    ecm_grid = self.ecm.prey_grid.copy() + np.rot90(np.pad(self.ecm.eps_grid_blur, pad_width=-nb_layers, mode='constant', constant_values=0).copy())
                else:
                    ecm_grid = self.ecm.prey_grid.copy() + np.rot90(self.ecm.eps_grid_blur.copy())

            #     ecm_grid = np.rot90(self.ecm.eps_grid_blur) + self.ecm.prey_grid  # Rotation of the eps map
            else:
                ecm_grid = np.rot90(self.ecm.eps_grid_blur)

            plt.imshow(ecm_grid,
                       extent=(-self.border, self.size + self.border, -self.border, self.size + self.border), 
                       cmap='Reds', 
                       vmin=0, 
                       vmax=self.par.max_eps_value)

        # Set aspect ratio and axis limits for the plot.
        # ax.set_aspect('equal', adjustable='box')
        if not self.par.plot_ecm_grid:
            ax.set_xlim(0, self.size)
            ax.set_ylim(0, self.size)
        ax.axis('off')  # Remove axis lines and labels for a cleaner plot.

        # Adjust subplot layout for tight bounds.
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
        fig.subplots_adjust(wspace=0, hspace=0)

        # Save the plot to the specified path with a resolution of 100 dpi.
        fig.savefig(path, dpi=self.par.dpi_simu)

        # Close the figure and collect garbage to free memory.
        plt.close()
        gc.collect()


    def plotty_rippling_swarming(self, path, t):
        """
        Create and save a scatter plot visualizing the spatial organization 
        of bacteria during rippling and swarming phases.

        Parameters:
        - path: String, the file path where the plot will be saved.
        - t: Current time step in the simulation.
        """
        # Initialize conditions for rippling and swarming based on simulation time.
        if t * self.par.dt < self.par.time_rippling_swarming_colored:
            # For early simulation, copy alignment and EPS conditions to track patterns.
            self.cond_rippling = self.gen.cond_space_alignment.copy()
            self.cond_swarming = self.gen.cond_space_eps.copy()

        # Initialize the figure and axis for plotting.
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111)

        # Scatter plot for bacteria in the rippling phase, colored by a predefined color.
        ax.scatter(self.gen.data[0, :, self.cond_rippling], 
                   self.gen.data[1, :, self.cond_rippling], 
                   s=self.point_size, 
                   linewidths=0.5, 
                   c=self.par.color_rippling, 
                   edgecolor='grey', 
                   zorder=1, 
                   alpha=self.par.alpha)

        # Scatter plot for bacteria in the swarming phase, with a different color.
        ax.scatter(self.gen.data[0, :, self.cond_swarming], 
                   self.gen.data[1, :, self.cond_swarming], 
                   s=self.point_size, 
                   linewidths=0.5, 
                   c=self.par.color_swarming, 
                   edgecolor='grey', 
                   zorder=2, 
                   alpha=self.par.alpha)

        # Optionally plot focal adhesion points if enabled in parameters.
        if self.par.plot_position_focal_adhesion_point:
            ax.scatter(self.gen.data[0, self.move.position_focal_adhesion_point[0, :, :]], 
                       self.gen.data[1, self.move.position_focal_adhesion_point[1, :, :]], 
                       s=self.point_size / 5, 
                       c='k', 
                       zorder=1, 
                       alpha=0.35)

        # Optionally plot the EPS grid if enabled in parameters.
        if self.par.plot_ecm_grid:
            cmap = plt.cm.get_cmap('Greys', int(self.par.max_eps_value))
            ax.imshow(np.rot90(self.ecm.eps_grid), 
                      extent=(-self.border, self.size + self.border, -self.border, self.size + self.border), 
                      cmap=cmap, 
                      vmin=0, 
                      vmax=self.par.max_eps_value)

        # Set aspect ratio and axis limits for the plot.
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.axis('off')  # Remove axis for a cleaner visualization.

        # Adjust subplot layout to ensure no padding around the plot.
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
        fig.subplots_adjust(wspace=0, hspace=0)

        # Save the plot to the specified path and clean up memory.
        fig.savefig(path, dpi=self.par.dpi_simu)
        plt.close()
        gc.collect()


    def plotty_reversing_and_non_reversing(self, path):
        """
        Create and save a scatter plot visualizing bacteria classified as 
        reversing or non-reversing.

        Parameters:
        - path: String, the file path where the plot will be saved.
        """
        # Use a dark background for improved visibility.
        plt.style.use('dark_background')

        # Initialize the figure and axis for plotting.
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111)

        # Scatter plot for reversing bacteria, colored by a predefined color.
        plt.scatter(self.gen.data[0, :, self.rev.cond_reversing], 
                    self.gen.data[1, :, self.rev.cond_reversing], 
                    s=self.point_size, 
                    linewidths=0.5, 
                    c=self.par.color_reversing, 
                    edgecolor='grey', 
                    zorder=1, 
                    alpha=self.par.alpha)

        # Scatter plot for non-reversing bacteria, with a different color.
        plt.scatter(self.gen.data[0, :, ~self.rev.cond_reversing], 
                    self.gen.data[1, :, ~self.rev.cond_reversing], 
                    s=self.point_size, 
                    linewidths=0.5, 
                    c=self.par.color_non_reversing, 
                    edgecolor='grey', 
                    zorder=2, 
                    alpha=self.par.alpha)

        # Optionally plot focal adhesion points if enabled in parameters.
        if self.par.plot_position_focal_adhesion_point:
            plt.scatter(self.gen.data[0, self.move.position_focal_adhesion_point[0, :, :]], 
                        self.gen.data[1, self.move.position_focal_adhesion_point[1, :, :]], 
                        s=self.point_size / 5, 
                        c='k', 
                        zorder=1, 
                        alpha=0.35)

        # Optionally plot the EPS grid if enabled in parameters.
        if self.par.plot_ecm_grid:
            cmap = plt.cm.get_cmap('Greys', int(self.par.max_eps_value))
            plt.imshow(np.rot90(self.ecm.eps_grid), 
                       extent=(-self.border, self.size + self.border, -self.border, self.size + self.border), 
                       cmap=cmap, 
                       vmin=0, 
                       vmax=self.par.max_eps_value)

        # Set aspect ratio and axis limits for the plot.
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.axis('off')  # Remove axis for a cleaner visualization.

        # Adjust subplot layout to ensure no padding around the plot.
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
        fig.subplots_adjust(wspace=0, hspace=0)

        # Save the plot to the specified path and clean up memory.
        fig.savefig(path, dpi=self.par.dpi_simu)
        plt.close()
        gc.collect()


    def plotty_colored_nb_neighbors(self, path):
        """
        Plot all bacterial nodes and color them according to the number of neighbors per bacterium,
        using a discrete flashy color map. Save the plot to the specified path.

        Parameters
        ----------
        path : str
            File path where the figure will be saved.
        """
        # Number of classes (discrete levels)
        max_neighbors = self.par.max_neighbours_signal_density
        n_classes = int(max_neighbors) + 1

        # Extend / cycle flashy_colors as needed
        colors = (self.flashy_colors * ((n_classes // len(self.flashy_colors)) + 1))[:n_classes]
        cmap = ListedColormap(colors)
        bounds = np.arange(-0.5, n_classes + 0.5, 1)
        norm = BoundaryNorm(boundaries=bounds, ncolors=cmap.N)

        # Repeat neighbor counts in (n_nodes, n_bact) then flatten column-wise (Fortran order)
        values = np.repeat(self.sig.nb_neighbors[np.newaxis, :], self.par.n_nodes, axis=0).flatten(order='F')

        # Get node coordinates with same order (n_nodes, n_bact) → flatten in Fortran order
        x = self.gen.data[0, :, :self.gen.first_index_prey_bact].flatten(order='F')
        y = self.gen.data[1, :, :self.gen.first_index_prey_bact].flatten(order='F')

        # Create figure and axes
        fig, ax = plt.subplots(figsize=self.figsize)

        # Scatter plot of nodes colored by neighbor count
        sc = ax.scatter(
            x, y,
            c=values,
            cmap=cmap,
            norm=norm,
            s=self.point_size,
            linewidths=0.,
            edgecolor='k',
            alpha=1
        )
        # Draw vectors at heads (node index 0)
        x_head = self.gen.data[0, 0, :self.gen.first_index_prey_bact]
        y_head = self.gen.data[1, 0, :self.gen.first_index_prey_bact]

        # Direction vector from first node
        head_dir = self.dir.nodes_direction[:, 0, :self.gen.first_index_prey_bact]  # shape (2, n_bact)
        norm_dir = head_dir / np.linalg.norm(head_dir, axis=0, keepdims=True)
        scale = self.par.width / 2

        ax.quiver(
            x_head, y_head,
            norm_dir[0] * scale,
            norm_dir[1] * scale,
            angles='xy',
            scale_units='xy',
            scale=1,
            width=0.002,           # très fin
            headwidth=2,           # plus petit
            headlength=3,          # plus petit
            headaxislength=2.5,    # plus petit
            color='k',
            alpha=0.8,
            zorder=2
        )

        # Discrete colorbar
        cbar = plt.colorbar(sc, ax=ax, ticks=np.arange(n_classes), boundaries=bounds)
        cbar.ax.set_yticklabels([str(i) for i in range(n_classes)])
        cbar.ax.tick_params(labelsize=50)

        # Styling
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.set_aspect('equal')
        # ax.axis('off')
        # # Adjust subplot layout for tight bounds.
        # fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
        # fig.subplots_adjust(wspace=0, hspace=0)

        # fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
        fig.savefig(path, dpi=self.par.dpi_simu)
        plt.close()
        gc.collect()


    def plotty_rev(self, x, y, path, point_size, fig, ax):
        """
        Plot reversal events on a separate frame from the cells.

        Parameters:
        - x: List of x-coordinates for reversal events.
        - y: List of y-coordinates for reversal events.
        - path: String, the file path where the plot will be saved.
        - point_size: Size of the points in the scatter plot.
        - fig: Matplotlib figure object to use for plotting.
        - ax: Matplotlib axis object to use for plotting.
        """
        # Flatten the coordinates for all reversal events.
        x_rev = np.concatenate(x)
        y_rev = np.concatenate(y)

        # Scatter plot of reversal events, using low-opacity blue points.
        ax.scatter(x_rev, y_rev, s=point_size, c="blue", zorder=1, alpha=0.05)

        # Set axis aspect ratio and limits for consistent scaling.
        ax.set_aspect('equal', adjustable='box')
        plt.xlim(-self.border, self.size + self.border)
        plt.ylim(-self.border, self.size + self.border)

        # Save the plot to the specified path.
        fig.savefig(path, dpi=self.par.dpi_simu)
        plt.close()
        gc.collect()  # Perform garbage collection to free memory.


    def plotty_eps(self, path):
        """
        Visualize the EPS (Extracellular Polymeric Substances) map.

        Parameters:
        - path: String, the file path where the EPS map will be saved.
        """
        # Initialize the figure and axis for the EPS plot.
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111)

        # Configure the colormap for the EPS grid visualization.
        cmap = plt.cm.get_cmap('RdPu', int(2 * (self.par.max_eps_value + 1)))

        # Plot the EPS grid using a rotated image and predefined color map.
        plt.imshow(np.rot90(self.ecm.eps_grid),
                   extent=(-0.1 * self.r, self.size + 0.1 * self.r, -0.1 * self.r, self.size + 0.1 * self.r),
                   cmap=cmap,
                   vmin=0,
                   vmax=self.par.max_eps_value / 2)

        # Set aspect ratio and axis limits for consistent scaling.
        ax.set_aspect('equal', adjustable='box')
        plt.xlim(-0.1 * self.r, self.size + 0.1 * self.r)
        plt.ylim(-0.1 * self.r, self.size + 0.1 * self.r)

        # Save the plot to the specified path with tight bounding box to minimize padding.
        fig.savefig(path, bbox_inches='tight', dpi=self.par.dpi_simu)
        plt.close()
        gc.collect()  # Perform garbage collection to free memory.


    def compute_tbr(self, n_sample=1000):
        """
        Compute and plot the mean Time Between Reversals (TBR) as a function of the reversal rate.

        Parameters:
        - n_sample: Number of samples to simulate (default: 1000).
        """
        # Initialize lists to store TBR values, mean, and variance.
        tbr_list = []
        mean_tbr = []
        var_tbr = []

        # Initialize the clock for tracking time since the last reversal.
        clock_tbr = np.zeros(n_sample)

        # Define a range of reversal rates to evaluate.
        reversal_rate = np.arange(0.1, 30.1, 1)

        # Loop through each reversal rate.
        for rr in tqdm(reversal_rate):
            # Simulate a fixed amount of time for each reversal rate.
            for time in range(int(50 / self.par.dt)):
                clock_tbr += self.par.dt  # Increment the clock by the simulation timestep.

                # Calculate the probability of a reversal occurring at each timestep.
                prob = np.ones(n_sample) * (1 - np.exp(-rr))

                # Determine which samples experience a reversal using a Bernoulli distribution.
                cond_rev = np.random.binomial(1, prob * self.par.dt).astype("bool")

                # Append the TBR values for the samples that reversed and reset their clocks.
                tbr_list.append(clock_tbr[cond_rev])
                clock_tbr[cond_rev] = 0

            # Compute the mean and variance of the TBR values across all samples.
            mean_tbr.append(np.mean(np.concatenate((tbr_list))))
            var_tbr.append(np.std(np.concatenate((tbr_list))))

        # Plot the results.
        fontsize = 25
        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot the mean TBR as a function of the reversal rate.
        ax.plot(mean_tbr, reversal_rate)

        # Add axis labels and error bars for variance.
        ax.set_xlabel("Time between reversals (min)", fontsize=self.par.fontsize)
        ax.set_ylabel("Reversal rate", fontsize=self.par.fontsize)
        ax.errorbar(mean_tbr, reversal_rate, xerr=var_tbr, fmt='.k')

        # Configure axis ticks for readability.
        plt.xticks(fontsize=self.par.fontsize)
        plt.yticks(fontsize=self.par.fontsize)

        # Display the plot.
        plt.show()


    def tbr(self, sample, T, duration=50, xmin=0, xmax=15):
        """
        Plot the distribution of time between reversals (TBR) for all events.

        Parameters:
        - sample: String, the sample name or directory for saving output.
        - T: Current simulation time.
        - duration: Time window (in minutes) to consider for the TBR data (default: 50).
        - xmin: Minimum value for the x-axis in the histogram (default: 0).
        - xmax: Maximum value for the x-axis in the histogram (default: 15).
        """
        # Determine the starting index based on the time window.
        start = int(np.maximum(T - duration, 0) / self.par.dt)

        # Extract TBR values and their positions from the specified time window.
        tbr_plot = np.concatenate(self.rev.tbr_list[start:]) # in minutes
        tbr_position_x = np.concatenate(self.rev.tbr_position_x_list[start:])
        tbr_position_y = np.concatenate(self.rev.tbr_position_y_list[start:])

        # Save TBR data and positions to a CSV file.
        column_names = ['tbr', 'x', 'y']
        data = np.column_stack((tbr_plot, tbr_position_x, tbr_position_y))
        np.savetxt(sample + '/tbr.csv', data, delimiter=',', header=','.join(column_names), comments='')

        # Create bins for the histogram based on the given range and save frequency.
        step = round((xmax - xmin) / self.par.save_freq_tbr) + 1
        bins_tbr = np.linspace(xmin, xmax, step)

        if self.par.generation_type == "gen_bact_rippling_swarming":
            color = 'k'
        elif self.par.alignment_type == "global_alignment":
            color = self.par.color_rippling
        elif self.par.eps_follower_type == "igoshin_eps_road_follower":
            color = self.par.color_swarming
        else:
            color = 'k'

        # Plot the histogram for the TBR distribution.
        fig, ax = plt.subplots(figsize=self.par.figsize)
        hist, bins, __ = ax.hist(tbr_plot, 
                              bins=bins_tbr, 
                              color=color, 
                              alpha = self.par.alpha,
                              density=True, 
                              histtype='bar', 
                              ec=color, 
                              linewidth=0.5, 
                              range=[xmin, xmax],
                              label='Mean: '+str(np.round(np.mean(tbr_plot), 1)) + ' min')
        
        ax.set_xlabel("Time between reversals (min)", fontsize=self.par.fontsize)
        ax.set_ylabel("Density of events", fontsize=self.par.fontsize)

        # Configure plot limits and ticks.
        ax.legend(loc='upper right', handlelength=1, borderpad=0, frameon=False, fontsize=self.par.fontsize_ticks)
        ax.tick_params(axis='both',
                    which='major',
                    labelsize=self.par.fontsize_ticks)
        ax.set_xticks(np.arange(xmin, xmax+1, step=5))

        path = self.sample + '/'
        tl.initialize_directory_or_file(path)
        fig.savefig(path + 'tbr_distribution.png', bbox_inches='tight', dpi=self.par.dpi)
        fig.savefig(path + 'tbr_distribution.svg', dpi=self.par.dpi)


    def tbr_cond_space(self, sample, T, duration=50, xmin=0, xmax=15):
        """
        Plot the distribution of time between reversals (TBR), 
        separating rippling and swarming regions.

        Parameters:
        - sample: String, the sample name or directory for saving output.
        - T: Current simulation time.
        - duration: Time window (in minutes) to consider for the TBR data (default: 50).
        - xmin: Minimum value for the x-axis in the histogram (default: 0).
        - xmax: Maximum value for the x-axis in the histogram (default: 15).
        """
        # Determine the starting index based on the time window.
        start = int(np.maximum(T - duration, 0) / self.par.dt)

        # Extract TBR values and x-coordinates from the specified time window.
        tbr_plot = np.concatenate(self.rev.tbr_list[start:]) # in minutes
        tbr_position_x = np.concatenate(self.rev.tbr_position_x_list[start:])

        # Define a condition for rippling based on the x-coordinate range.
        cond_rippling = (tbr_position_x > self.par.interval_rippling_space[0] * self.par.space_size) & \
                        (tbr_position_x < self.par.interval_rippling_space[1] * self.par.space_size)

        # Create bins for the histogram based on the given range and save frequency.
        step = round((xmax - xmin) / self.par.save_freq_tbr) + 1
        bins_tbr = np.linspace(xmin, xmax, step)

        # Plot the histogram for the TBR distribution of the rippling.
        fig1, ax1 = plt.subplots(figsize=self.par.figsize)
        hist, bins, __ = ax1.hist(tbr_plot[cond_rippling], 
                                  bins=bins_tbr, 
                                  color=self.par.color_rippling, 
                                  alpha = self.par.alpha,
                                  density=True, 
                                  histtype='bar', 
                                  ec=self.par.color_rippling, 
                                  linewidth=0.5, 
                                  range=[xmin, xmax],
                                  label='Mean: '+str(np.round(np.mean(tbr_plot[cond_rippling]), 1)) + ' min')
        
        ax1.set_xlabel("Time between reversals (min)", fontsize=self.par.fontsize)
        ax1.set_ylabel("Density of events", fontsize=self.par.fontsize)

        # Configure plot limits and ticks.
        ax1.legend(loc='upper right', handlelength=1, borderpad=0, frameon=False, fontsize=self.par.fontsize_ticks)
        ax1.tick_params(axis='both',
                    which='major',
                    labelsize=self.par.fontsize_ticks)
        ax1.set_xticks(np.arange(xmin, xmax+1, step=5))

        # Plot the histogram for the TBR distribution of the swarming.
        fig2, ax2 = plt.subplots(figsize=self.par.figsize)
        hist, bins, __ = ax2.hist(tbr_plot[~cond_rippling], 
                                  bins=bins_tbr, 
                                  color=self.par.color_swarming, 
                                  alpha = self.par.alpha,
                                  density=True, 
                                  histtype='bar', 
                                  ec=self.par.color_swarming, 
                                  linewidth=0.5, 
                                  range=[xmin, xmax],
                                  label='Mean: '+str(np.round(np.mean(tbr_plot[~cond_rippling]), 1)) + ' min')
        
        ax2.set_xlabel("Time between reversals (min)", fontsize=self.par.fontsize)
        ax2.set_ylabel("Density of events", fontsize=self.par.fontsize)

        # Configure plot limits and ticks.
        ax2.legend(loc='upper right', handlelength=1, borderpad=0, frameon=False, fontsize=self.par.fontsize_ticks)
        ax2.tick_params(axis='both',
                    which='major',
                    labelsize=self.par.fontsize_ticks)
        ax2.set_xticks(np.arange(xmin, xmax+1, step=5))

        path = self.sample + '/'
        tl.initialize_directory_or_file(path)
        fig1.savefig(path + 'tbr_distribution_rippling.png', bbox_inches='tight', dpi=self.par.dpi)
        fig2.savefig(path + 'tbr_distribution_swarming.png', bbox_inches='tight', dpi=self.par.dpi)
        fig1.savefig(path + 'tbr_distribution_rippling.svg', dpi=self.par.dpi)
        fig2.savefig(path + 'tbr_distribution_swarming.svg', dpi=self.par.dpi)


    def velocity(self, velocity_list, velocity_max, width_bin):
        """
        Plot the velocity distribution.

        Parameters:
        - velocity_list: List of velocities from simulation data.
        - velocity_max: Maximum velocity value for binning.
        - width_bin: Width of each bin in the histogram.

        Outputs:
        - A histogram representing the density of velocities.
        - Saves the plot to 'velocities_distribution.png'.
        """
        vel = np.concatenate((velocity_list))  # Flatten the velocity list.
        vel = vel[vel <= velocity_max]  # Exclude velocities above the specified maximum.
        vel = vel # in µm / minute
        bins = np.arange(0, velocity_max + 1, width_bin)  # Define bins for histogram.

        if self.par.generation_type == "gen_bact_rippling_swarming":
            color = 'k'
        elif self.par.alignment_type == "global_alignment":
            color = self.par.color_rippling
        elif self.par.eps_follower_type == "igoshin_eps_road_follower":
            color = self.par.color_swarming
        else:
            color = 'k'

        fig, ax = plt.subplots(figsize=self.par.figsize)
        hist, bins, __ = ax.hist(
            vel, bins=bins, color=color, density=True, alpha=0.4, histtype='bar', ec='black'
        )

        # Plot configuration
        ax.set_xlabel(r"Velocity ($\mu$m/min)", fontsize=self.par.fontsize)
        ax.set_ylabel("Density", fontsize=self.par.fontsize)
        plt.text(
            velocity_max - 0.5, max(hist) - 0.02,
            'Mean = ' + str(format(np.mean(vel), '.3g')) + ' min',
            ha='right', va='center', fontsize=self.par.fontsize_ticks, 
            bbox=dict(facecolor='none', edgecolor='k')
        )
        ax.set_xlim(0, velocity_max)
        plt.xticks(fontsize=self.par.fontsize)
        plt.yticks(fontsize=self.par.fontsize)

        # Ensure output directory exists
        path = self.sample + '/'
        tl.initialize_directory_or_file(path)

        fig.savefig(path + 'velocities_distribution.png', bbox_inches='tight', dpi=self.par.dpi)
        fig.savefig(path + 'velocities_distribution.svg', dpi=self.par.dpi)


    def frustration(self, T, frustration, all_frustration=True, bact_id=0, width_bins=0.1):
        """
        Plot frustration data across the movie or for a specific bacterium over time.

        Parameters:
        - T: Total simulation time.
        - frustration: Array of frustration values.
        - all_frustration: Boolean flag to plot global or single-bacterium frustration.
        - bact_id: ID of the bacterium (if single-bacterium plot is chosen).
        - width_bins: Width of bins for the histogram.

        Outputs:
        - Histogram of all frustration values or time-series plot of a specific bacterium's frustration.
        """
        fig, ax = plt.subplots(figsize=self.par.figsize)

        if all_frustration:
            # Plot a histogram of all frustration values
            min_fru, max_fru = np.min(frustration), np.max(frustration)
            bins_fru = round((max_fru - min_fru) / width_bins)
            mean_fru = np.mean(frustration)
            ax.hist(
                frustration, bins=bins_fru,
                label='mean = ' + str(round(mean_fru, 3)),
                density=True, alpha=0.7, histtype='bar', ec='grey', color='lightblue'
            )
            ax.set_xlim(min_fru, max_fru)
            ax.set_xlabel('Frustration', fontsize=self.par.fontsize)
            ax.tick_params(labelsize=self.par.fontsize_ticks)
            ax.legend(loc='best', fontsize=self.par.fontsize_ticks)
        else:
            # Plot frustration over time for a specific bacterium
            tmp = np.reshape(frustration, (int(T / self.par.dt), self.par.n_bact))
            mean_cumul = tmp[:, bact_id]
            start = int(self.par.time_memory / self.par.dt)  # Account for initial memory duration.
            plt.plot(
                np.arange(start * self.par.dt, T, self.par.dt),
                mean_cumul[start:]
            )

        plt.show()


    def reversal_functions(self, signal, refractory_period, reversal_rate):
        """
        Plot reversal dynamics including frz activity, refractory period, and reversal rate.

        Parameters:
        - sample: Identifier for the sample/simulation.
        - signal: Signal array used for reversal dynamics.
        - refractory_period: Array of refractory periods.
        - reversal_rate: Array of reversal rates.

        Outputs:
        - Plot showing reversal dynamics.
        - Saves the plot to 'reversal_functions_plot.png'.
        """
        fig, ax1 = plt.subplots(figsize=self.par.figsize)
        ax2 = plt.twinx()

        # Plot refractory period on the left y-axis
        ax1.plot(
            signal, refractory_period,
            label="RP", alpha=0.8, color="k"
        )

        # Plot reversal rate on the right y-axis
        ax2.plot(
            signal, reversal_rate,
            label="RR", alpha=0.8, color="limegreen"
        )

        ax1.set_xlabel('Signal', fontsize=self.par.fontsize)
        ax1.set_ylabel('Refractory period (min)', fontsize=self.par.fontsize)
        ax2.set_ylabel(r'Reversal rate (min$^{-1}$)', fontsize=self.par.fontsize)
        ax1.tick_params(labelsize=self.par.fontsize_ticks)
        ax2.tick_params(labelsize=self.par.fontsize_ticks)

        # Configure plot axes and titles
        ax1.set_xlim(0, 2 * self.par.s1)
        ax1.set_ylim(0, self.par.rp_max + 1)
        ax2.set_ylim(0, self.par.r_max + 1)
        ax1.legend(loc='upper right', fontsize=self.par.fontsize_ticks)
        ax2.legend(loc='upper left', fontsize=self.par.fontsize_ticks)

        # Save the plot
        path = self.sample + '/reversal_functions_plot.png'
        fig.savefig(path, bbox_inches='tight', dpi=self.par.dpi)


    def kymograph_density(self, T, start):
        """
        Plot the kymograph density to visualize spatial-temporal dynamics.

        Parameters:
        - T: Total simulation time.
        - start: Start time for the kymograph.

        Outputs:
        - Kymograph density plot.
        - Saves the plot to 'kymo_plot.png'.
        """
        path = self.sample + '/kymo_plot.png'
        fig = plt.figure(figsize=self.par.figsize)
        ax = fig.add_subplot(111)

        # Display kymograph density data
        plt.imshow(
            self.kym.density_kymo,
            extent=(0, self.size, T, start),
            cmap='Greys'
        )
        ax.set_aspect('equal', adjustable='box')

        # Save the plot
        fig.savefig(path, bbox_inches='tight', dpi=self.par.dpi)
