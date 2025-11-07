# %%
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from parameters import Parameters
from matrices import Matrix
from tools import Tools

class TypeTypeError(Exception):
    pass


class Plot:

    def __init__(self):
        
        self.par = Parameters()
        self.mat = Matrix()
        self.tool = Tools()

    
    def forceAspect(self, ax, aspect):
        im = ax.get_images()
        extent =  im[0].get_extent()
        ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


    def plot_eigenvalues_1d(self, 
                            data_array, 
                            path_save,
                            filename,
                            x_label,
                            y_label):
        """
        Plot signal in function of the eigenvalues
        
        """
        xmin = data_array[0, 1]
        xmax = data_array[-1, 1]
        eps = 10e-8
        cond_instabilities = data_array[:, 0] > eps

        fig, ax = plt.subplots(figsize=self.par.figsize)
        # ax = fig.add_subplot(111)
        ax.plot(data_array[:, 1], data_array[:, 0], color='k', linewidth=2, zorder=0)
        ax.scatter(data_array[~cond_instabilities, 1], data_array[~cond_instabilities, 0], c='b', marker='v', s=25, label='values $\leq 0$', zorder=1, alpha=0.5)
        ax.scatter(data_array[cond_instabilities, 1], data_array[cond_instabilities, 0], c='r', marker='^', s=25, label='values $> 0$', zorder=1, alpha=0.5)
        ax.tick_params(labelsize=self.par.fontsize/1.5)
        ax.set_xlabel(x_label, fontsize=self.par.fontsize)
        ax.set_ylabel(y_label, fontsize=self.par.fontsize)
        ax.set_xlim(0, xmax)
        ax.legend(fontsize=self.par.fontsize/1.5, markerscale=4)
        plt.show()

        self.tool.initialize_directory_or_file(path=path_save)
        fig.savefig(path_save+filename+".png", bbox_inches="tight", dpi=100)
        fig.savefig(path_save+filename+".svg", bbox_inches="tight", dpi=100)


    def plot_eigenvalues_2d(self,
                            data_array,
                            path_save,
                            filename,
                            x_label,
                            y_label,
                            loc_or_dir,
                            linewidth=3,
                            plot_curve=False):
        """
        Plot the map of the eigenvalues
        
        """
        cmap = plt.get_cmap('plasma_r')
        xmin = data_array[1][0]
        xmax = data_array[1][-1]
        ymin = data_array[2][0]
        ymax = data_array[2][-1]
        lambda_max = np.nanmax(data_array[0])
        eigen_map_plot = data_array[0].copy()
        eigen_map_plot[eigen_map_plot < 10e-8] = np.nan

        fig, ax = plt.subplots(figsize=self.par.figsize)
        # ax = fig.add_subplot(111)
        im = plt.imshow(eigen_map_plot.T, extent=[xmin,xmax,ymin,ymax], origin='lower', cmap=cmap, aspect='auto', vmin=0, vmax=lambda_max)
        
        if plot_curve:
            if loc_or_dir == 'loc':
                ax.plot(self.par.S_array, self.mat.C_R_local(self.par.S_array), color='k',  linewidth=linewidth, linestyle='dotted')
                ax.plot(self.par.S_array, self.mat.C_P_local(self.par.S_array), color='k',  linewidth=linewidth, linestyle='dashdot')
            elif loc_or_dir == 'dir':
                ax.plot(self.par.S_array, self.mat.C_R_directional(self.par.S_array), color='k',  linewidth=linewidth, linestyle='dotted')
                ax.plot(self.par.S_array, self.mat.C_P_directional(self.par.S_array), color='k',  linewidth=linewidth, linestyle='dashdot')
        
        ax.tick_params(labelsize=self.par.fontsize/1.5)
        ax.set_xlabel(x_label, fontsize=self.par.fontsize)
        ax.set_ylabel(y_label, fontsize=self.par.fontsize)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        v1 = np.linspace(0, lambda_max, 4, endpoint=True)
        cb = plt.colorbar(ticks=v1, shrink=1)
        cb.ax.set_yticklabels(["{:.1e}".format(i) for i in v1], fontsize=self.par.fontsize/1.5)
        cb.set_label(r"$\lambda$", fontsize=self.par.fontsize)
        plt.show()

        self.tool.initialize_directory_or_file(path=path_save)
        fig.savefig(path_save+filename+".png", bbox_inches="tight", dpi=100)
        fig.savefig(path_save+filename+".svg", bbox_inches="tight", dpi=100)


    def map_eigenvalues(self, eigen_map, S_array, C_array, fct_plot, path_save, loc_or_dir, linewidth=3, legend=True):
        """
        Plot the map of the eigenvalues
        
        """
        cmap = plt.get_cmap('plasma_r')
        xmin = S_array[0]
        xmax = S_array[-1]
        ymin = C_array[0]
        ymax = C_array[-1]
        lambda_max = np.nanmax(eigen_map)
        eigen_map_plot = eigen_map.copy()
        eigen_map_plot[eigen_map_plot < 1e-8] = np.nan

        fig, ax = plt.subplots(figsize=self.par.figsize)
        # ax = fig.add_subplot(111)
        im = plt.imshow(eigen_map_plot, extent=[xmin,xmax,ymin,ymax], origin='lower', cmap=cmap, aspect='auto', vmin=0, vmax=lambda_max)
        if fct_plot:
            if loc_or_dir == 'loc':
                ax.plot(S_array, self.mat.C_P_local(S_array), color='k',  linewidth=linewidth, linestyle='dashdot', alpha=1, label=r'$\tilde{C}(\bar{S})=\frac{0.5\times\bar{S}}{1+\bar{S}}$')
                ax.plot(S_array, self.mat.C_R_local(S_array), color='k', linewidth=linewidth, linestyle='dotted', alpha=1, label=r'$\tilde{C}(\bar{S})=\frac{0.5}{1+\bar{S}}$')
            if loc_or_dir == 'dir':
                ax.plot(S_array, self.mat.C_P_directional(S_array), color='k',  linewidth=linewidth, linestyle='dashdot', alpha=1, label=r'$\tilde{C}(\bar{S})=\frac{0.5\times\bar{S}}{1+\bar{S}}$')
                ax.plot(S_array, self.mat.C_R_directional(S_array), color='k', linewidth=linewidth, linestyle='dotted', alpha=1, label=r'$\tilde{C}(\bar{S})=\frac{0.5}{1+\bar{S}}$')

        ax.tick_params(labelsize=self.par.fontsize/1.5)
        ax.set_xlabel(r'$\bar{S}=\bar{R}\times\bar{F}$', fontsize=self.par.fontsize)
        ax.set_ylabel(r'Coupling intensity $\tilde{C}$', fontsize=self.par.fontsize)
        plt.xticks(np.arange(0, xmax, step=2), fontsize=self.par.fontsize/1.5)
        # plt.yticks(np.arange(0, ymax, step=0.2), fontsize=self.par.fontsize)
        # self.forceAspect(ax=ax, aspect=1)
        v1 = np.linspace(0, lambda_max, 4, endpoint=True)
        cb = plt.colorbar(ticks=v1, shrink=1)
        # cbar.ax.tick_params(labelsize=12)
        cb.ax.set_yticklabels(["{:3.1f}".format(i) for i in v1], fontsize=self.par.fontsize/1.5)
        # plt.gca().invert_yaxis()
        if legend:
            plt.legend(handles=[Line2D([0], [0], linestyle='dashdot', color='k', lw=linewidth, label=r'$\tilde{C}(\bar{S})=\frac{0.5\times\bar{S}}{1+\bar{S}}$'),
                                Line2D([0], [0], linestyle='dotted', color='k', lw=linewidth, label=r'$\tilde{C}(\bar{S})=\frac{0.5}{1+\bar{S}}$')],
                                loc='center left', bbox_to_anchor=(1.4, 0.5),
                                fontsize=self.par.fontsize)
        cb.set_label(r"$\lambda$", fontsize=self.par.fontsize)
        # plt.legend(fontsize=self.par.fontsize*0.7)
        plt.show()
        fig.savefig(path_save, bbox_inches="tight", dpi=100)


    def multiple_map_eigenvalues(self, 
                                 eigen_maps, 
                                 S_array, 
                                 C_array, 
                                 path_save,
                                 alpha=0.5,
                                 linewidth=3,
                                 color_local='purple',
                                 color_directional='green'):
        """
        Plot multiple maps of eigenvalues on the same plot
        
        """
        xmin = S_array[0]
        xmax = S_array[-1]
        ymin = C_array[0]
        ymax = C_array[-1]

        color_white = '#FFFFFF'  # Blanc

        # Définir les points du dégradé pour le colormap
        # Ici, nous utilisons une liste de dictionnaires pour définir les couleurs et leurs positions sur le dégradé
        colors_directional = [
            (0.0, color_white),          # Blanc au début du dégradé (position 0.0)
            (1.0, color_directional)   # Violet pastel à la fin du dégradé (position 1.0)
        ]

        colors_local = [
            (0.0, color_white),          # Blanc au début du dégradé (position 0.0)
            (1.0, color_local)   # Violet pastel à la fin du dégradé (position 1.0)
        ]
        
        # Créer le colormap personnalisé avec le dégradé de violet pastel allant du blanc au violet pastel
        cmap_P_directional = LinearSegmentedColormap.from_list('PastelDirectionalColors', colors_directional)
        cmap_P_local = LinearSegmentedColormap.from_list('PastelLocalColors', colors_local)

        # cmap_P_directional = plt.get_cmap('Greens')
        # cmap_P_local = plt.get_cmap('Purples')

        num_err = 10e-8
        array_P_directional = eigen_maps[0].copy()
        array_P_local = eigen_maps[1].copy()
        array_P_directional[array_P_directional<num_err] = np.nan
        array_P_local[array_P_local<num_err] = np.nan

        array_P_directional_uniform = array_P_directional
        array_P_directional_uniform[array_P_directional_uniform>0] = 1
        array_P_local_uniform = array_P_local
        array_P_local_uniform[array_P_local_uniform>0] = 1

        array_P_directional_uniform[~np.isnan(array_P_local_uniform)] = np.nan
        fig, ax = plt.subplots(figsize=self.par.figsize)
        plt.imshow(array_P_directional_uniform, extent=[xmin,xmax,ymin,ymax], origin='lower', cmap=cmap_P_directional, aspect='auto', vmin=0, vmax=1, alpha=alpha)
        plt.imshow(array_P_local_uniform, extent=[xmin,xmax,ymin,ymax], origin='lower', cmap=cmap_P_local, aspect='auto', vmin=0, vmax=1, alpha=alpha, label='Directional signal with $F$ modulation')
        ax.plot(S_array, self.mat.C_P(S_array), color='k', linewidth=linewidth, linestyle='dashdot', alpha=1, label=r'$\tilde{C}(\bar{S})=\frac{0.5\times\bar{S}}{1+\bar{S}}$')
        ax.plot(S_array, self.mat.C_R(S_array), color='k', linewidth=linewidth, linestyle='dotted', alpha=1, label=r'$\tilde{C}(\bar{S})=\frac{0.5}{1+\bar{S}}$')
        ax.tick_params(labelsize=self.par.fontsize/1.5)
        ax.set_xlabel(r'$\bar{S}=\bar{R}\times\bar{F}$', fontsize=self.par.fontsize)
        ax.set_ylabel(r'Coupling intensity $\tilde{C}$', fontsize=self.par.fontsize)
        plt.xticks(np.arange(0, xmax, step=2), fontsize=self.par.fontsize/1.5)
        # plt.yticks(np.arange(0, ymax, step=0.2), fontsize=self.par.fontsize/1.5)
        self.forceAspect(ax, aspect=1.1)
        # plt.gca().invert_yaxis()
        plt.legend(handles=[Line2D([0], [0], linestyle='dotted', color='k', lw=linewidth, label='Signal < threshold'),
                            Line2D([0], [0], linestyle='dashdot', color='k', lw=linewidth, label='Signal > threshold'),
                            plt.Rectangle((0, 0), 1, 1, color=color_local, alpha=alpha, label='Local density signal'),
                            plt.Rectangle((0, 0), 1, 1, color=color_directional, alpha=alpha, label='Directional density signal')],
                            loc='center left', bbox_to_anchor=(1, 0.5),
                            fontsize=self.par.fontsize)
        plt.show()
        fig.savefig(path_save, bbox_inches="tight", dpi=100)


# # %%
# # TESTS
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import parameters
# import linearisation_signal
# par = parameters.Parameters()
# lin = linearisation_signal.Linearisation(a=1)
# plo = Plot(fontsize=20, figsize=(10,10))
# # Exemple de matrice n*n avec des valeurs et des NaN
# path_R_directional = "W:/jb/python/myxococcus_xanthus_simu1d/linearisation/results/eigenvalue_map_directional_density/data_eigenvalues_directional_density_reversal_rate.csv"
# array_R_directional = pd.read_csv(path_R_directional, header=None).values

# path_P_directional = "W:/jb/python/myxococcus_xanthus_simu1d/linearisation/results/eigenvalue_map_directional_density/data_eigenvalues_directional_density_refractory_period.csv"
# array_P_directional = pd.read_csv(path_P_directional, header=None).values

# path_R_local = "W:/jb/python/myxococcus_xanthus_simu1d/linearisation/results/eigenvalue_map_local_density/data_eigenvalues_local_density_reversal_rate.csv"
# array_R_local = pd.read_csv(path_R_local, header=None).values

# path_P_local = "W:/jb/python/myxococcus_xanthus_simu1d/linearisation/results/eigenvalue_map_local_density/data_eigenvalues_local_density_refractory_period.csv"
# array_P_local = pd.read_csv(path_P_local, header=None).values

# num_err = 10e-8
# array_P_directional[array_P_directional<num_err] = np.nan
# array_R_directional[array_R_directional<num_err] = np.nan

# array_P_local[array_P_local<num_err] = np.nan
# array_R_local[array_R_local<num_err] = np.nan

# # %%
# xmin = par.S_array[0]
# xmax = par.S_array[-1]
# ymin = par.C_array[0]
# ymax = par.C_array[-1]
# labelsize = 20

# array_P_directional_uniform = array_P_directional.copy()
# array_P_directional_uniform[array_P_directional_uniform>0] = 1
# array_R_directional_uniform = array_R_directional.copy()
# array_R_directional_uniform[array_R_directional_uniform>0] = 1

# array_P_local_uniform = array_P_local.copy()
# array_P_local_uniform[array_P_local_uniform>0] = 1
# array_R_local_uniform = array_R_local.copy()
# array_R_local_uniform[array_R_local_uniform>0] = 1


# cond_plot_local = ~np.isnan(array_P_local_uniform).flatten()
# cond_plot_directional = ~np.isnan(array_P_directional_uniform).flatten() & np.isnan(array_P_local_uniform).flatten()
# alpha = 0.4
# fontsize = 22
# figsize = (10,10)
# linewidth = 4
# color_local = 'purple'
# color_directional = 'green'
# square_marker = [(0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5), (0.5, -0.5), (0.5, 0.5)]
# marker_local = square_marker
# marker_directional = square_marker
# s_local = 80
# s_directional = 80

# # fig, ax = plt.subplots()
# # ax.scatter(x=par.S_grid[:, :, 0].flatten()[cond_plot_local], 
# #            y=par.C_grid[:, :, 0].flatten()[cond_plot_local],
# #            alpha=alpha, 
# #            label='Local density signal',
# #            c=color_local,
# #            s=s_local,
# #            marker=marker_local,
# #            zorder=2)
# # ax.scatter(x=par.S_grid[:, :, 0].flatten()[cond_plot_directional], 
# #            y=par.C_grid[:, :, 0].flatten()[cond_plot_directional],
# #            alpha=alpha, 
# #            label='Directional density signal',
# #            c=color_directional,
# #            s=s_directional,
# #            marker=marker_directional)
# # ax.plot(par.S_array, lin.C_P(par.S_array), color='k', linewidth=linewidth, linestyle='dashdot', alpha=1, label=r'High signalling constrain')
# # ax.plot(par.S_array, lin.C_R(par.S_array), color='k', linewidth=linewidth, linestyle='dotted', alpha=1, label=r'Low signalling constrain')
# # ax.set_xlabel(r'$\bar{S}$', fontsize=fontsize)
# # ax.set_ylabel(r'$\tilde{C}$', fontsize=fontsize)
# # ax.set_xlim(xmin, xmax)
# # ax.set_ylim(ymin, ymax)
# # ax.tick_params(labelsize=fontsize)
# # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.65), ncol=1, fontsize=fontsize, markerscale=fontsize/8)
# # Tracer la matrice avec les valeurs en couleurs et les NaN en blanc
# alpha = 0.5
# fontsize = 25
# figsize = (10,10)
# linewidth = 3

# cmap_P_directional = plt.get_cmap('Greens')
# cmap_P_local = plt.get_cmap('Purples')
# color_local = 'purple'
# color_directional = 'green'

# array_P_directional_uniform[~np.isnan(array_P_local_uniform)] = np.nan
# fig, ax = plt.subplots(figsize=figsize)
# plt.imshow(array_P_directional_uniform, extent=[xmin,xmax,ymax,ymin], cmap=cmap_P_directional, aspect=10, vmin=0, vmax=1, alpha=alpha)
# plt.imshow(array_P_local_uniform, extent=[xmin,xmax,ymax,ymin], cmap=cmap_P_local, aspect=10, vmin=0, vmax=1, alpha=alpha, label='Directional signal with $F$ modulation')
# ax.plot(par.S_array, lin.C_P(par.S_array), color='k', linewidth=linewidth, linestyle='dashdot', alpha=1, label=r'$\tilde{C}(\bar{S})=\frac{0.5\times\bar{S}}{1+\bar{S}}$')
# ax.plot(par.S_array, lin.C_R(par.S_array), color='k', linewidth=linewidth, linestyle='dotted', alpha=1, label=r'$\tilde{C}(\bar{S})=\frac{0.5}{1+\bar{S}}$')
# ax.tick_params(labelsize=fontsize)
# ax.set_xlabel(r'$\bar{S}$', fontsize=fontsize)
# ax.set_ylabel(r'$\tilde{C}$', fontsize=fontsize)
# plt.xticks(np.arange(0, xmax, step=2), fontsize=fontsize)
# plt.yticks(np.arange(0, ymax, step=0.2), fontsize=fontsize)
# plo.forceAspect(ax, aspect=1.1)
# plt.gca().invert_yaxis()
# # plt.legend(fontsize=labelsize)
# plt.legend(handles=[Line2D([0], [0], linestyle='dotted', color='k', lw=linewidth, label='Signal < threshold'),
#                     Line2D([0], [0], linestyle='dashdot', color='k', lw=linewidth, label='Signal > threshold'),
#                     plt.Rectangle((0, 0), 1, 1, color=color_local, alpha=alpha, label='Local density signal'),
#                     plt.Rectangle((0, 0), 1, 1, color=color_directional, alpha=alpha, label='Directional density signal')],
#                     loc='center left', bbox_to_anchor=(1, 0.5),
#                     fontsize=fontsize)
# # %%
# cmap = plt.get_cmap('plasma_r')
# xmin = par.S_array[0]
# xmax = par.S_array[-1]
# ymin = par.C_array[0]
# ymax = par.C_array[-1]
# labelsize = 20
# lambda_max = np.nanmax(array_P_local)
# alpha = 1

# fig, ax = plt.subplots()
# plt.imshow(array_P_local, extent=[xmin,xmax,ymax,ymin], cmap=cmap, aspect=10, vmin=0, vmax=lambda_max, alpha=alpha)
# ax.plot(par.S_array, lin.C_P(par.S_array), color='limegreen', linewidth=2, linestyle='dotted', alpha=1, label=r'$\tilde{C}=\frac{0.5\times\bar{S}}{1+\bar{S}}$')
# ax.tick_params(labelsize=labelsize)
# ax.set_xlabel(r'$\bar{S}=R^*\times \bar{F}$', fontsize=labelsize)
# ax.set_ylabel(r'$\tilde{C}$', fontsize=labelsize)
# plt.xticks(np.arange(0, xmax, step=2), fontsize=labelsize)
# plt.yticks(np.arange(0, ymax, step=0.2), fontsize=labelsize)
# plo.forceAspect(ax, aspect=1)
# v1 = np.linspace(0, lambda_max, 4, endpoint=True)
# cb = plt.colorbar(ticks=v1, shrink=1)
# cb.ax.set_yticklabels(["{:3.1f}".format(i) for i in v1], fontsize='15')
# plt.gca().invert_yaxis()
# cb.set_label("$\lambda$", fontsize=labelsize)
# plt.legend(fontsize=labelsize*0.7)

# lambda_max = np.nanmax(array_R_local)

# fig, ax = plt.subplots()
# plt.imshow(array_R_local, extent=[xmin,xmax,ymax,ymin], cmap=cmap, aspect=10, vmin=0, vmax=lambda_max, alpha=alpha)
# ax.plot(par.S_array, lin.C_R(par.S_array), color='limegreen', linewidth=2, linestyle='dotted', alpha=1, label=r'$\tilde{C}=\frac{0.5\times\bar{S}}{1+\bar{S}}$')
# ax.tick_params(labelsize=labelsize)
# ax.set_xlabel(r'$\bar{S}=R^*\times \bar{F}$', fontsize=labelsize)
# ax.set_ylabel(r'$\tilde{C}$', fontsize=labelsize)
# plt.xticks(np.arange(0, xmax, step=2), fontsize=labelsize)
# plt.yticks(np.arange(0, ymax, step=0.2), fontsize=labelsize)
# plo.forceAspect(ax, aspect=1)
# v1 = np.linspace(0, lambda_max, 4, endpoint=True)
# cb = plt.colorbar(ticks=v1, shrink=1)
# cb.ax.set_yticklabels(["{:3.1f}".format(i) for i in v1], fontsize='15')
# plt.gca().invert_yaxis()
# cb.set_label("$\lambda$", fontsize=labelsize)
# plt.legend(fontsize=labelsize*0.7)
# # %%
