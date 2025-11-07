import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from tools import Tools
from igoshin_matrices import IgoshinMatrix

class TypeTypeError(Exception):
    pass


class Plot:


    def __init__(self, inst_par):
        
        self.par = inst_par
        self.tool = Tools()
        self.mat = IgoshinMatrix(inst_par)


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
        eps = 1e-13
        cond_instabilities = data_array[:, 0] > eps

        fig, ax = plt.subplots(figsize=self.par.figsize)
        ax.plot(data_array[:, 1], data_array[:, 0], color='k', linewidth=2, zorder=0)
        ax.scatter(data_array[~cond_instabilities, 1], data_array[~cond_instabilities, 0], c='b', marker='v', s=25, label='values $\leq 0$', zorder=1, alpha=0.5)
        ax.scatter(data_array[cond_instabilities, 1], data_array[cond_instabilities, 0], c='r', marker='^', s=25, label='values $> 0$', zorder=1, alpha=0.5)

        # for j in range(len(cond_instabilities)):
        #         if cond_instabilities[j]:
        #             linestyle = '--'
        #         else:
        #             linestyle = '-'
        #         ax.plot(data_array[j:j+2, 1], data_array[j:j+2, 0], color='k', linestyle=linestyle, lw=4, zorder=1, alpha=0.5)

        # custom_lines = [Line2D([0], [0], color='k', linestyle='--', label='Eigenvalues > 0', lw=4, alpha=0.5),
        #                 Line2D([0], [0], color='k', linestyle='-', label='Eigenvalues < 0', lw=4, alpha=0.5)
        #                 ]


        ax.tick_params(labelsize=self.par.fontsize/1.5)
        ax.set_xlabel(x_label, fontsize=self.par.fontsize)
        ax.set_ylabel(y_label, fontsize=self.par.fontsize)
        ax.set_xlim(0, xmax)
        # ax.legend(handles=custom_lines, fontsize=self.par.fontsize/1.5, markerscale=4)
        ax.legend(fontsize=self.par.fontsize/1.5, markerscale=4)
        plt.show()

        self.tool.initialize_directory_or_file(path=path_save)
        fig.savefig(path_save+filename+".png", bbox_inches="tight", dpi=100)
        fig.savefig(path_save+filename+".svg", dpi=100)


    def plot_eigenvalues_1d_multiple(self, 
                                     data_arrays, 
                                     path_save,
                                     filename,
                                     legend_names,
                                     colors,
                                     x_label,
                                     y_label,
                                     xmax):
        """
        Plot signal in function of the eigenvalues
        
        """
        fig, ax = plt.subplots(figsize=self.par.figsize)
        xmin = data_arrays[0][0, 1]
        if xmax:
            pass
        else:
            xmax = data_arrays[0][-1, 1]
        custom_lines = []

        for i in range(len(data_arrays)):
            eps = 1e-13
            cond_instabilities = data_arrays[i][:, 0] > eps
            
            for j in range(len(cond_instabilities)):
                if cond_instabilities[j]:
                    linestyle = 'dotted'
                else:
                    linestyle = '-'
                ax.plot(data_arrays[i][j:j+2, 1], data_arrays[i][j:j+2, 0], color=colors[i], linestyle=linestyle, lw=6, zorder=1, alpha=0.5)

            custom_lines.append(Line2D([0], [0], color=colors[i], label=legend_names[i], lw=6, alpha=0.5))
        
        ax.tick_params(labelsize=self.par.fontsize/1.5)
        ax.set_xlabel(x_label, fontsize=self.par.fontsize)
        ax.set_ylabel(y_label, fontsize=self.par.fontsize)
        ax.set_xlim(0, xmax)

        ax.legend(handles=custom_lines, fontsize=self.par.fontsize/1.5, markerscale=1)
        plt.show()

        self.tool.initialize_directory_or_file(path=path_save)
        fig.savefig(path_save+filename+".png", bbox_inches="tight", dpi=100)
        fig.savefig(path_save+filename+".svg", dpi=100)


    def plot_eigenvectors_1d_multiple(self, 
                                      data_arrays, 
                                      path_save,
                                      filename,
                                      legend_names,
                                      colors,
                                      x_label,
                                      y_label,
                                      xmax):
        """
        Plot signal in function of the eigenvalues
        
        """
        fig, ax = plt.subplots(figsize=self.par.figsize)
        xmin = data_arrays[0][0, 1]
        if xmax:
            pass
        else:
            xmax = data_arrays[0][-1, 1]
        custom_lines = []

        for i in range(len(data_arrays)):

            ax.plot(data_arrays[i][:, 1], data_arrays[i][:, 2], color=colors[i], lw=6, zorder=1, alpha=0.5)

            custom_lines.append(Line2D([0], [0], color=colors[i], label=legend_names[i], lw=6, alpha=0.5))
        
        ax.tick_params(labelsize=self.par.fontsize/1.5)
        ax.set_xlabel(x_label, fontsize=self.par.fontsize)
        ax.set_ylabel(y_label, fontsize=self.par.fontsize)
        ax.set_xlim(0, xmax)

        ax.legend(handles=custom_lines, fontsize=self.par.fontsize/1.5, markerscale=1)
        plt.show()

        self.tool.initialize_directory_or_file(path=path_save)
        fig.savefig(path_save+filename+".png", bbox_inches="tight", dpi=100)
        fig.savefig(path_save+filename+".svg", dpi=100)


    def plot_eigenvalues_2d(self,
                            data_array,
                            path_save,
                            filename,
                            x_label,
                            y_label,
                            type=None):
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
        eigen_map_plot[eigen_map_plot < 1e-13] = np.nan

        fig, ax = plt.subplots(figsize=self.par.figsize)
        # ax = fig.add_subplot(111)
        im = plt.imshow(eigen_map_plot.T, 
                        extent=[xmin, xmax, ymin, ymax], 
                        origin='lower', 
                        cmap=cmap, 
                        aspect='auto', 
                        vmin=0, 
                        vmax=lambda_max)
        
        if type == 'main':
            K_1 = []
            for S in self.par.S_array:
                K_1.append(self.mat.K_1_linear(S=S))
            ax.plot(self.par.S_array, K_1)
            ax.scatter(self.par.S_array, K_1)
            print('S: ', self.par.S_array)
            print('K_1: ', K_1)
        elif type == 'rp':
            K_1_list = []
            for S in self.par.S_array:
                K_1 = self.mat.K_1_rp(S=S)
                K_1_list.append(K_1)
            ax.plot(self.par.S_array, K_1_list, linewidth=1)
            ax.scatter(self.par.S_array, K_1_list)
            print('S: ', self.par.S_array)
            print('K_1: ', K_1_list)
        elif type is None:
            pass
        else:
            raise TypeTypeError()
        ax.tick_params(labelsize=self.par.fontsize/1.5)
        ax.set_xlabel(x_label, fontsize=self.par.fontsize)
        ax.set_ylabel(y_label, fontsize=self.par.fontsize)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        # plt.xticks(np.arange(0, xmax, step=2), fontsize=self.par.fontsize/1.5)
        # plt.yticks(np.arange(0, ymax, step=0.2), fontsize=self.par.fontsize)
        # self.forceAspect(ax=ax, aspect=1)
        v1 = np.linspace(0, lambda_max, 4, endpoint=True)
        cb = plt.colorbar(ticks=v1, shrink=1)
        # cbar.ax.tick_params(labelsize=12)
        cb.ax.set_yticklabels(["{:.1e}".format(i) for i in v1], fontsize=self.par.fontsize/1.5)
        # plt.gca().invert_yaxis()
        cb.set_label(r"$\lambda$", fontsize=self.par.fontsize)
        # plt.legend(fontsize=self.par.fontsize*0.7)
        plt.show()

        self.tool.initialize_directory_or_file(path=path_save)
        fig.savefig(path_save+filename+".png", bbox_inches="tight", dpi=100)
        fig.savefig(path_save+filename+".svg", dpi=100)