import matplotlib.pyplot as plt
import numpy as np
from parameters import Parameters
from tools import Tools
from igoshin_matrices import IgoshinMatrix

class TypeTypeError(Exception):
    pass


class Plot:


    def __init__(self):
        
        self.par = Parameters()
        self.tool = Tools()
        self.mat = IgoshinMatrix()


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


    # def plot_eigenvalues_2d(self, 
    #                         data_array,
    #                         path_save,
    #                         filename,
    #                         x_label,
    #                         y_label):
    #     """
    #     Plot the map of the eigenvalues
        
    #     """
    #     cmap = plt.get_cmap('plasma_r')
    #     xmin = data_array[1][0]
    #     xmax = data_array[1][-1]
    #     ymin = data_array[2][0]
    #     ymax = data_array[2][-1]
    #     lambda_max = np.nanmax(data_array[0])
    #     eigen_map_plot = data_array[0].copy()
    #     eigen_map_plot[eigen_map_plot < 10e-8] = np.nan

    #     fig, ax = plt.subplots(figsize=self.par.figsize)
    #     # ax = fig.add_subplot(111)
    #     im = plt.imshow(eigen_map_plot, extent=[xmin,xmax,ymin,ymax], origin='lower', cmap=cmap, aspect='auto', vmin=0, vmax=lambda_max)
    #     ax.tick_params(labelsize=self.par.fontsize/1.5)
    #     ax.set_xlabel(x_label, fontsize=self.par.fontsize)
    #     ax.set_ylabel(y_label, fontsize=self.par.fontsize)
    #     ax.set_xlim(xmin, xmax)
    #     ax.set_ylim(ymin, ymax)
    #     # plt.xticks(np.arange(0, xmax, step=2), fontsize=self.par.fontsize/1.5)
    #     # plt.yticks(np.arange(0, ymax, step=0.2), fontsize=self.par.fontsize)
    #     # self.forceAspect(ax=ax, aspect=1)
    #     v1 = np.linspace(0, lambda_max, 4, endpoint=True)
    #     cb = plt.colorbar(ticks=v1, shrink=1)
    #     # cbar.ax.tick_params(labelsize=12)
    #     cb.ax.set_yticklabels(["{:.1e}".format(i) for i in v1], fontsize=self.par.fontsize/1.5)
    #     # plt.gca().invert_yaxis()
    #     cb.set_label(r"$\lambda$", fontsize=self.par.fontsize)
    #     # plt.legend(fontsize=self.par.fontsize*0.7)
    #     plt.show()

    #     self.tool.initialize_directory_or_file(path=path_save)
    #     fig.savefig(path_save+filename+".png", bbox_inches="tight", dpi=100)
    #     fig.savefig(path_save+filename+".svg", bbox_inches="tight", dpi=100)


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
        eigen_map_plot[eigen_map_plot < 10e-8] = np.nan

        fig, ax = plt.subplots(figsize=self.par.figsize)
        # ax = fig.add_subplot(111)
        im = plt.imshow(eigen_map_plot.T, extent=[xmin,xmax,ymin,ymax], origin='lower', cmap=cmap, aspect='auto', vmin=0, vmax=lambda_max)
        
        if type == 'main':
            w_1 = []
            K_1 = []
            for rho_bar in self.par.rho_bar_array:
                w_1.append(self.mat.w_1(rho_bar, self.par.q_value_constant))
                K_1.append(self.mat.K_1(rho_bar, self.par.q_value_constant))
            ax.plot(w_1, K_1)
            ax.scatter(w_1, K_1)
            print('w_1: ', w_1)
            print('K_1: ', K_1)
        elif type == 'rp':
            phi_r_list = []
            K_2_list = []
            for rho_bar in self.par.rho_bar_rp_array:
                phi_r = self.mat.phi_r(rho_bar=rho_bar)
                K_2 = self.mat.K_2(phi_r=phi_r)
                phi_r_list.append(phi_r)
                K_2_list.append(K_2)
            ax.plot(phi_r_list, K_2_list, linewidth=1)
            ax.scatter(phi_r_list, K_2_list)
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
        fig.savefig(path_save+filename+".svg", bbox_inches="tight", dpi=100)