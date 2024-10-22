import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
import numpy as np
from tools import Tools
from igoshin_matrices import IgoshinMatrix
import os

class TypeTypeError(Exception):
    pass


class Plot:


    def __init__(self, inst_par):
        
        self.par = inst_par
        self.tool = Tools()
        self.mat = IgoshinMatrix(inst_par)


    def plot_eigenvalues_2d(self,
                            data_array,
                            output_folder, output_file_name,
                            x_label, y_label,
                            cbar_label,
                            cmap,
                            vmin, vmax, vstep,
                            uncertenties = True):
        """
        Plot the map of the eigenvalues
        
        """
        xmin = data_array[1][0]
        xmax = data_array[1][-1]
        ymin = data_array[2][0]
        ymax = data_array[2][-1]
        eigen_map_plot = data_array[0].copy()
        if uncertenties:
            eigen_map_plot[eigen_map_plot < self.par.dp] = np.nan

        fig, ax = plt.subplots(figsize=self.par.figsize)
        im = plt.imshow(eigen_map_plot.T, 
                        extent=[xmin, xmax, ymin, ymax], 
                        origin='lower', 
                        cmap=cmap, 
                        aspect='auto', 
                        vmin=vmin, 
                        vmax=vmax)
        
        ax.set_xlabel(x_label, fontsize=self.par.fontsize)
        ax.set_ylabel(y_label, fontsize=self.par.fontsize)
        ax.tick_params(labelsize=self.par.fontsize_ticks)

        # Division de l'axe pour ajouter la colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        # Ajout de la colorbar
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label(cbar_label, fontsize=self.par.fontsize)
        cbar.set_ticks(np.arange(vmin, vmax+vstep, vstep))  # Set the tick positions for the colorbar
        cbar.ax.tick_params(labelsize=self.par.fontsize_ticks)

        isExist = os.path.exists(output_folder)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(output_folder)
            print("The new directory is created!")
        fig.savefig(output_folder+output_file_name+'.png', bbox_inches='tight', dpi=self.par.dpi)
        fig.savefig(output_folder+output_file_name+'.svg', dpi=self.par.dpi)