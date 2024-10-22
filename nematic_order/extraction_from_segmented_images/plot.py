# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# import tikzplotlib
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'cmr10'  # "Computer Modern Roman"


class Plot:

    def __init__(self, inst_par):
        
        self.par = inst_par


    def plot_nematic_over_distance(self, list_df, list_names, list_colors, list_linestyles, list_markers, name_save, legend):
        """
        Plot the nematic over distance and compute the error bar
        
        """
        nb_different_style = np.unique(list_names)

        count = 0
        array_legend = []
        fontsize = 30
        fig, ax = plt.subplots(figsize=(8,6))
        for name in nb_different_style:

            nb_same_name = np.sum(np.array(list_names) == name)
            array_legend.append(Line2D([0], [0], linestyle=list_linestyles[count], color=list_colors[count], lw=3, label=list_names[count], marker=list_markers[count], markeredgewidth=5))

            for j in range(nb_same_name):
                df = list_df[count]
                r_array = df.loc[:, 'dist'].values.astype('int')
                nematic_order = df.loc[:, 'nematic_order'].values
                distances = np.unique(r_array)
                mean_nematic_order = np.zeros(len(distances))
                err_nematic_order = np.zeros(len(distances))
                for count_dist, distance in enumerate(distances):
                    cond_dist = r_array == distance
                    nematic_at_dist = nematic_order[cond_dist]
                    mean_nematic_order[count_dist] = np.nanmean(nematic_at_dist)
                    err_nematic_order[count_dist] = np.nanstd(nematic_at_dist)# / np.sqrt(len(nematic_at_dist))
                ax.plot(distances, mean_nematic_order, color=list_colors[count], linestyle=list_linestyles[count], marker=list_markers[count], lw=3)
                # ax.errorbar(distances, mean_nematic_order, yerr=err_nematic_order, fmt='none', ecolor=list_colors[count], capsize=8)
                y_upper = [mean_nematic_order[i] + err_nematic_order[i] for i in range(len(mean_nematic_order))]
                y_lower = [mean_nematic_order[i] - err_nematic_order[i] for i in range(len(mean_nematic_order))]
                ax.fill_between(distances, y_upper, y_lower, color=list_colors[count], alpha=0.15, edgecolor='None')
                count += 1

        ax.set_xlabel(r'$r\ (\mu m)$', fontsize=fontsize)
        # ax.set_ylabel(r'$S = \langle \sum_j \cos(2(\theta_i - \theta_j)) \lrangle$', fontsize=fontsize)
        eq2 = (r"$S(r) = \left\langle \frac{1}{N} \sum_{i=1}^N \cos^2(2\theta_i(r)) \right\rangle$")
        ax.set_ylabel(eq2, fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.set_xticks(np.arange(np.min(r_array), np.max(r_array), 20))
        ax.set_yticks(np.arange(0, 1.2, 0.2))
        if legend:
            ax.legend(handles=array_legend,
                    loc='best',
                    fontsize=fontsize)
        fig.savefig('results/plot/'+name_save+'.png', dpi=100, bbox_inches='tight')


    # def plot_nematic_over_distance(self, list_files):
    #     """
    #     Plot the mean nematic value of each grid cell over the distance
    #     of other cell inside a shell at distance r
        
    #     """
    #     files_rippling = list_files[0]
    #     files_swarming = list_files[1]
    #     fontsize = 15
    #     fig, ax = plt.subplots(figsize=(8,6))

    #     for i in range(len(files_rippling)):
    #         correlation_nematic_order = pd.read_csv(files_rippling[i]).loc[:, 'nematic_value'].values
    #         r_array = pd.read_csv(files_rippling[i]).loc[:, 'distance'].values * self.par.scale

    #         ax.plot(r_array, correlation_nematic_order, color='k', linestyle='-')
    #         # ax.scatter(r_array, correlation_nematic_order, s=10)

    #     for i in range(len(files_swarming)):
    #         correlation_nematic_order = pd.read_csv(files_swarming[i]).loc[:, 'nematic_value'].values
    #         r_array = pd.read_csv(files_swarming[i]).loc[:, 'distance'].values * self.par.scale

    #         ax.plot(r_array, correlation_nematic_order, color='grey', linestyle='-')
    #         # ax.scatter(r_array, correlation_nematic_order, s=10)

    #     ax.set_xlabel(r'$r\ (\mu m)$', fontsize=fontsize)
    #     # ax.set_ylabel(r'$S = \langle \sum_j \cos(2(\theta_i - \theta_j)) \lrangle$', fontsize=fontsize)
    #     eq2 = (r"\begin{eqnarray*}"
    #             r"S &=& \left \langle \sum_j \cos\left ( 2(\theta_i - \theta_j) \right ) \right \rangle "
    #             r"\end{eqnarray*}")

    #     ax.set_ylabel(eq2, fontsize=fontsize)
    #     ax.tick_params(labelsize=fontsize)
    #     # ax.legend(handles=[Line2D([0], [0], linestyle='-', color='k', lw=1, label='Rippling'),
    #     #                    Line2D([0], [0], linestyle='-', color='k', lw=1, label='Swarming'),],
    #     #                    loc='center left', bbox_to_anchor=(1, 0.5),
    #     #                    fontsize=fontsize)
    #     # ax.legend(['Rippling'], loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fontsize)
    #     fig.savefig('results/plot/nematic_order_swarming_rippling.png', dpi=100)
    #     # tikzplotlib.save('results/plot/nematic_order_swarming_rippling.tikz')
