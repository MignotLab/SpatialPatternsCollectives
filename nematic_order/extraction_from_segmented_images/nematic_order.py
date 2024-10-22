import numpy as np
import matplotlib.pyplot as plt
import grispy as gsp
from tqdm import tqdm
import csv
import os

import tools

class NematicOrder:


    def __init__(self, inst_par, sample):
        
        # Import class
        self.par = inst_par
        self.tool = tools.Tools()

        self.max_search_dist = self.par.space_size / 2
        self.dr = self.par.dr

        columns = ['frame', 'nematic_order', 'dist']
        self.file_name_save = self.par.path_save+sample
        self.tool.initialize_directory_or_file(self.file_name_save, columns)


    def fct_nematic_order(self, theta):
        """
        Fonction to compute the nematic order
        
        """

        return 1/len(theta) * np.nansum(2 * np.cos(theta)**2) - 1
    

    def compute_nematic_order(self, df):
        """
        Compute the nematic order of bacteria
        The mean angle of each bacteria is define as end to end bacteria point
        The nematic order for a distance r is define as the mean of all nematic order of each bacterium with
        bacteria in a shell at distance r
        
        """
        cond_frame = df.loc[:, self.par.t_column] >= self.par.first_frame
        df_tmp = df.loc[cond_frame, :].copy()

        x0 = df_tmp.loc[:, self.par.x_column_first].values
        x_mid = df_tmp.loc[:, self.par.x_column].values
        xn = df_tmp.loc[:, self.par.x_column_last].values
        y0 = df_tmp.loc[:, self.par.y_column_first].values
        y_mid = df_tmp.loc[:, self.par.y_column].values
        yn = df_tmp.loc[:, self.par.y_column_last].values

        frames_indices = np.unique(df_tmp.loc[:, self.par.t_column])[::self.par.interval_frames]
        r_array = np.arange(self.dr, int(self.par.space_size / 2 - self.dr), self.dr)

        for frame_count, frame_id in enumerate(tqdm(frames_indices, desc='# frame')):
            cond_frame = (df_tmp.loc[:, self.par.t_column] == frame_id).values
            n_bact = np.sum(cond_frame)
            nematic_order = np.zeros(n_bact)
            angles = np.arctan2(yn[cond_frame] - y0[cond_frame], xn[cond_frame] - x0[cond_frame])

            coord = np.column_stack((x_mid[cond_frame], y_mid[cond_frame]))
            grid = gsp.GriSPy(coord, N_cells=32)

            for r in r_array:
                upper_radii = r + self.dr / 2
                lower_radii = r - self.dr / 2
                __, shell_ind = grid.shell_neighbors(coord,
                                                     distance_lower_bound=lower_radii,
                                                     distance_upper_bound=upper_radii)
                
                # Loop over all bacteria
                # Debug 1: plot a bacteria and its selected neighbhours for a given radius
                for bact_count in range(n_bact):
                    # Compute the mean nematic order of the bacteria i with their neighbours
                    if len(shell_ind[bact_count]) > self.par.min_shell_neighbours:
                        theta = angles[bact_count] - angles[shell_ind[bact_count]]
                        nematic_order[bact_count] = self.fct_nematic_order(theta)
                    else:
                        nematic_order[bact_count] = np.nan
                
                # SAVE DATA AS TEXT
                with open(self.file_name_save, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(np.column_stack((frame_count * self.par.tbf * np.ones(n_bact), 
                                                      nematic_order, r * self.par.scale * np.ones(n_bact))))