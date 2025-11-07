# %%
#%%
if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    import sys

    # .py files import
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from analysis.reversals_detection import (
        reversals_detection,
        cell_direction,
        reversal_signal,
        tools,
        plots
    )

    class Parameters:


        def __init__(self):

            ### PARALLELIZATION ###
            self.n_jobs = 12

            ### MOVIE ###
            self.scale = 0.0646028 # in µm/px
            # self.scale = 0.0430685 # µm / px
            # self.scale = 1 # in µm/px
            self.tbf = 2/60 # in min/frame
            self.start_frame = 0
            # self.start_frame = 1800

            ### TRACKING ###
            self.track_mate_csv = False
            # In the case where several point have been detected for the bacteria
            # In this case the csv coordinates must be as (0_x,0_y,...,n_x,n_y)
            # Put to 1 if only one point is detect per bacteria
            self.n_nodes = 11
            self.width = 0.7 # µm
            self.kn = 30 # Number of detected neighbours to then compute the number of neighbour
            
            ### REVERSALS DETECTION ###
            ## csv tracking path
            # You can add several input files for analysis
            self.path_input_folders = [
                "output/tracking/",
                # add more folders as needed
            ]
            self.input_filenames = [
                'test_tracking.csv',
                # add same amount of filenames as path_folders
            ]

            # Path to save results
            self.path_output_folder = 'output/reversals_and_signaling_analysis/'

            # csv column names
            self.track_id_column = 'id'
            self.t_column = 'frame'
            self.x_column = 'x_centroid'
            self.y_column = 'y_centroid'
            # self.x_column = 'x5'
            # self.y_column = 'y5'

            # Name of columns that could be created in this script
            self.v_column = 'velocity'
            self.rev_column = 'reversals'
            self.rev_memory_column = 'reversals_memory'
            self.tbr_column = 'tbr'
            self.traj_length_column = 'traj_length'
            self.local_frustration_column = 'local_frustration'
            self.cumul_frustration_column = 'cumul_frustration'
            self.n_neighbours_column = 'n_neighbours'
            self.n_neg_neighbours_column = 'n_neg_neighbours'
            self.mean_polarity_column = 'mean_polarity'
            self.n_neighbours_igoshin_column = 'n_neighbours_igoshin'
            self.mean_polarity_igoshin_column = 'mean_polarity_igoshin'

            self.iteration = 5
            # reversals parameters for their detection
            self.angle_rev = np.pi / 2 # in rad
            # self.min_smooth_size = 2 # in µm

            ### REVERSAL SIGNALING ###
            self.max_frustration = 2
            self.frustration_time_memory = 18 / 60 # in minutes
            self.cumul_frustration_decreasing_rate = 1 # in 1 / minutes
            self.angle_view = np.pi / 2

            ### FRUSTRATION ###
            self.path_save_movie_frustration = "movie_frustration/"

    par = Parameters()
    tool = tools.Tools()


    ###################################### REVERSALS DETECTION ######################################
    #################################################################################################
    start_min_size_smoothed_um = 0.1 # this allow to smooth trajectory starting from a small to big smoothing parameter
    step_min_size_smooth_um = 0.1 # start at 0.1 µm and increase by 0.1 µm steps until 1 µm
    list_end_min_size_smoothed_um = [1] # you can add more values for testing
    color_rippling = tool.get_rgba_color(color_name='mediumturquoise', alpha=0.7)
    color_swarming = tool.get_rgba_color(color_name='darkorange', alpha=0.7)
    
    for i, (path_input_folder, input_filename) in enumerate(zip(par.path_input_folders, par.input_filenames)):
        print('Analyse file', input_filename)
        df_input = pd.read_csv(path_input_folder + input_filename)

        if par.start_frame != 0:
            # Remove all frames strictly before start_frame
            df_input = df_input[df_input[par.t_column] >= par.start_frame].copy()
            # Rebase frames to start at 0
            df_input[par.t_column] = df_input[par.t_column] - df_input[par.t_column].min()

        for end_min_size_smoothed_um in list_end_min_size_smoothed_um:
            print('Computation for smooth parameter equal to', end_min_size_smoothed_um, 'um')
            input_filename_no_ext = os.path.splitext(input_filename)[0]
            path_output = os.path.join(par.path_output_folder, input_filename_no_ext)
            # REVERSALS
            rev = reversals_detection.ReversalsDetection(par=par, 
                                                         df=df_input,
                                                         end_min_size_smoothed_um=end_min_size_smoothed_um,
                                                         start_min_size_smoothed_um=start_min_size_smoothed_um,
                                                         step_min_size_smooth_um=step_min_size_smooth_um
                                                        )
            rev.reversals_detection()
            end_filename = 'min_size_smoothed_um=' + str(end_min_size_smoothed_um) + '_um'
            
            # DIRECTION EXTRACTION
            cdir = cell_direction.CellDirection(par=par, df=rev.df, end_filename=end_filename)
            cdir.nodes_directions()

            # SIGNALS COMPUTATION
            sig = reversal_signal.ReversalSignal(par=par, df=cdir.df, end_filename=end_filename)
            sig.compute_polarity_and_nb_neighbors()
            sig.compute_polarity_and_nb_neighbors_angle_view()
            # sig.compute_local_frustration(method='initial')
            sig.compute_cumul_frustration(method='michele')
            print('SAVE DF_REV_SIG...')
            tool.save_df(df=sig.df, path=path_output, filename=input_filename_no_ext+'__DATA_REV_SIG__'+end_filename+'.csv')
