import numpy as np


class Parameters:


    def __init__(self):

        ### PARALLELIZATION ###
        self.n_jobs = 12

        ### MOVIE ###
        self.scale = 1 # in µm/px
        self.tbf = 2/60 # in min/frame

        ### TRACKING ###
        self.track_mate_csv = False
        # In the case where several point have been detected for the bacteria
        # In this case the csv coordinates must be as (0_x,0_y,...,n_x,n_y)
        # Put to 1 if only one point is detect per bacteria
        self.n_nodes = 11
        self.vmax = 30 # µm/min
        self.width = 0.7 # µm
        self.kn = 30 # Number of detected neighbours to then compute the number of neighbour
        
        ### REVERSALS DETECTION ###
        # CSV location
        self.name_folder = ''
        self.name_file = 'file_tracking_rippling_movie_1.csv'
        self.name_file = 'file_tracking_swarming_movie_1.csv'

        self.track_id_column = 'id'
        self.t_column = 'frame'
        # self.x_column = 'x_centroid'
        # self.y_column = 'y_centroid'
        self.x_column = 'x5'
        self.y_column = 'y5'
        self.v_column = 'velocity'
        self.rev_column = 'reversals'
        self.path_tracking = self.name_folder + self.name_file

        self.path_reversals = self.name_folder + 'reversals/'
        self.path_directions = self.name_folder + 'directions/'
        self.path_reversal_signal = self.name_folder + 'reversal_signal/'
        # smoothed parameters
        self.min_size_smoothed_um = 1 # in µm
        # Correction of the parameters in pixels
        self.min_size_smoothed = self.min_size_smoothed_um / self.scale
        self.iteration = 10
        # reversals parameters for their detection
        self.angle_rev = np.pi / 2 # in rad

        ### SAVE ###
        self.end_name_file = '_min_size_smoothed_um=' + str(self.min_size_smoothed_um) + '_um'

        ### REVERSAL SIGNALING ###
        self.max_frustration = 2
        self.frustration_time_memory = 18 / 60 # in minutes
        self.cumul_frustration_decreasing_rate = 1 # in 1 / minutes
        self.angle_view = np.pi / 2

        ### FRUSTRATION ###
        self.path_save_movie_frustration = "movie_frustration/"

