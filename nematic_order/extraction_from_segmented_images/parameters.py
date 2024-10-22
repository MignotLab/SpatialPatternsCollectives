import numpy as np


class Parameters:


    def __init__(self):

        ### MOVIE ###
        self.scale = 0.0646028 # in µm/px
        # self.scale = 1 # in µm/px
        self.first_frame = 0 # Frame from which we begin to compute the nematic order (usefull to take the end of a simulation for example)
        self.tbf = 2/60 # minutes

        ### TRACKING FILE ###
        # A minimum of three detected points (nodes) are required
        # i.e the head, the middle and the tail
        self.n_nodes = 11

        self.path_folder = "folder_name/"
        self.file_name = "filename.csv"
        self.path_save = 'results/dataset/'
        # csv
        self.track_id_column = 'id'
        self.t_column = 'frame'
        self.x_column = 'x5'
        self.y_column = 'y5'
        self.x_column_first = 'x0'
        self.y_column_first = 'y0'
        self.x_column_last = 'x10'
        self.y_column_last = 'y10'

        ### COMPUTATION PARAMETERS ###
        self.space_size = 65 # µm
        self.dr = 10
        self.min_shell_neighbours = 3
        self.interval_frames = 500


