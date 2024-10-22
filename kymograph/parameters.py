import numpy as np


class Parameters:


    def __init__(self):

        ### PARALLELIZATION ###
        self.n_jobs = 12

        ### MOVIE ###
        # self.scale = 0.0646028 # in µm/px
        self.scale = 1 # in µm/px # simu
        # self.tbf = 2/60 # in min/frame
        self.tbf = 1.8/60 # in min/frame  # simu

        ### TRACKING ###
        # In the case where several point have been detected for the bacteria
        # In this case the csv coordinates must be as (0_x,0_y,...,n_x,n_y)
        # Put to 1 if only one point is detect per bacteria
        # self.n_nodes = 11
        self.n_nodes = 10 # simu
        self.width = 0.7 # µm
        
        ### REVERSALS DETECTION ###
        # csv
        self.track_id_column = 'id'
        self.t_column = 'frame'
        # self.x_column = 'x_centroid'
        # self.y_column = 'y_centroid'
        # self.len_traj_column = 'traj_length'
        self.x_column = 'x4' # simu
        self.y_column = 'y4' # simu
        self.len_traj_column = 'x5' # simu
        self.rev_column = 'reversal'


