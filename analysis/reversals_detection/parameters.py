import numpy as np


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
        ## csv path
        # mac
        self.path_input_folders = [
            '/Volumes/dock_ssd/DATA/a_thesis_2019/codes/python/my_libraries_mx/scripts/output/tracking/tracking_paper_review_2025/rippling_movie_1/',
            '/Volumes/dock_ssd/DATA/a_thesis_2019/codes/python/my_libraries_mx/scripts/output/tracking/tracking_paper_review_2025/rippling_movie_2/',
            '/Volumes/dock_ssd/DATA/a_thesis_2019/codes/python/my_libraries_mx/scripts/output/tracking/tracking_paper_review_2025/rippling_movie_3/',
            '/Volumes/dock_ssd/DATA/a_thesis_2019/codes/python/my_libraries_mx/scripts/output/tracking/tracking_paper_review_2025/swarming_movie_1/',
            '/Volumes/dock_ssd/DATA/a_thesis_2019/codes/python/my_libraries_mx/scripts/output/tracking/tracking_paper_review_2025/swarming_movie_2/',
            '/Volumes/dock_ssd/DATA/a_thesis_2019/codes/python/my_libraries_mx/scripts/output/tracking/tracking_paper_review_2025/swarming_movie_3/',
            '/Volumes/dock_ssd/DATA/a_thesis_2019/codes/python/my_libraries_mx/scripts/output/tracking/tracking_paper_review_2025/swarming_movie_4/',
            '/Volumes/dock_ssd/DATA/a_thesis_2019/codes/python/my_libraries_mx/scripts/output/tracking/tracking_paper_review_2025/swarming_movie_5/',
        ]
        self.input_filenames = [
            'rippling_movie_1_tracking.csv',
            'rippling_movie_2_tracking.csv',
            'rippling_movie_3_tracking.csv',
            'swarming_movie_1_tracking.csv',
            'swarming_movie_2_tracking.csv',
            'swarming_movie_3_tracking.csv',
            'swarming_movie_4_tracking.csv',
            'swarming_movie_5_tracking.csv',
        ]
        self.path_output_folder = '/Volumes/dock_ssd/DATA/a_thesis_2019/codes/python/my_libraries_mx/scripts/output/tracking/reversal_analysis_paper_review_2025'

        # csv column names
        self.track_id_column = 'id'
        self.t_column = 'frame'
        # self.x_column = 'x_centroid'
        # self.y_column = 'y_centroid'
        self.x_column = 'x5'
        self.y_column = 'y5'

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

