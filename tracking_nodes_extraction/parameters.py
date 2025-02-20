import numpy as np


class Parameters:


    def __init__(self):
        
        # Add the path where the segmented images are
        self.name_folder_seg_movie = ""
        # Folder of the trackmate csv
        self.name_folder_csv = ""
        # Name of the trackmate csv
        self.name_file_csv = ''

        self.frame_gap = 15 # frames
        self.n_nodes = 11
        self.tbf = 2 / 60 # min / frame
        # Scale of the movie in µm / pixels
        self.scale = 0.0646028 # µm / px
        self.min_size_bacteria = 2 # in µm
        self.max_velocity = 15 # µm / min
        self.track_id_column = 'TRACK_ID'
        self.t_column = 'FRAME'
        self.seg_id_column = 'MAX_INTENSITY_CH1'
        self.area_column = 'AREA'
        self.x_centroid_column = 'POSITION_X'
        self.y_centroid_column = 'POSITION_Y'
        # Set to False if the tracking csv is not generated by trackmate
        self.track_mate_csv = True
