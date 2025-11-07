#%%
import pandas as pd
import count_images
import settings
import import_image
import plt_everything
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import sys
import settings 
from tqdm import tqdm

def plt_reversal_analysis_fct():

    # load paths and constants from settings
    settings_dict_general = vars(settings.settings_general_fct())
    settings_dict = vars(settings.settings_plot_reversal_analysis_fct())

    path_seg_dir = settings_dict_general["path_seg_dir"]
    path_fluo_dir = settings_dict_general["path_fluo_dir"]

    path_save_image_dir = settings_dict["path_save_image_dir"]
    path_save_dataframes_dir = settings_dict["path_save_dataframe_dir"]
    
    plot_indices = settings_dict["plot_indices"]
    show_plot = settings_dict["show_plot"]
    compression = settings_dict["compression"]
    compression_quality = settings_dict["compression_quality"]
    frame_end = settings_dict["frame_end"]
    step_frames = settings_dict["step_frames"]

    width = settings_dict_general["width"]
    pcf = settings_dict_general["pcf"]

    # import reversal analysis data
    path_merged_df = path_save_dataframes_dir + "merged_df.csv"
    merged_df = pd.read_csv(path_merged_df, index_col = 0)

    # loop to plot all images
    for t in tqdm(range(0, frame_end, step_frames)):

        #sys.stdout.write("\rt = " + str(t) + ". Waiting for plot...   ")

        # select all indices at time t.
        cond_t = merged_df.loc[:,"t"] == t

        # also reset the sub-table index, so that the new index goes according to the track_ids (at least partially - not always a track id exists and thus the track id can differ slightly by the table index)
        df_t = merged_df.loc[cond_t,:].reset_index(drop = True)

        # load image data, normalize and denoise. For both it is an x,y - intensity table (ID is the intensity)
        path_seg, path_fluo, label_img_seg, label_img_fluo, mean_noise, sd_noise, mean_bacteria_intensity, sd_bacteria_intensity = import_image.import_denoise_normalize(t = t, path_seg_dir = path_seg_dir, path_fluo_dir = path_fluo_dir)
        
        # load the data that is usually provided from the other functions 
        regions_fluo = regionprops(label_image=label_img_seg, intensity_image=label_img_fluo)

        # plot for every time
        plt_everything.plot_fluo_analysis_fct(label_img_seg, label_img_fluo, regions_fluo, mean_noise, sd_noise, width, pcf, df_t, path_save_image_dir, time = t, reversals = "yes", plot_indices = plot_indices, show_plot = show_plot, compression = compression, compression_quality = compression_quality, mark_selection = "no")
    print("\nReversal plot finished!")
