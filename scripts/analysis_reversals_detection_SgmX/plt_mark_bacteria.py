#%%
import pandas as pd
import count_images
import settings
import import_image
import plt_everything
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import settings
from tqdm import tqdm

# the second input argument is only important if the "compare correction"
# function calls this plotting function
def plt_mark_bacteria_fct(df_with_selection_information, before_or_after_correction, iteration = None): #the standard case is nothing
    
    # load paths and constants from settings 
    settings_general = vars(settings.settings_general_fct())
    path_seg_dir = settings_general["path_seg_dir"]
    path_fluo_dir = settings_general["path_fluo_dir"]
    width = settings_general["width"]
    pcf = settings_general["pcf"]

    settings_fluo_plot = vars(settings.settings_plot_fluorescence_detection_fct())
    path_save_image_dir = settings_fluo_plot["path_save_image_dir"]

    plot_indices = settings_fluo_plot["plot_indices"]
    show_plot = settings_fluo_plot["show_plot"]
    compression = settings_fluo_plot["compression"]
    compression_quality = settings_fluo_plot["compression_quality"]
    frame_end = settings_fluo_plot["frame_end"]
    step_frames = settings_fluo_plot["step_frames"]

    # loop to plot all images
    for t in tqdm(range(0, frame_end + 1, step_frames)):

        # print("Plot of frame nr. " + str(t) + " started.", flush = True)

        # select all indices at time t.
        cond_t = df_with_selection_information.loc[:,"t"] == t

        # also reset the sub-table index, so that the new index goes according to the track_ids (at least partially - not always a track id exists and thus the track id can differ slightly by the table index)
        df_t = df_with_selection_information.loc[cond_t,:].reset_index(drop = True)

        # load image data, normalize and denoise. For both it is an x,y - intensity table (ID is the intensity)
        path_seg, path_fluo, label_img_seg, label_img_fluo, mean_noise, sd_noise, mean_bacteria_intensity, sd_bacteria_intensity = import_image.import_denoise_normalize(t = t, path_seg_dir = path_seg_dir, path_fluo_dir = path_fluo_dir)
        
        # load the data that is usually provided from the other functions 
        regions_fluo = regionprops(label_image=label_img_seg, intensity_image=label_img_fluo)

        # plot for every time
        plt_everything.plot_fluo_analysis_fct(label_img_seg, label_img_fluo, regions_fluo, mean_noise, sd_noise, width, pcf, df_t, path_save_image_dir, time = t, reversals = "no", plot_indices = plot_indices, show_plot = show_plot, compression = compression, compression_quality = compression_quality, mark_selection = "yes" + before_or_after_correction, iteration = iteration)
        # print("Plot of frame nr. " + str(t) + " completed.", flush = True)

    print("\nFluorescence plot finished!")

    # %%
