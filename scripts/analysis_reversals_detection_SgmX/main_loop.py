#import numpy own functions
import numpy as np
import pandas as pd
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed


#import my functions
import fluo_detection
import plt_everything
import import_image
import dataframe
import count_images
import settings

# --------------------------MAIN LOOP. ----------------------------------------------------------------------
# Calls fluorescence detection for every selected bacteria. Plots its afterwards-------------------------


def main_loop_fct(settings_dict):
    
    # load paths and constants from settings.
    # general settings
    settings_dict_general = vars(settings.settings_general_fct())
    path_seg_dir = settings_dict_general["path_seg_dir"]
    path_fluo_dir = settings_dict_general["path_fluo_dir"]
    width = settings_dict_general["width"]
    pcf = settings_dict_general["pcf"]

    # specific settings
    path_save_image_dir = settings_dict["path_save_image_dir"]
    path_save_dataframe_dir = settings_dict["path_save_dataframe_dir"]
    plot_single_onoff = settings_dict["plot_single_onoff"]
    plot_final_onoff = settings_dict["plot_final_onoff"]
    show_plot = settings_dict["show_plot"]
    plot_indices = settings_dict["plot_indices"]
    compression = settings_dict["compression"]
    compression_quality = settings_dict["compression_quality"]
    selection_or_all = settings_dict["selection_or_all"]
    frame_end = settings_dict["frame_end"]
    step_frames = settings_dict["step_frames"]
    lead_pole_factor = settings_dict["lead_pole_factor"]
    noise_tol_factor = settings_dict["noise_tol_factor"]
    bord_thresh = settings_dict["bord_thresh"]
    min_skel_length = settings_dict["min_skel_length"]
    fluo_thresh = settings_dict["fluo_thresh"]
    pole_on_thresh = settings_dict["pole_on_thresh"]


    #--------------LOOP OVER ALL FRAMES------------------

    # Initialize an empty dataframe FOR ALL TIMES for bacterial data
    df_fluo = dataframe.create_dataframe_fct()
    df_suppl = dataframe.create_dataframe_supplementary_fct()

    for t in tqdm(range(0, frame_end, step_frames)):

       #  print("Analysis of frame nr. " + str(t) + " started.", flush = True)

        #--------------INITIALIZE IMAGE AND DATA------------------
        # load image data, normalize and denoise. For both it is an x,y - intensity table (ID is the intensity)
        path_seg, path_fluo, label_img_seg, label_img_fluo, mean_noise, sd_noise, mean_bacteria_intensity, sd_bacteria_intensity = import_image.import_denoise_normalize(t = t, path_seg_dir = path_seg_dir, path_fluo_dir = path_fluo_dir)

        # initialize regions. Regions measure all kind of different properties of labeled image regions = bacteria
        regions = regionprops(label_img_seg) #only the segmented image has information about the regions
        regions_fluo = regionprops(label_image=label_img_seg, intensity_image=label_img_fluo) #the fluorescenece is selected on the regions, but as an extra intensity map

        # set dataframe FOR ONE FRAME as global variable and import main loop, which used df
        global df_fluo_t #we declare the dataframe as a global variable, that we can edit in all the subroutines
        global df_suppl_t

        # create a dataframe FOR ONE TIME for the bacterial information
        df_fluo_t = dataframe.initialize_dataframe_fct(length = len(regions_fluo), time = t)
       # df_suppl = dataframe.initialize_dataframe_supplementary_fct(time = t)

        # store supplementary image information
        df_suppl.loc[t,"t"] = t
        df_suppl.loc[t,"mean_noise"] = mean_noise
        df_suppl.loc[t,"sd_noise"] = sd_noise
        df_suppl.loc[t,"path_seg"] = path_seg
        df_suppl.loc[t,"path_fluo"] = path_fluo
        df_suppl.loc[t,"mean_bact_intens"] = mean_bacteria_intensity
        df_suppl.loc[t,"sd_bact_intens"] = sd_bacteria_intensity

        # update pole_on_thresh to counter the decrease in fluorescence intensity
        pole_on_thresh_iteration = pole_on_thresh * df_suppl.loc[t,"mean_bact_intens"] / df_suppl.loc[0,"mean_bact_intens"]

        #--------------LOOP OVER ALL BACTERIA: DETECT FLUO AT POLES-------------------
        # simulate just a selection of bacteria
        if selection_or_all == "selection":
            bact_index_selection = list(map(int, input("Enter Bact Index or Multiple Indices (separate by space bar):  ").strip().split()))
            for bact_index in bact_index_selection:
                df_fluo_t = fluo_detection.fluorescence_detection_fct(label_img_fluo = label_img_fluo, regions_fluo = regions_fluo, bact_index = bact_index, bord_thresh = bord_thresh, min_skel_length = min_skel_length, fluo_thresh = fluo_thresh, lead_pole_factor = lead_pole_factor, noise_tol_factor = noise_tol_factor, pole_on_thresh = pole_on_thresh_iteration, width = width, pcf = pcf, sd_noise = sd_noise, df = df_fluo_t, plot_onoff= plot_single_onoff, time = t)

        # simulate all bacteria
        else: 
            bact_index_selection = np.arange(len(regions_fluo)) #select all
            for bact_index in bact_index_selection:
                df_fluo_t = fluo_detection.fluorescence_detection_fct(label_img_fluo = label_img_fluo, regions_fluo = regions_fluo, bact_index = bact_index, bord_thresh = bord_thresh, min_skel_length = min_skel_length, fluo_thresh = fluo_thresh, lead_pole_factor = lead_pole_factor, noise_tol_factor = noise_tol_factor, pole_on_thresh = pole_on_thresh, width = width, pcf = pcf, sd_noise = sd_noise, df = df_fluo_t, plot_onoff= plot_single_onoff, time = t)

        # Concatenate ONE TIME'S dataframe to ALL TIME dataframe
        df_fluo = pd.concat([df_fluo,df_fluo_t], ignore_index=True)   
      #  df_suppl_all_t = pd.concat([df_suppl_all_t,df_suppl], ignore_index=True) 
      #  print("Analysis of frame nr. " + str(t) + " completed.", flush = True)

        # Save dataframe (in every timestep, to also have a save in case it fails at some frame)
        # df_fluo.to_csv(path_save_dataframe_dir + "fluo_analysis.csv")
        # df_suppl.to_csv(path_save_dataframe_dir + "supplementary.csv")

        #--------------PLOT POLE LOCATIONS (optional)---------------------
        # plot the final image with all bacteria on it
        # can now be done in an extra function
        if plot_final_onoff == "on":
            #plot all the bacteria
            print("Waiting for final plot...", flush = True)
            plt_everything.plot_fluo_analysis_fct(label_img_seg, label_img_fluo, regions_fluo, mean_noise, sd_noise, width, pcf, df = df_fluo_t, path_save_image_dir = path_save_image_dir, time = t, reversals = "no", plot_indices = plot_indices, show_plot = show_plot, compression = compression, compression_quality = compression_quality, mark_selection = "no")
            plt.show()

    # Save dataframe (only at the end, remove comment on the top to save at each time step)        
    df_fluo.to_csv(path_save_dataframe_dir + "fluo_analysis.csv")
    df_suppl.to_csv(path_save_dataframe_dir + "supplementary.csv")
    
    print("\nFluorescence analysis finished!", flush = True)
    return(df_fluo, df_suppl)