import numpy as np
import error_message
import pandas as pd
import count_images
from tqdm import tqdm
import dataframe
import settings



def detect_lead_pole_fct(df, mean_intensity_pole_1, mean_intensity_pole_2, pole_1_n, pole_2_n, lead_pole_factor, noise_tol_factor, pole_on_thresh, sd_noise, bact_index, iteration):
    
    # Only for the leading pole determination, negative pole intensities are treated as zeros.
    # This is not applied to the table, since otherwise the maximum likelihood estimation of the pole distributions
    # (in the step of the parameter estimation) does not work anymore --> the pole_on_thresh does not converge
    # It does not work anymore because around a third of the mean intensities are negative and thus have a great impact shaping the distribution.
    if mean_intensity_pole_1 < 0:
        mean_intensity_pole_1 = 0
    if mean_intensity_pole_2 < 0:
        mean_intensity_pole_2 = 0

    #--------------DECIDE THE LEADING POLE AND ITS CERTAINTY--------------------------------
    # pole 1 leading pole if 1. it is lpf times brighter than the other pole and 2. if the pole difference is bigger than tol * sd_noise. We talk about the sd of the mean of the calculation square. 3. it has to be bigger than a certain threshold
    if (abs(mean_intensity_pole_1) > abs(lead_pole_factor * mean_intensity_pole_2)) and (abs(mean_intensity_pole_1 - mean_intensity_pole_2)) > noise_tol_factor*sd_noise/np.sqrt(pole_1_n) and abs(mean_intensity_pole_1) > pole_on_thresh:
        df.loc[bact_index,"leading_pole"] = 1 

    # pole 2 is bigger
    elif (abs(mean_intensity_pole_2) > abs(lead_pole_factor * mean_intensity_pole_1)) and abs(mean_intensity_pole_2 - mean_intensity_pole_1) > noise_tol_factor*sd_noise/np.sqrt(pole_2_n) and abs(mean_intensity_pole_2) > pole_on_thresh:
        df.loc[bact_index,"leading_pole"] = 2


    # there is no leading pole
    else:

        # either the pole difference (factor) is too small
        df.loc[bact_index,"leading_pole"] = np.nan
        if np.maximum( abs(mean_intensity_pole_1) , abs(mean_intensity_pole_2) ) <= lead_pole_factor * np.minimum( mean_intensity_pole_1 , mean_intensity_pole_2 ):
           # print("\nWarning: Pole difference generally too small\n")
            error_message.create_error_msg_fct(df, bact_index, "small pole dif", affected_poles= [1,2])

        # or the noise dominates
        elif (abs(mean_intensity_pole_1 - mean_intensity_pole_2) < noise_tol_factor*sd_noise/np.sqrt(pole_1_n)) or (abs(mean_intensity_pole_2 - mean_intensity_pole_1) < noise_tol_factor*sd_noise/np.sqrt(pole_2_n)):
           # print("\nWarning: Pole difference small compared to tol_factor * noise mean standard deviation\n")
            error_message.create_error_msg_fct(df, bact_index, "noise dominates", affected_poles= [1,2])

        # or the minimum threshold for being a leading pole is not hit
        elif abs(mean_intensity_pole_1) <= pole_on_thresh or abs(mean_intensity_pole_2) <= pole_on_thresh:
          #  print("\nWarning: No pole is above the leading pole threshold\n")
            error_message.create_error_msg_fct(df, bact_index, "no lead pole thresh hit", affected_poles= [1,2])

    #-----------------------CREATE FURTHER WARNING MESSAGES--------------------------------
    # Create further warning messages if both poles are on

    # both poles off
    if mean_intensity_pole_1 <= pole_on_thresh and mean_intensity_pole_2 <= pole_on_thresh:
       # print("Warning: No pole is definitely on")
        error_message.create_error_msg_fct(df, bact_index, "poles off", affected_poles= [1,2])
    
    # both pole on
    elif mean_intensity_pole_1 > pole_on_thresh and mean_intensity_pole_2 > pole_on_thresh:
       # print("Warning: Both poles are on --> probably pollution")
        error_message.create_error_msg_fct(df, bact_index, "poles on", affected_poles= [1,2])


def leading_pole_detection_fct(settings_dict, iteration):
    
    #--------------IMPORT SETTINGS--------------------------------
    # general settings
    settings_general_dict = vars(settings.settings_general_fct())
    path_seg_dir = settings_general_dict["path_seg_dir"]
    path_fluo_dir = settings_general_dict["path_fluo_dir"]

    # specific settings
    frame_end = settings_dict["frame_end"]
    step_frames = settings_dict["step_frames"]
    lead_pole_factor = settings_dict["lead_pole_factor"]
    noise_tol_factor = settings_dict["noise_tol_factor"]
    pole_on_thresh = settings_dict["pole_on_thresh"]
    path_save_dataframe_dir = settings_dict["path_save_dataframe_dir"]

    #--------------IMPORT FLUORESCENCE ANALYSIS DATAFRAME--------------------------------
    # first leading pole detection on original analysis vs second one on corrected analysis
    if iteration == 0:
        path_analysis = path_save_dataframe_dir + "fluo_analysis.csv"
    else:
        path_analysis = path_save_dataframe_dir + "fluo_analysis_corrected.csv"
    df_fluo = pd.read_csv(path_analysis, index_col = 0)
    path_analysis_suppl = path_save_dataframe_dir + "supplementary.csv"
    df_suppl = pd.read_csv(path_analysis_suppl, index_col = 0)


    # remove old warning that are not valid anymore
    if iteration != 0:
        # since we are now in a new iteration where the parameters changed:
        # remove all warnings/remarks related to the leading pole detection.
        # because these warning could repeat when iterating it multiple times
        # or they could become obsolete, e.g. "pole thresh not hit" would not
        # be valid if the thresh was lowered with the new parameters
        cond_warning = (df_fluo["error_message_pole_1"].str.contains("Warning") == True) | (df_fluo["error_message_pole_1"].str.contains("Remark") == True) # as you can see in the error_msg file, only the entries with warning or remark are related to the leadgin pole detection error messages
        df_fluo.loc[cond_warning,"error_message_pole_1"] = "no error"
        df_fluo.loc[cond_warning,"error_message_pole_2"] = "no error"
 
    #--------------START LOOPING OVER T AND ALL BACTERIA--------------------------------
    # only extract of times or all times?
    if frame_end == "all":
        # Count number of images in the directory
        frame_end = count_images.count_and_prepare_images_fct(path_seg_dir, path_fluo_dir)
        # Else: Loop until specified timeframe

    # time loop
    for t in tqdm(range(0, frame_end, step_frames)):
        cond_t = df_fluo.loc[:,"t"] == t

        # extract analysis_t, the dataframe "analysis" for a single timeframe and reset indices
        global df_fluo_t
        df_fluo_t = df_fluo.loc[cond_t,:]
        df_fluo_t = df_fluo_t.loc[:, ~df_fluo_t.columns.str.contains('^Unnamed')] #drop unnamed column
        df_fluo_t = df_fluo_t.reset_index(drop = True)

        cond_t_suppl = df_suppl.loc[:,"t"] == t
        sd_noise = df_suppl.loc[cond_t_suppl,"sd_noise"].values[0]

        # loop over all bacteria
        for bact_index in range(np.max(df_fluo_t.loc[:,"seg_id"].values)):

            # extract leading pole detection parameters
            mean_intensity_pole_1 = df_fluo_t.loc[bact_index, "mean_intensity_pole_1"]
            mean_intensity_pole_2 = df_fluo_t.loc[bact_index, "mean_intensity_pole_2"]
            pole_1_n = df_fluo_t.loc[bact_index, "pole_1_n"]
            pole_2_n = df_fluo_t.loc[bact_index, "pole_2_n"]

            #--------------LEADING POLE DETECTION STARTS HERE--------------------------------
            # if both poles have mean intensities calculated
            if np.isnan(mean_intensity_pole_1)== False  and np.isnan(mean_intensity_pole_2) == False:
                detect_lead_pole_fct(df_fluo_t, mean_intensity_pole_1, mean_intensity_pole_2, pole_1_n, pole_2_n, lead_pole_factor, noise_tol_factor, pole_on_thresh, sd_noise, bact_index, iteration)
        
        # overwrite the dataframe with new leading poles, error messages    
        # it is important to put a .values on the right side, otherwise pandas will be confused by the indices
        df_fluo.loc[cond_t,["error_message_pole_1","error_message_pole_2","leading_pole"]] = df_fluo_t.loc[:,["error_message_pole_1","error_message_pole_2","leading_pole"]].values

    # save it
    if iteration == 0:
        df_fluo.to_csv(path_save_dataframe_dir + "fluo_analysis.csv")
    else:
        df_fluo.to_csv(path_save_dataframe_dir + "fluo_analysis_corrected.csv")

    return(df_fluo)

