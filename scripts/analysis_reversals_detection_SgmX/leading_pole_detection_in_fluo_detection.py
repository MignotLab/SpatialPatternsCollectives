import numpy as np
import error_message


def detect_lead_pole_fct(df, mean_intensity_pole_1, mean_intensity_pole_2, pole_1_n, pole_2_n, lead_pole_factor, noise_tol_factor, pole_on_thresh, sd_noise, bact_index):
    
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