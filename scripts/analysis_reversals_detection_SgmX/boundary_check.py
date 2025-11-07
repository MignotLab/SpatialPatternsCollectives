import numpy as np
import error_message


# algorithm to check if a bacteria lies at the boundary, i.e. has a pixel that is "bord_thresh"- pixels away from the image boundary


def boundary_check_fct(bact_index, bord_thresh, regions_fluo, label_img_fluo, df):
    
    # find the bacterial borders
    x_min = df.loc[bact_index,"x_min_global"]
    x_max = df.loc[bact_index,"x_max_global"]
    y_min = df.loc[bact_index,"y_min_global"]
    y_max = df.loc[bact_index,"y_max_global"]

    # --> exclude bacterias at the boundary
    if x_min < 0 + bord_thresh or x_max >= np.shape(label_img_fluo)[1] - bord_thresh or y_min < 0 + bord_thresh or y_max >= np.shape(label_img_fluo)[0] - bord_thresh:
        at_boundary = "yes"
        error_message.create_error_msg_fct(df, bact_index, "boundary", affected_poles= [1,2])
        df.loc[bact_index,"mean_intensity_pole_1"] = np.nan
        df.loc[bact_index,"mean_intensity_pole_2"] = np.nan

    else:
        at_boundary = "no"     

    return(at_boundary)
# %%
