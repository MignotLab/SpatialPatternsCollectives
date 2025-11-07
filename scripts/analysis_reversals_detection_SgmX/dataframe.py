#%%
import numpy as np
import pandas as pd

# create an empty dataframe
def create_dataframe_fct():
    dataframe = pd.DataFrame(columns=("t","seg_id","mean_intensity_pole_1","mean_intensity_pole_2","error_message_pole_1","error_message_pole_2","pole_1_n", "pole_2_n","leading_pole","center_x_local","center_y_local","pole_1_x_local","pole_1_y_local","pole_2_x_local","pole_2_y_local","x_min_global","y_min_global","x_max_global","y_max_global",))
    return(dataframe)

# initialize a dataframe given a time and number of segmented bacteria
def initialize_dataframe_fct(length, time):
    # initialize a dataframe like before    
    dataframe = pd.DataFrame(columns=("t","seg_id","mean_intensity_pole_1","mean_intensity_pole_2","error_message_pole_1","error_message_pole_2","pole_1_n", "pole_2_n","leading_pole","center_x_local","center_y_local","pole_1_x_local","pole_1_y_local","pole_2_x_local","pole_2_y_local","x_min_global","y_min_global","x_max_global","y_max_global"))
    time = time * np.ones(length, dtype= int) #we observe only t = 0
    indices = np.arange(length, dtype = int) #we can always initialize all segmentation ids
    errors = np.repeat("no error",length)
    dataframe.loc[:,"t"] = time
    dataframe.loc[:,"seg_id"] = indices
    dataframe.loc[:,"error_message_pole_1"] = errors
    dataframe.loc[:,"error_message_pole_2"] = errors
    
    return(dataframe)

# create a dataframe giving supplementary information for every frame
def create_dataframe_supplementary_fct():
    dataframe = pd.DataFrame(columns=("t", "mean_noise", "sd_noise","mean_bact_intens","sd_bact_intens","path_seg","path_fluo"))

    return(dataframe)

# initialize the supplementary information given a time
def initialize_dataframe_supplementary_fct(time):
    dataframe = pd.DataFrame(columns=("t", "mean_noise", "sd_noise","mean_bact_intens","sd_bact_intens","path_seg","path_fluo"))
    dataframe.loc[:,"t"] = time
    dataframe.loc[:,"mean_noise"] = np.nan
    dataframe.loc[:,"sd_noise"] = np.nan

    return(dataframe)
