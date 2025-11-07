import pandas as pd
import numpy as np
import time
import sys
import settings
import error_message
from tqdm import tqdm

def find_reversal_pairs(list_of_zeros_ones, distance, track_id): # 
    """
    input: 
    list_of_zeros_ones: The list of the dataframe containing the information for every frame if a reversal is detected or not
    distance: Takes the distance in frame_step_size * frames. For example since the reversal analysis was only conducted every 5 frames, a distance of 1 means a distance of 5 frames.
    track_id: When looking at the list of reversal events, we have to make sure to always be at the same track id when spotting for multiple reversals.
    
    example of function:
    track_id: 1             2
    list:     10100000000000000100000
    --> two reversals happening in a short time for track id 1!
    """
    
    # create an empty array of indices. We will memorize all indices (t of reversal in timeline) that have a close reversal. If we have 3 close reversals, we want to keep 1, so this list wont contain the third one. 
    indices = np.array([])
    # loop over all the reversal indices in the total list
    # for i in range(distance, len(list_of_zeros_ones) - distance):
    for i in range(0,len(list_of_zeros_ones)):

        # check if there is a reversal somewhere and that we have not treated it before (they are already treated as a pair - to prevent that a third one, which we want to keep, is participating in two pairs --> 3 indices removed)
        if (list_of_zeros_ones[i] == 1) and (i not in indices):
            
            # Check the following indices within a range of "distance" places, if there is a reversal on the same track id
            indices_left = len(list_of_zeros_ones) - 1 - i #check how many indices are left to go to on the right
            for j in range( i + np.minimum(1, indices_left + 1) , i + np.minimum(distance + 1, indices_left + 1) ): # if we are already at the end of the array, dont look for too much to the right to not leave the array
                if (list_of_zeros_ones[j] == 1) & (track_id[j] == track_id[i]):

                    # If this is true, a close SECOND reversal is found.
                    # We memorize the indices, just to know that they have been treated already.
                    # break and look for the next neighbourhood, to prevent that a third reversal could be detected and removed, since we always want to keep the third reversal.
                    indices = np.append(indices, i)
                    indices = np.append(indices, j)
                    break 

    # the list of indices is to be removed from the complete reversal list later in the main reversal_analysis function.
    indices = indices.astype(int)
    return indices

def reversal_analysis_fct():

    # ---------------IMPORT AND MERGE TRACKING AND ANALYSIS DATA-------------------
    # = Add the tracking information to the fluorescence analysis

    sys.stdout.write("\nImport and merge tracking and analysis data... ")

    #df_t.to_csv(path_save_dir + "analysis.csv")
    # specify paths of dataframes
    path_tracking = vars(settings.settings_general_fct())["path_tracking"]
    path_analysis = vars(settings.settings_reversal_analysis_fct())["path_save_dataframe_dir"] + "fluo_analysis_corrected.csv"
    path_save_dataframes_dir = vars(settings.settings_reversal_analysis_fct())["path_save_dataframe_dir"]
    forbidden_frame_distance_factor = vars(settings.settings_reversal_analysis_fct())["forbidden_frame_distance_factor"]
    choice_filter_out_short_reversals = vars(settings.settings_reversal_analysis_fct())["choice_filter_out_short_reversals"]
    analysis = pd.read_csv(path_analysis, index_col = 0)
    tracking = pd.read_csv(path_tracking)

    # extract only meaningful information form the tracking
    tracking = tracking.iloc[3:,2:].astype(float)
    tracking = tracking.loc[:,["TRACK_ID","FRAME","MAX_INTENSITY_CH1"]]

    # rename columns
    tracking = tracking.rename(columns = {"TRACK_ID": "track_id", "FRAME": "t", "MAX_INTENSITY_CH1": "seg_id"})
    
    # set track count 1...N+1 to 0...N
    tracking["seg_id"] = tracking["seg_id"]-1

    # merge fluorescence analysis and tracking dataframe
    merged_df = tracking.merge(analysis, on = ["t", "seg_id"])

    # sort by firstly id and the time
    #merged_df.sort_values(by = ["seg_id", "t"], ignore_index = True, inplace = True)
    merged_df.loc[:,"t"] = merged_df.loc[:,"t"].astype(int)
    merged_df.loc[:,"seg_id"] = merged_df.loc[:,"seg_id"].astype(int)
    merged_df = merged_df.sort_values(by=["track_id","t"])

    # to not keep the old indices (which follow the segmentation ids and are now completely reshuffled)
    # we drop the index. 
    # this is also helpful because: we will later drop all nan entries. This leads to some seg ids being completely dropped --> table id would say 307 but in fact, since 2 seg ids were dropped, the index would be 309.
    merged_df = merged_df.reset_index(drop = True)

    # introduce two new columns for reversal analysis
    insert_position = merged_df.columns.get_loc("leading_pole") + 1 # insert next to leading pole column
    merged_df.insert(loc = insert_position, column = "reversal", value = np.nan)
    merged_df.insert(loc = insert_position + 1, column = "reversal_error", value = "no error")

    # reduce our analysis to the tracking ids that are not nan
    # find all tracking ids that are not nan
    merged_df = merged_df.dropna(subset = ["track_id"])
    merged_df.loc[:,"track_id"] = merged_df.loc[:,"track_id"].astype(int) #now we can also convert it to int

    # ---------------POLE REORDERING--------------------------
    # loop through all the track ids

    sys.stdout.write("\nReorder Poles... ")
    time.sleep(2)

    max_track_id = np.max(merged_df.loc[:,"track_id"].values)
    for track_index in tqdm(range(max_track_id), position = 0, leave = True):

        #print("\rReorder Poles - track_index: " + str(track_index) + "/" + str(max_track_id) + "  ", end = "", flush = True)

        # select all data for this track id.
        cond_track_id = merged_df.loc[:,"track_id"] == track_index

        # extract all available times (sometimes a time is not available. This happens if two bacteria merge in their segmentation and thus a track id has to be thrown away. In the next timestep, they might be together again)
        times = merged_df.loc[cond_track_id,"t"].values
        n_times = len(times)

        # select all coordinates of the poles:
        x_min_global = merged_df.loc[cond_track_id, "x_min_global"].values
        y_min_global = merged_df.loc[cond_track_id, "y_min_global"].values

        # global coordinates for calculation
        pole_1_x_global = merged_df.loc[cond_track_id, "pole_1_x_local"].values + x_min_global
        pole_1_y_global = merged_df.loc[cond_track_id, "pole_1_y_local"].values + y_min_global
        pole_2_x_global = merged_df.loc[cond_track_id, "pole_2_x_local"].values + x_min_global
        pole_2_y_global = merged_df.loc[cond_track_id, "pole_2_y_local"].values + y_min_global
        # local coordinates which should be changed actually
        pole_1_x_local = merged_df.loc[cond_track_id, "pole_1_x_local"].values
        pole_1_y_local = merged_df.loc[cond_track_id, "pole_1_y_local"].values
        pole_2_x_local = merged_df.loc[cond_track_id, "pole_2_x_local"].values
        pole_2_y_local = merged_df.loc[cond_track_id, "pole_2_y_local"].values
        x_array_local = np.concatenate([pole_1_x_local, pole_2_x_local]).reshape(2,n_times) #first line: pole 1, 2nd line: pole 2
        y_array_local = np.concatenate([pole_1_y_local, pole_2_y_local]).reshape(2,n_times)

        # select more properties that are to be flipped
        mean_intensity_array = merged_df.loc[cond_track_id, ["mean_intensity_pole_1","mean_intensity_pole_2"]].values
        error_msg_array = merged_df.loc[cond_track_id, ["error_message_pole_1","error_message_pole_2"]].values
        leading_pole_1st_line = merged_df.loc[cond_track_id, "leading_pole"].values
        # now create a virtual 2nd column in order to flip
        leading_pole_2nd_line = np.empty(n_times)
        leading_pole_2nd_line[:] = np.nan
        for i in range(len(leading_pole_1st_line)):
            if leading_pole_1st_line[i] == 2:
                leading_pole_2nd_line[i] = 1
            if leading_pole_1st_line[i] == 1:
                leading_pole_2nd_line[i] = 2
        leading_pole_array = np.concatenate([leading_pole_1st_line, leading_pole_2nd_line]).reshape(2,n_times)  
        

        # loop through all the available times
        for i in range(n_times-1):

            # calculate two exemplaric distances.
            distance_normal = np.linalg.norm([pole_1_x_global[i] - pole_1_x_global[i+1], pole_1_y_global[i] - pole_1_y_global[i+1]])
            distance_cross = np.linalg.norm([pole_1_x_global[i] - pole_2_x_global[i+1], pole_1_y_global[i] - pole_2_y_global[i+1]])

            # if the dist( p1(t+1) , p1(t) ) >  dist( p2(t+1) , p1(t) ) - flip the poles
            # i.e. if suddendly pole 2 is closer to pole 1 from one timestep to the next
            if distance_normal > distance_cross: 

                x_array_local[:,i+1:] = np.flip(x_array_local[:,i+1:],axis=0)
                y_array_local[:,i+1:] = np.flip(y_array_local[:,i+1:],axis=0)
                mean_intensity_array[i+1:,:] = np.flip(mean_intensity_array[i+1:,:],axis=1)
                error_msg_array[i+1:,:] = np.flip(error_msg_array[i+1:,:],axis=1)
                leading_pole_array[:,i+1:] = np.flip(leading_pole_array[:,i+1:],axis=0)

        # write flipped information into the dataframe
        merged_df.loc[cond_track_id,["pole_1_x_local","pole_2_x_local"]] = x_array_local.T.copy()
        merged_df.loc[cond_track_id,["pole_1_y_local","pole_2_y_local"]] = y_array_local.T.copy()
        merged_df.loc[cond_track_id,["mean_intensity_pole_1","mean_intensity_pole_2"]] = mean_intensity_array.copy()
        merged_df.loc[cond_track_id,["error_message_pole_1","error_message_pole_2"]] = error_msg_array.copy()
        merged_df.loc[cond_track_id,["leading_pole"]] = leading_pole_array[0].copy()

    # ---------------TRANSFORM TABLE-------------------
    # correct the nans in the table to probable leading pole values
    # idea :
            # lead pole
            # nan   --> 2
            # 2         2
            # nan       2
            # nan       2
            # 1         1
            # 1         1

    sys.stdout.write("\nTransform Pole nans... ")

    # # loop over all the track ids
    # for track_id in tqdm(range(np.max(merged_df.loc[:,"track_id"].values))):

    #     cond_track_id = merged_df.loc[:, "track_id"] == track_id

    #     # ignore all the track ids where it's only nans 
    #     if np.sum(np.isnan(merged_df.loc[cond_track_id, "leading_pole"].values) == False) != 0:

    #         # select leading poles for this track id
    #         selection_leading_pole = merged_df.loc[cond_track_id, "leading_pole"].values

    #         # find where the non-nan values are
    #         notnan_index_list = np.where( np.isnan(selection_leading_pole) == False )[0]
    #         nan_index_list = np.where( np.isnan(selection_leading_pole) == True )[0]

    #         # replace the nans by the previous non nan value:
    #         for i in range(len(notnan_index_list)-1):
                
    #             # find the first notnan index which is bigger than the current nan index
    #             start_index = notnan_index_list[i] 

    #             replacing_value = selection_leading_pole[start_index]
                
    #             if len(notnan_index_list) != 1:
    #                 end_index = notnan_index_list[i+1]  # in the normal case, the end index is just the next notnan index
    #             else: 
    #                 end_index = notnan_index_list[i] + 1 # in this case, there exists no 2nd (=end) value, so we pick the next one. Does not change anything in this case
                
    #             selection_leading_pole[start_index:end_index] = replacing_value

    #         # this loop does not respect the first and the last nan indices.
    #         # the last entries are filled subsequently
    #         if len(notnan_index_list)-1 == 0:
    #             start_index = notnan_index_list[0] 
    #             replacing_value = selection_leading_pole[start_index]

    #         selection_leading_pole[end_index:] = replacing_value

    #         # for the first nan indices, there is no previous non-nan. So we give them just the values of the first non nan
    #         if 0 not in notnan_index_list:
    #             selection_leading_pole[0:notnan_index_list[0]] = selection_leading_pole[notnan_index_list[0]]

    #         # fill it into the table
    #         merged_df.loc[cond_track_id, "leading_pole"] = selection_leading_pole
        
    # loop over all the track ids
    track_ids = np.unique(merged_df.loc[:, 'track_id'])
    # for track_id in tqdm(range(np.max(merged_df.loc[:,"track_id"].values))):
    for track_id in tqdm(track_ids):

        cond_track_id = merged_df.loc[:, "track_id"] == track_id

        # ignore all the track ids where it's only nans 
        if np.sum(np.isnan(merged_df.loc[cond_track_id, "leading_pole"].values) == False) != 0:

            # select leading poles for this track id for all times
            selection_leading_pole = merged_df.loc[cond_track_id, "leading_pole"].values
            series = pd.Series(selection_leading_pole)

            # fill the list forward with the last valid observation 
            series_filled = series.fillna(method='ffill')

            # if we have not got any last valid observation then just take the next valid observation
            series_filled = series_filled.fillna(method='bfill')

            # fill it into the table
            merged_df.loc[cond_track_id, "leading_pole"] = series_filled.tolist()




    # ---------------DETECT REVERSALS-------------------
    # finally use the numbers in the table to say if an reversal occured or not

    # idea :
            # lead pole
            # 2
            # 2
            # 2
            # 1   --> Reversal

    # it is a bit more complex, since 


    print("\nDetect reversals... ")

    # initialize arrays to not work in the table
    leading_pole = merged_df.loc[:,"leading_pole"].values
    track_id = merged_df.loc[:,"track_id"].values

    # check if the leading pole changes from one frame to another
    reversing_value = np.full(len(leading_pole), False)
    reversing_value[1:] = (leading_pole[1:] - leading_pole[0:-1]) != 0

    # since this rule also detects nan-nan as nonzero, we have to set those buggy places to 0
    error_indices = np.where( (np.isnan(leading_pole[1:]) == True) | (np.isnan(leading_pole[:-1]) == True) )[0]
    reversing_value[error_indices] = 0

    # consider that we only have to look within one trajectory.
    # so see where the transition from one trajectory to the next is not
    no_traj_change = np.full(len(leading_pole), False)
    no_traj_change[1:] = (track_id[1:] - track_id[0:-1]) == 0

    # a reversal takes place if the leading pole changes and if the trajectory does not change (within one trajectory) 
    reversal_list = np.zeros(len(reversing_value))
    reversal_list[reversing_value & no_traj_change] = 1

    # we will ignore all pair of two reversals, that happen between 30 seconds (15 frames here):
    # pro: strong pollution for 5 or 10 frames leads to a wrong leading pole detection for a short time
    # --> two reversals result, since the indicator jumps to a wrong pole and back
    # these reversals are supressed. They were around 2.5% of the detected reversals.
    # contra: short reversals are forbidden by definition. But we have not seen short ones yet.
    if choice_filter_out_short_reversals == True:
        reversal_pair_inidces = find_reversal_pairs(reversal_list, forbidden_frame_distance_factor, track_id)
        reversal_list[reversal_pair_inidces] = 0

    # save the list in the dataframe
    merged_df.loc[:,"reversal"] = reversal_list


    # ---------------DETECT DEAD BACTERIA-------------------
    # if for one traj, for more than 70% of the frames you have the error message: "Warning: No pole is definitely on"
    # then we add a "bacteria dead" warning message
    for track_id in range(np.max(merged_df.loc[:,"track_id"].values.astype(int))):
        cond_track_id = merged_df.loc[:,"track_id"] == track_id
        error_messages = merged_df.loc[cond_track_id,"error_message_pole_1"].values.astype(str)
        contains_off_message_arr = ["No pole is definitely on" in error_messages[i] for i in range(len(error_messages))]
        
        # check if 70% of the tracked time the bacteria dont have a leading pole
        if ( np.sum(contains_off_message_arr) / len(error_messages) ) > 0.7:
            dead_indices = merged_df.loc[cond_track_id,:].index
            for bact_index in dead_indices:
                # write that they are dead and remove their reversal information
                error_message.create_error_msg_fct(merged_df, bact_index, "dead bacterium", affected_poles= [1,2])
                merged_df.loc[bact_index, "reversal"] = 0

    # Save dataframe
    merged_df.to_csv(path_save_dataframes_dir + "merged_df.csv")

    print("Reversal analysis finished.")

    return(merged_df)

  














# # %% See why the tracking data has two rows more

# for t in range(10):
#     # merge both arrays
#     cond_t_ana = analysis.loc[:,"t"] == t
#     cond_t_track = tracking.loc[:,"t"] == t
#     merged = np.concatenate((analysis.loc[cond_t_ana,"seg_id"].values, tracking.loc[cond_t_track,"seg_id"].values))

#     # now we see how often each id occurs in a timestep.
#     # if a value occurs NOT TWICE at this timestep, then there is a problem
#     values, occurences = np.unique(merged, return_counts = True)
#     values = values.astype(int)
#     values_unique = values[occurences ==1] #these values occur only once. 
#     #--> revealed the problem with different indexing:
#     #result: the first and the last element occur twice! This is because fiji counts from 1!
#     # So the problem is that fiji counts the track id from 0 but the id from 1  

#     values_non_unique = values[occurences > 2] #these values occur more than twice. 

#     print(values_non_unique)

#     #tracking.sort_values(by=["seg_id","t"])
#     cond_id = (tracking.loc[:,"seg_id"] == 1726) | (tracking.loc[:,"seg_id"] == 101)
#     tracking.loc[cond_id].sort_values(by=["seg_id","t"])

#     cond_id = (merged_df.loc[:,"seg_id"] == 1726) | (merged_df.loc[:,"seg_id"] == 101)
#     merged_df.loc[cond_id].sort_values(by=["seg_id","t"])

# # --> we can see that for some segment IDs, the algorithm gives two tracking IDs.
# # Firstly he proposes a Nan, secondly a real tracking ID. This is okay - we will just ignore all the
# # nan tracking IDs, since we are only interested in real paths that we can follow
    



#     # %%

