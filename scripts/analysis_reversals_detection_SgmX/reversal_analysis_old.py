import pandas as pd
import numpy as np
import time
import sys
import settings
from tqdm import tqdm

def reversal_analysis_fct():

    # load paths and constants from settings
    path_dataframes_dir = list(vars(settings.settings_reversal_analysis_fct()).values())[0]

    # ---------------IMPORT AND MERGE TRACKING AND ANALYSIS DATA-------------------
    # = Add the tracking information to the fluorescence analysis

    sys.stdout.write("\nImport an merge tracking and analysis data... ")

    #df_t.to_csv(path_save_dir + "analysis.csv")
    # specify paths of dataframes
    path_tracking = path_dataframes_dir + "tracking.csv"
    path_analysis = path_dataframes_dir + "analysis.csv"
    analysis = pd.read_csv(path_analysis)
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



    # ---------------TRANSFORM THE POLE NANS-------------------
    # firstly: find bacteria where there is nan over ALL times. Eg Boundary bacteria
    # then we will transform the pole nans into numbers where there is pole values detected
    
    # idea of transformation:
            # lead pole     lead pole new
            # nan       -->   2
            # nan       -->   2
            # 2         -->   2
            # nan       -->   2
            # 1         -->   1

    print("\nTransform pole nans...")
    #sys.stdout.write("\nTransform pole nans: ", end = "", flush = True)
    time.sleep(2)

    # loop through all the track ids
    for track_index in tqdm(range(len(merged_df))):

        # select all data for this track id
        cond_track_id = merged_df.loc[:,"track_id"] == track_index
        selection = merged_df.loc[cond_track_id,:]

        # if in this selection all pole detections are nan, then we will just write that no reversal could be detected.
        # this is the case e.g. for the bacteria that stay at the boundary for the entire time
        if np.sum(np.isnan(selection.loc[:,"leading_pole"].values).astype(int)) == len(selection):
            selection_indices = selection.index
            merged_df.loc[selection_indices,"reversal_error"] = "The pole could never be identified --> no reversal detected"

        # ELSE we can TRANSFORM THE TABLE, so that nans are replaced by the next detected number
        else:
            index_count = np.array([])
            non_nan_count = 0

            # loop in this selection through the time
            for t in selection.loc[:,"t"].values:
                cond_t = selection.loc[:,"t"] == t

                # as long as we have nans in the table, we note the indices
                if np.isnan(selection.loc[cond_t,"leading_pole"].values) == True:
                    index_count = np.append(index_count,selection.loc[cond_t,"leading_pole"].index)
                else:
                    # if there is no nan, we count it
                    non_nan_count = non_nan_count + 1
                    
                    # as long as there are less than two non-nans, we just continue counting the index
                    if non_nan_count != 2:
                        non_nan_value = selection.loc[cond_t,"leading_pole"].values[0]
                        index_count = np.append(index_count,selection.loc[cond_t,"leading_pole"].index)
                    
                    # but as soon as we found the 2nd non-nan, fill the counted indices
                    # with this non-nan value and reset our counters
                    else:
                        merged_df.loc[index_count,"leading_pole"] = non_nan_value
                        index_count = np.array([])
                        non_nan_value = selection.loc[cond_t,"leading_pole"].values[0]
                        non_nan_count = 1
            
            # repeat this until we are finished.
            # fill the last nans with the last number
            merged_df.loc[index_count,"leading_pole"] = non_nan_value
            #print("\rTransform pole nans - track_index: " + str(track_index) + "/" + str(max_track_id), end = "")

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
    reversing_value = (leading_pole[1:] - leading_pole[0:-1]) != 0

    # since this rule also detects nan-nan as nonzero, we have to set those buggy places to 0
    error_indices = np.where(np.isnan(leading_pole[1:]) == True)[0]
    reversing_value[error_indices] = 0

    # consider that we only have to look within one trajectory.
    # so see where the transition from one trajectory to the next is not
    traj_change = (track_id[1:] - track_id[0:-1]) == 0

    # a reversal takes place if the leading pole changes and if the trajectory does not change (within one trajectory) 
    reversal_list = np.zeros(len(reversing_value))
    reversal_list[reversing_value & traj_change] = 1

    # on the first step, no reversal can be detected. Thus we set it to no reversal = 0
    reversal_list = np.concatenate(([0],reversal_list))

    # this analysis makes only goes  
    merged_df.loc[:,"reversal"] = reversal_list

    # Save dataframe
    merged_df.to_csv(path_dataframes_dir + "merged_df.csv")

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
