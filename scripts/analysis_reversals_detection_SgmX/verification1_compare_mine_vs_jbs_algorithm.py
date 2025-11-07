#%% 

# for clarification: 
# fluo - Fluorescence based reversal detection (Method of Jonathan Schrohe)
# traj - Trajectory based reversal detection (Method of Jean-Baptiste Saulnier)

import pandas as pd
import numpy as np
import settings
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

# initialize statistical indicators
total_trajectories_with_reversals_list = np.array([])
total_trajectories_with_reversals_fluo_list = np.array([])
total_trajectories_with_reversals_traj_list = np.array([])
fluo_total_reversals_list = np.array([])
fluo_total_acceptable_reversals_list = np.array([])
traj_total_reversals_list = np.array([])
traj_total_acceptable_reversals_list = np.array([])
jo_strong_match_list = np.array([])
jb_strong_match_list = np.array([])
michele_precision_list = np.array([])
michele_recall_list = np.array([])
michele_harmonic_mean_list = np.array([])
weak_match_list = np.array([])
matches_list = np.array([])
kicked_min_distance_to_traj_star_or_end_list = np.array([])
total_acceptable_reversals_list = np.array([])


# Set smooth parameter range (parameter from trajs trajectory-based reversal detector)
# smooth_parameter_list = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 3, 4, 5, 6, 7, 8, 9, 10])
smooth_parameter_list = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 3, 4, 5, 6, 7, 8, 9])
# smooth_parameter_list = np.array([0.6])

for smooth_parameter in smooth_parameter_list:
    # round smooth parameter to 1 decimal
    smooth_parameter = np.round(smooth_parameter,1)

    # remove komma from integers
    if (round(smooth_parameter) - smooth_parameter) == 0:
        smooth_parameter = int(smooth_parameter)
    print("\nSmooth-parameter = " + str(smooth_parameter))

    #-------------------------IMPORT AND RENAME DATA---------------------------
    settings_dict = vars(settings.settings_fluorescence_detection_fct())
    step_frames = settings_dict["step_frames"]

    # load dataframes. pay attention that they both have been created from the same segmentation/tracking dataframe (in order to ensure equal tracking ids)
    path_fluo_df = settings_dict["path_program_dir"]+"output/dataframes/merged_df.csv"
    path_traj_df = settings_dict["path_program_dir"]+"input/dataframes/reversals/data_rev_min_size_smoothed_um=" + str(smooth_parameter) + "_um.csv"

    fluo_df = pd.read_csv(path_fluo_df, index_col = 0)
    traj_df = pd.read_csv(path_traj_df)

    # extract / rename important columns
    fluo_df = fluo_df.loc[:,["track_id","t","reversal","error_message_pole_1"]]
    fluo_df = fluo_df.rename(columns = {"track_id" : "track_id","t" : "t","reversal" : "reversal_fluo","error_message_pole_1" : "error_message_pole_1"} )
    # !! ATTENTION (SEE PRINT) !!
    print("\n!!!!!Pay attention that you have to select the center node here in multiple functions. It might not be x5!!!!\n")
    traj_df = traj_df.loc[:,["id", "frame", "reversals","x_centroids","y_centroids"]]
    traj_df = traj_df.rename(columns = {"id": "track_id", "frame": "t", "reversals": "reversal_traj", "x_centroids": "center_x_global_smooth", "y_centroids": "center_y_global_smooth"})

    ##%%
    #-------------------------BRING THEM BOTH TO THE SAME MAXIMUM TIME---------------------------

    # see that both dataframes analyze till the same maximum time
    fluo_maxtime = np.max(fluo_df.loc[:,"t"].values)
    traj_maxtime = np.max(traj_df.loc[:,"t"].values)

    if fluo_maxtime != traj_maxtime:
        if fluo_maxtime > traj_maxtime:
            cond_below_maxtime = fluo_df.loc[:,"t"] <= traj_maxtime
            fluo_df_new = fluo_df.loc[cond_below_maxtime, :]

        elif traj_maxtime > fluo_maxtime:
            cond_below_maxtime = traj_df.loc[:,"t"] <= fluo_maxtime
            traj_df = traj_df.loc[cond_below_maxtime, :]

    ##%% 

    #-------------------------MERGE DATAFRAMES---------------------------

    df_merged = fluo_df.merge(traj_df, how='right', on=['track_id',"t"])
    # fluos dataframe has a lower time resolution - I only evaluate every 5th frame while traj evaluates every frame

    ##%% 

    # #-------------------------ADD CUMULATIVE DISTANCE COLUMN---------------------------
    # # goal: measure path progress for checking conditions trajectory length etc
    # # add the cum. distance to the merged dataframe from the smoothed traj trajectories (so they will both use the same trajectory)

    # go through all track_ids available in trajs dataframe. (many Track IDs are only present in fluo's dataframe, as traj kicks more track_ids out. fluo also kicks Track IDs out, but traj kicks the same out and more. So by going through trajs track ids, we go through the common track ids.) - (this was just the result of a rough check and the code always worked like this. It might be wrong and traj might have one track id that fluo does not have in the future...)
    available_track_ids = np.unique(df_merged.loc[:,"track_id"].values.astype(int) )
    for track_id in available_track_ids:
        
        cond_track_id = df_merged.loc[:,"track_id"] == track_id
        center_x_cumsum = np.append( 0 , np.cumsum( np.abs( df_merged.loc[cond_track_id,"center_x_global_smooth"].values[1:] - df_merged.loc[cond_track_id,"center_x_global_smooth"].values[:-1] ) ) )
        center_y_cumsum = np.append( 0 , np.cumsum( np.abs( df_merged.loc[cond_track_id,"center_y_global_smooth"].values[1:] - df_merged.loc[cond_track_id,"center_y_global_smooth"].values[:-1] ) ) )   
        path_length = np.sqrt( center_x_cumsum**2 + center_y_cumsum**2 )
        # df_merged.loc[cond_track_id, "center_x_cumsum"] = center_x_cumsum
        # df_merged.loc[cond_track_id, "center_y_cumsum"] = center_y_cumsum
        df_merged.loc[cond_track_id, "path_length"] = path_length

    # Remark: I hope it makes sense to make the comparison with the traj's smoothed coordinates for both.


    ##%%---------------DATAFRAME (reversal time) COMPARISON-------------------
    #-----------------------SETTINGS---------------------------
    # a trajectory might be detected at a certain time = cumulative distance in fluo's and traj's dataframe
    # different parameters decide if these two reversal events are "close enough" and if the trajectories are even applicable for comparison - they might not be e.g. the trajectories are too short
    close_time_distance = 20 # in frames: define in which frame distance we are looking for matches
    pcf = 0.0646028 # conversion factor of mu m to pixels
    mean_velocity = 4 # micrometers per minute
    frames_per_second = 1 / 2
    mean_velocity_frames = 4 / (60 * frames_per_second) # um per frame
    close_distance_px = close_time_distance * mean_velocity_frames / pcf # give it in mu m and it gets converted to pixel. How much cumulative distance could they be away, to be considered for a match?
    close_time_distance_boundary = 50 # in frames: define in which time distance the bacteria should not have hit the boundary, measured from the reversal event
    smooth_parameter_px = smooth_parameter / pcf # trajs smoothing parameter gets transferred from mu m into pixels
    min_allowed_distance_to_traj_start_or_end_traj = smooth_parameter_px # a reversal event is not considered for comparison if it is too close to the start of a trajectory. fluos fluo-based method can e.g. detect reversals right at the beginning of a trajectory, but trajs trajectory-based algorithm needs to look at bacteria over longer distances to decide if they reversed or not
    min_allowed_distance_to_traj_start_or_end_fluo = smooth_parameter_px # a reversal event is not considered for comparison if it is too close to the start of a trajectory. fluos fluo-based method can e.g. detect reversals right at the beginning of a trajectory, but trajs trajectory-based algorithm needs to look at bacteria over longer distances to decide if they reversed or not

    #min_allowed_time_distance_to_traj_start_or_end = 30

    #-------------------------TOTAL AMOUNT OF REVERSALS---------------------------
    # see total amount of detected reversals
    traj_total_reversals = np.sum(df_merged.loc[:,"reversal_traj"].values)
    fluo_total_reversals = np.nansum(df_merged.loc[:,"reversal_fluo"].values)

    # total amount of trajectories with reversals
    cond_reversal = (df_merged.loc[:,"reversal_fluo"] == 1) | (df_merged.loc[:,"reversal_traj"] == 1) # we consider all trajectories where either of us detected a reversal
    track_ids_with_reversals = np.unique(df_merged.loc[cond_reversal,"track_id"].values) # unique - to not count trajectories twice if multiple reversals took place on them.
    total_trajectories_with_reversals = len(track_ids_with_reversals)

    # total amount of trajectories with reversals for fluo
    cond_reversal_fluo = (df_merged.loc[:,"reversal_fluo"] == 1) 
    track_ids_with_reversals_fluo = np.unique(df_merged.loc[cond_reversal_fluo,"track_id"].values)
    total_trajectories_with_reversals_fluo = len(track_ids_with_reversals_fluo)

    # total amount of trajectories with reversals for traj
    cond_reversal_traj = (df_merged.loc[:,"reversal_traj"] == 1) 
    track_ids_with_reversals_traj = np.unique(df_merged.loc[cond_reversal_traj,"track_id"].values)
    total_trajectories_with_reversals_traj = len(track_ids_with_reversals_traj)

    #-------------------------TOTAL AMOUNT OF "GOOD" REVERSALS (traj)---------------------------
    # try to select good/comparable detections from traj: 
    # = see total amount of detected reversals on trajectories which satisfy the conditions:
    # 1.) reversal sufficient distance away from beginning/end of trajectory (traj needs sufficient information from the past)
    # 2.) fluos dataframe does not exclude the bacterium because "too close to boundary" in close times
    cond_traj_reversals = df_merged.loc[:, "reversal_traj"] == 1 # select all indices where traj had reversals 
    traj_reversal_indicies = df_merged.loc[cond_traj_reversals,:].index
    traj_acceptable_reversal_indices = np.array([]) # create an array where we now note all indices of acceptable reversals
    traj_track_id_with_issue = np.array([]) 
    kicked_min_distance_to_traj_star_or_end = 0

    for index in traj_reversal_indicies:
        # extract reversal information about the reversals that traj detected
        track_id = df_merged.loc[index, "track_id"]         # track id of this reversal
        cond_track_id = df_merged.loc[:, "track_id"] == track_id

        time = df_merged.loc[index, "t"] # time of reversal
        available_times = df_merged.loc[cond_track_id, "t"].values # all recorded times for that track id
        duration_trajectory = available_times[-1] - available_times[0] # total duration of the trajectory
        #time_distance_to_traj_start = abs(time - available_times[0])
        #time_distance_to_traj_end = abs(time - available_times[-1])
        #min_time_distance_to_traj_star_or_end = np.minimum(time_distance_to_traj_start, time_distance_to_traj_end)

        path_length_at_reversal = df_merged.loc[index, "path_length"] # how much cumulative distance the bacterium center travelled at the time of reversal
        available_path_lenghts = df_merged.loc[cond_track_id, "path_length"].values # the total spectrum of path lengths available (s = 0,...,s_end)
        length_trajectory = available_path_lenghts[-1] # spatial length of trajectory
        distance_to_traj_start = abs(path_length_at_reversal - 0) # distance to the start of the trajectory
        distance_to_traj_end = abs(path_length_at_reversal - length_trajectory)
        min_distance_to_traj_star_or_end = np.minimum(distance_to_traj_start, distance_to_traj_end) # the lower value of both the distance to the start and the distance to the end

        # check the three above criteria: traj longer than 50 frames, sufficient distance from traj beginning and end, bacteria not touching boundary in close times (which was originally written into fluos df)
        if (min_distance_to_traj_star_or_end >= min_allowed_distance_to_traj_start_or_end_traj):

            # check also if fluo had no boundary exclusion for this index in "close" times
            fluo_cond_close_times_boundary = (df_merged.loc[:,"t"] >= time - close_time_distance_boundary) & (df_merged.loc[:,"t"] <= time + close_time_distance_boundary) # select all the times which have the maximum allowed time distance from the reversal event to a possible boundary detection
            selection_error_messages = df_merged.loc[cond_track_id & fluo_cond_close_times_boundary , "error_message_pole_1"].values
            # if there is any boundary error message in close times, ignore this bacterium
            boundary_errors_amount = np.sum(["boundary" in str(a) for a in selection_error_messages])

            if boundary_errors_amount == 0:
                traj_acceptable_reversal_indices = np.append(traj_acceptable_reversal_indices, index)
            else:
                # add this trajectory to the list of trajectories with at least one issue-reversal on them
                traj_track_id_with_issue = np.append(traj_track_id_with_issue, df_merged.loc[index,"track_id"])

        else:
            # add this trajectory to the list of trajectories with at least one issue-reversal on them
            traj_track_id_with_issue = np.append(traj_track_id_with_issue, df_merged.loc[index,"track_id"])
            kicked_min_distance_to_traj_star_or_end +=1

    traj_total_acceptable_reversals = len(traj_acceptable_reversal_indices)

    #-------------------------AMOUNT OF MATCHES OF TRAJ TO FLUO---------------------------
    # this section also contains the total amount of "good" reversals from fluo - the subset of those reversals, which are comparable with trajs, e.g. all those taken after a specific time/ distance from the movie start, when traj already had enough time to gather information from the past for his reversals

    # start comparison of matches of traj to fluos algorithm:

    # mark all the positions where fluo detected reversals
    cond_fluo_reversals = df_merged.loc[:, "reversal_fluo"] == 1 
    fluo_reversal_indicies = df_merged.loc[cond_fluo_reversals,:].index
    fluo_acceptable_reversal_indices = np.array([])
    fluo_matching_reversal_indices = np.array([])
    fluo_track_id_with_issue = np.array([]) 
    traj_already_matched_indices = np.array([])
    fluo_traj_rev_index_correspondence = np.array([])

    # count how often trajs algorithm matches to fluos in close times:

    # we loop through all reversals
    matches = 0
    #considered_fluo_reversal_indicies = np.array([]) #WHAT IS THIS????
    for index in fluo_reversal_indicies:
        # extract reversal information about the reversals that fluo detected
        track_id = df_merged.loc[index, "track_id"]
        cond_track_id = df_merged.loc[:, "track_id"] == track_id

        time = df_merged.loc[index, "t"]
        available_times = df_merged.loc[cond_track_id, "t"].values
        duration_trajectory = available_times[-1] - available_times[0]
        #time_distance_to_traj_start = abs(time - available_times[0])
        #time_distance_to_traj_end = abs(time - available_times[-1])
        #min_time_distance_to_traj_star_or_end = np.minimum(time_distance_to_traj_start, time_distance_to_traj_end)

        path_length_at_reversal = df_merged.loc[index, "path_length"]
        available_path_lenghts = df_merged.loc[cond_track_id, "path_length"].values
        length_trajectory = available_path_lenghts[-1]
        distance_to_traj_start = abs(path_length_at_reversal - 0)
        distance_to_traj_end = abs(path_length_at_reversal - length_trajectory)
        min_distance_to_traj_star_or_end = np.minimum(distance_to_traj_start, distance_to_traj_end)

        # check the three above criteria: trajectory longer than 50 frames, sufficient distance from trajectory beginning and end, bacteria not touching boundary in close times (which was originally written into fluos df)
        if (min_distance_to_traj_star_or_end >= min_allowed_distance_to_traj_start_or_end_fluo):

            # check also if fluo had no boundary exclusion for this index in "close" times
            fluo_cond_close_times_boundary = (df_merged.loc[:,"t"] >= time - close_time_distance_boundary) & (df_merged.loc[:,"t"] <= time + close_time_distance_boundary)
            selection_error_messages = df_merged.loc[cond_track_id & fluo_cond_close_times_boundary , "error_message_pole_1"].values
            # if there is any boundary error message in close times, ignore this bacterium
            boundary_errors_amount = np.sum(["boundary" in str(a) for a in selection_error_messages])

            # if there is also no bacterium which goes close to the boundary once, all three rules for a "good" reversal are fulfilled!
            if boundary_errors_amount == 0:

                # we can add this fluo reversal index to the good reversals since it satisfies the conditions
                fluo_acceptable_reversal_indices = np.append(fluo_acceptable_reversal_indices, index)

                # look now in trajs dataframe if I can find traj reversals for the same track id at close times / distances to the selected fluo reversal (index)
                # check for reversals close in time
                cond_selection_close_times = (df_merged.loc[:,"t"] <= (time + close_time_distance)) & (df_merged.loc[:,"t"] >= (time - close_time_distance))
                selection_close_times = df_merged.loc[cond_track_id & cond_selection_close_times, "reversal_traj"] # list all traj reversals values (000101000) within the "close-time"-range of my detected reversals
                cond_reversal_in_close_times = selection_close_times == 1 # is there a one in the close-time-reversal values? e.g. is there a reversal
                close_reversals_indices_time = selection_close_times.loc[cond_reversal_in_close_times].index # select the index of the dataframe entry where a reversal of traj was found in the close time
                
                # check for reversals close in distance (analog procedure)
                cond_selection_close_distances = (df_merged.loc[:,"path_length"] <= (path_length_at_reversal + close_distance_px)) & (df_merged.loc[:,"t"] >= (path_length_at_reversal - close_distance_px))
                selection_close_distances = df_merged.loc[cond_track_id & cond_selection_close_distances, "reversal_traj"]
                cond_reversal_in_close_distances = selection_close_distances == 1
                close_reversals_indices_space = selection_close_distances.loc[cond_reversal_in_close_distances].index

                # merge these two arrays and keep the bigger information 
                # = so keep the index in both cases: if a reversal was detected in close times or if it was detected in close distances. For sure also keep it if both procedured found it
                close_reversals_indices = np.unique(np.concatenate((close_reversals_indices_time, close_reversals_indices_space)))
                # --> Now, the index/indices of a traj reversal event, which is close enough in time or space to a fluo reversal event, were memorized! (if there exists a close one and the fluo reversal is one a trajectory which is good enough for comparison)

                # if one of the close reversals from traj was already matched before, exclude it from the selection. We dont want to match multiple traj reversals with one reversal of fluo.
                for i in close_reversals_indices: # loop through all detected close traj reversals
                    if i in traj_already_matched_indices: # check if they have been already detected
                        cond_exclude_index = close_reversals_indices != i
                        close_reversals_indices = close_reversals_indices[cond_exclude_index]

                # if one of the close reversals from traj was not an accepted one, exclude it from the selection. We dont want to match multiple traj reversals with one reversal of fluo.
                for i in close_reversals_indices: # loop through all detected close traj reversals
                    if i not in traj_acceptable_reversal_indices: # check if they have been already detected
                        cond_exclude_index = close_reversals_indices != i
                        close_reversals_indices = close_reversals_indices[cond_exclude_index]

                # note a match and remark that this index is already used for a match, in case a close traj reversal was found
                if len(close_reversals_indices) >= 1:
                    matches = matches + 1
                    fluo_matching_reversal_indices = np.append(fluo_matching_reversal_indices, index)
                    traj_matched_index = close_reversals_indices[0] #select the first matched index (this is the best method, otherwise we would miss some possible matches)
                    traj_already_matched_indices = np.append(traj_already_matched_indices, traj_matched_index)

                    # also create a list of index fluo-traj-index-correspondings
                    fluo_traj_rev_index_correspondence = np.concatenate((fluo_traj_rev_index_correspondence, [index, traj_matched_index]), axis = 0)

            else: # else: if boundary errors detected -just skip that index --> it is not put into the list of indices with acceptable reversals (fluo_acceptable_reversal_indices)
                # furthermore, add this trajectory to the list of trajectories with at least one issue-reversal on them
                fluo_track_id_with_issue = np.append(fluo_track_id_with_issue, df_merged.loc[index,"track_id"])

        # if one of the reversals on a trajectory fails to meet the minimum distance to the end of a trajectory, we remark that it's excluded, since another iteration with a good reversal on the same trajectory would add it to the accepted list 
        else: 
            # furthermore, add this trajectory to the list of trajectories with at least one issue-reversal on them
            fluo_track_id_with_issue = np.append(fluo_track_id_with_issue, df_merged.loc[index,"track_id"])


    #-------------------------GET THE REMAINING TEST STATISTICS---------------------------

    # conclude the amount of fluo's total acceptable reversals
    fluo_total_acceptable_reversals = len(fluo_acceptable_reversal_indices)

    # now note all the track ids = trajectories which are good, i.e have at least one reversal on them and dont have any issues for all the reversals on them. Exclude all the track ids marked as bad due to an issue with one reversal on them
    fluo_acceptable_track_ids_old = np.unique(df_merged.loc[fluo_acceptable_reversal_indices,"track_id"].values) # first assume that all the acceptable reversals lie on a trajectory with no issues
    fluo_acceptable_track_ids = [x for x in fluo_acceptable_track_ids_old if x not in fluo_track_id_with_issue] # now remove all the trajectories with at least one issue on them

    traj_acceptable_track_ids_old = np.unique(df_merged.loc[traj_acceptable_reversal_indices,"track_id"].values) # first assume that all the acceptable reversals lie on a trajectory with no issues
    traj_acceptable_track_ids = [x for x in traj_acceptable_track_ids_old if x not in traj_track_id_with_issue] # now remove all the trajectories with at least one issue on them

    # also reshape the fluo-traj index correspondence
    fluo_traj_rev_index_correspondence = np.reshape(fluo_traj_rev_index_correspondence, [int(len(fluo_traj_rev_index_correspondence)/2), 2] )

    # Calculate the match fraction:
    # Take the total number of acceptable reversal events:
    total_acceptable_reversals = traj_total_acceptable_reversals + fluo_total_acceptable_reversals
    
    # each match now reduces two of these events. The remaining number says, that either fluo had a reversal and traj didn't detect it or traj detected a reversal where fluo did not
    if total_acceptable_reversals != 0:
        jo_strong_match_fraction = 2*matches / total_acceptable_reversals
        # --> This match fraction increases if traj matched each of fluos reversals and decreases if traj made a mistake (or fluo made a mistake - but his approach is seen as the reference)
    else: 
        strong_match_fraction = np.nan

    if fluo_total_acceptable_reversals != 0:
        weak_match_fraction = matches / fluo_total_acceptable_reversals
        # --> This weak match fraction just looks how many of fluos events were spotted by traj in time. But if he detects a lot more reversals, e.g. with a low smooth parameter, it would just stay at 100% and is thus not as good
    else:
        weak_match_fraction = np.nan

    # also jbs idea:
    jb_strong_match_fraction = matches / (total_acceptable_reversals - matches)

    # Michèle idea
    true_positive = matches
    false_negative = fluo_total_acceptable_reversals - matches
    false_positive = traj_total_acceptable_reversals - matches
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    harmonic_mean = 2 * precision * recall / (precision + recall)

    #-------------------------WRITE COMPARISON OVERVIEW---------------------------
    print("Total number of trajectories with reversals: " + str(total_trajectories_with_reversals))
    print("Total number of trajectories with reversals for fluo: " + str(total_trajectories_with_reversals_fluo))
    print("Total number of trajectories with reversals for traj: " + str(total_trajectories_with_reversals_traj))
    print("fluo total reversals: " + str(fluo_total_reversals) + str(" (Reference for Comparison)."), flush = True)
    print("fluo total acceptable reversals: " + str(fluo_total_acceptable_reversals) )
    print("traj total reversals: " + str(traj_total_reversals))
    print("traj total acceptable reversals: " + str(traj_total_acceptable_reversals))
    print("Total number of acceptable reversals of both methods: " + str(total_acceptable_reversals))
    print("Amount of maches: " + str(matches))
    print("Weak match fraction: " + str( np.round(100 * weak_match_fraction,2) ) + str(" % (how many percent of reversal events caused a match?)"))
    print("Strong match fraction jo: " + str( np.round(100 * jo_strong_match_fraction,2) ) + str(" % (how many percent of the reversal events fluo detected could traj detect?)"))
    print("Strong match fraction jb: " + str( np.round(100 * jb_strong_match_fraction,2) ) + str(" % (how many percent of the reversal events fluo detected could traj detect?)"))
    print("Accepted traj detection times: t_fluo +-", str(close_time_distance))
    print("\nRemark: Acceptable means: \n1. Distance of reversal to trajectories start/end at least", str(np.round(min_allowed_distance_to_traj_start_or_end_fluo,2)), " pixels for fluo and", str(np.round(min_allowed_distance_to_traj_start_or_end_traj,2)), "pixels for traj \n2. Fluo program has not detected bacteria to be 'too close to boundary' in reversal time +-" , str(close_time_distance_boundary)+ " frames \n")
    print("--------------------------")

    #-------------------------SAVE STATISTICS INTO LISTS FOR EACH SMOOTHING PARAM---------------------------
    total_trajectories_with_reversals_list = np.append(total_trajectories_with_reversals_list, total_trajectories_with_reversals)
    total_trajectories_with_reversals_fluo_list = np.append(total_trajectories_with_reversals_fluo_list, total_trajectories_with_reversals_fluo)
    total_trajectories_with_reversals_traj_list = np.append(total_trajectories_with_reversals_traj_list, total_trajectories_with_reversals_traj)
    fluo_total_reversals_list = np.append(fluo_total_reversals_list, fluo_total_reversals)
    fluo_total_acceptable_reversals_list = np.append(fluo_total_acceptable_reversals_list, fluo_total_acceptable_reversals)
    traj_total_reversals_list = np.append(traj_total_reversals_list, traj_total_reversals)
    traj_total_acceptable_reversals_list = np.append(traj_total_acceptable_reversals_list, traj_total_acceptable_reversals)
    total_acceptable_reversals_list = np.append(total_acceptable_reversals_list, total_acceptable_reversals)
    jo_strong_match_list = np.append(jo_strong_match_list, jo_strong_match_fraction)
    jb_strong_match_list = np.append(jb_strong_match_list, jb_strong_match_fraction)
    michele_precision_list = np.append(michele_precision_list, precision)
    michele_recall_list = np.append(michele_recall_list, recall)
    michele_harmonic_mean_list = np.append(michele_harmonic_mean_list, harmonic_mean)
    weak_match_list = np.append(weak_match_list, weak_match_fraction)
    matches_list = np.append(matches_list, matches)
    kicked_min_distance_to_traj_star_or_end_list = np.append(kicked_min_distance_to_traj_star_or_end_list, kicked_min_distance_to_traj_star_or_end)
    # ratio_n_traj_n_fluo_list = 
    # ratio_n_traj_n_fluo_acceptable_list = 

#%%
#----------------PLOT TEST STATISTICS---------------------
plt.figure(1)
plt.xlabel("Smooth parameter $\sigma$")
plt.title("Total amount of trajectories with reversals detected on,\nusing different methods")
plt.plot(smooth_parameter_list, total_trajectories_with_reversals_list, "x--", label = "both methods")
plt.plot(smooth_parameter_list, total_trajectories_with_reversals_fluo_list, "x--", label = "fluorescence = const. = " + str(total_trajectories_with_reversals_fluo_list[0]))
plt.plot(smooth_parameter_list, total_trajectories_with_reversals_traj_list, "x--", label = "trajectories")
plt.legend()

plt.figure(2)
plt.xlabel("Smooth parameter $\sigma$")
plt.title("Total amount of reversals detected,\nusing different methods")
plt.plot(smooth_parameter_list, fluo_total_reversals_list, "x--", label = "fluorescence = const. = " + str(fluo_total_reversals_list[0]))
plt.plot(smooth_parameter_list, traj_total_reversals_list, "x--", label = "trajectories")
plt.legend()

plt.figure(3)
plt.xlabel("Smooth parameter $\sigma$")
plt.title("Total amount of acceptable reversals detected,\nusing different methods. (acceptable means: you can compare both algorithms)")
plt.plot(smooth_parameter_list, fluo_total_acceptable_reversals_list, "x--", label = "fluorescence")
plt.plot(smooth_parameter_list, traj_total_acceptable_reversals_list, "x--", label = "trajectories")
plt.legend()

plt.figure(8)
plt.xlabel("Smooth parameter $\sigma$")
plt.title("Fraction of weak matches")
plt.plot(smooth_parameter_list, weak_match_list, "x--")

plt.figure(9)
plt.xlabel("Smooth parameter $\sigma$")
plt.title("Fraction of jo vs jb strong matches")
plt.plot(smooth_parameter_list, jo_strong_match_list, "x--", label = "jo")
plt.plot(smooth_parameter_list, jb_strong_match_list, "x--", label = "jb")
plt.legend()

plt.figure(10)
plt.xlabel("Smooth parameter $\sigma$")
plt.title("Fraction of jo vs jb strong matches (michele method)")
plt.plot(smooth_parameter_list, michele_precision_list, "x--", label = "precision")
plt.plot(smooth_parameter_list, michele_recall_list, "x--", label = "recall")
plt.plot(smooth_parameter_list, michele_harmonic_mean_list, "x--", label = "harmonic mean")
plt.legend()
print("list of smooth parameters: ", smooth_parameter_list)
print("list of harmonic mean: ", michele_harmonic_mean_list)

data_save = {"smooth_parameter_list": smooth_parameter_list, 
            "michele_precision_list": michele_precision_list, 
            "michele_recall_list": michele_recall_list,
            "michele_harmonic_mean_list": michele_harmonic_mean_list}
    
import settings
settings_dict_param_estim = vars(settings.settings_parameter_estimation_fct())
path_save_image_dir = settings_dict_param_estim["path_save_image_dir"]
with open(path_save_image_dir + "final_plot_michele_method.pkl", 'wb') as pickle_file:
    pickle.dump(data_save, pickle_file)

plt.figure(12)
plt.xlabel("Smooth parameter $\sigma$")
# plt.title("Amount of matches")
plt.plot(smooth_parameter_list, matches_list, "x--")


# %%

#-------------PLOT TRAJECTORIES FOR A CERTAIN SMOOTHING PARAMETER---------------------

# ONLY WORKS IF YOU RAN THE ANALYSIS ABOVE FOR !ONE! SMOOTHING PARAMETER / JUST PICKS THE LAST





# THIS IS MAYBE OUTDATES!! I JUST KEEP IT BECAUSE I FORGOT WHY I DID IT

# # -----------GET DATA ABOUT ALL THE REVERSAL EVENTS (NOT ONLY GOOD ONES)----------
# # If traj or fluo finds a reversal, the corresponding data gets loaded here

# # find all trajectories with reversals
# df_track_ids_info = pd.DataFrame(columns=("id","frame","x5","y5",))
# track_ids_with_reversals = np.array([])

# # loop through all track ids and check if they have a reversal detected on them with any method
# for track_id in tqdm(range( np.max( df_merged.loc[:,"track_id"].values.astype(int) ) )):
#     cond_track_id = df_merged.loc[:,"track_id"] == track_id
#     cond_track_id_reversals = (df_merged.loc[:,"track_id"] == track_id) & ( (df_merged.loc[:,"reversal_fluo"] == 1) | (df_merged.loc[:,"reversal_traj"] == 1) )

#     # check if there is a reversal for this trajectory and save the position and time of the event
#     if len(df_merged.loc[cond_track_id_reversals,:]) >= 1: # its not empty, there must have been some indices with reversals from the above criterium
#         df_track_ids_info_i = df_merged.loc[cond_track_id_reversals, ["track_id", "t", "center_x_global_smooth", "center_y_global_smooth"]]
#         df_track_ids_info = pd.concat( [df_track_ids_info, df_track_ids_info_i], ignore_index = True)
#         track_ids_with_reversals = np.append(track_ids_with_reversals, track_id)




#-------------------------PLOT REVERSAL EVENTS / TRAJECTORIES---------------------------

#-----------------CHOOSE WHICH TRAJECTORIES TO LOOK AT-----------------------------

# METHOD 1
# you could look at trajectories with reversals in general

cond_reversal = (df_merged.loc[:,"reversal_fluo"] == 1) | (df_merged.loc[:,"reversal_traj"] == 1)
track_ids_with_reversals = np.unique(df_merged.loc[cond_reversal,"track_id"].values)
# track_id = track_ids_with_reversals[700]
track_id = np.random.choice(track_ids_with_reversals)


# METHOD 2 
# or at fluo reversals on accepted TRAJECTORIES (= not a single reversal with issue on them) 

# track_id = np.random.choice(fluo_acceptable_track_ids)

# ----> this method should be the best, since we expect that fluo is mostly right. So we can see here if traj matches nearby or not

# METHOD 3
# or at "good" trajectories with reversals of traj
# you just should have to mark all the bad ids again and subtract their trajectories from the list of good trajectories



#------------------#GATHER ADDITIONAL INFO AND PLOT-------------------------

# extract our trajectory coords
cond_track_id = df_merged.loc[:,"track_id"] == track_id
trajectory_x = df_merged.loc[cond_track_id,"center_y_global_smooth"].values
trajectory_y = df_merged.loc[cond_track_id,"center_x_global_smooth"].values

# extract the coords and time of the reversal event of traj method 
traj_cond_track_id_and_reversal = (df_merged.loc[:,"track_id"] == track_id) & (df_merged.loc[:,"reversal_traj"] == 1) # make a condition to select that part of the dataframe where the reversals events are for given track id
traj_reversal_x = df_merged.loc[traj_cond_track_id_and_reversal, "center_y_global_smooth"]
traj_reversal_y = df_merged.loc[traj_cond_track_id_and_reversal, "center_x_global_smooth"]
traj_reversal_t = df_merged.loc[traj_cond_track_id_and_reversal,"t"].values

# extract the coords and time of the reversal event of fluo method 
fluo_cond_track_id_and_reversal =  (df_merged.loc[:,"track_id"] == track_id) & (df_merged.loc[:,"reversal_fluo"] == 1)
fluo_reversal_x = df_merged.loc[fluo_cond_track_id_and_reversal, "center_y_global_smooth"]
fluo_reversal_y = df_merged.loc[fluo_cond_track_id_and_reversal, "center_x_global_smooth"]
fluo_reversal_t = df_merged.loc[fluo_cond_track_id_and_reversal,"t"].values


# extract the amount of matches on this trajectory. We just have to see which of trajs reversal indices are present on this trajectory and then check if they are in the macthing - correspondence list (i.e. if they have been close enough)
traj_cond_track_id_and_reversal = (df_merged.loc[:,"track_id"] == track_id) & (df_merged.loc[:,"reversal_traj"] == 1)
traj_reversal_indicies_now = df_merged.loc[traj_cond_track_id_and_reversal, :].index
amount_matches_on_this_traj_fluo = len([1 for x in traj_reversal_indicies_now if (x in fluo_traj_rev_index_correspondence)])


# plot reversal information
plt.close("all")
plt.title("Track id: " + str(track_id) + ", All fluo rev. acceptable (available for match)?: " + str(track_id in fluo_acceptable_track_ids) + ", \nAll traj rev. acceptable (available for match)?: " + str(track_id in traj_acceptable_track_ids) +  ", Matches = " + str(amount_matches_on_this_traj_fluo)) # we say if the entire trajectory is acceptable or not, i.e. there might be 2 fluo reversals, one too close to the boundary. Thus, the trajectory itsself will not be on the list of acceptable trajectories, thus the answer to the question is false, even though one reversal is acceptable
#plt.title("One detection of fluo on a 'good' trajectory is picked (see three rules) and trajs reversals are additionally plotted, if they exist.")
plt.scatter(traj_reversal_x * pcf, traj_reversal_y * pcf, s = 30**2, c = "blue", label = "traj reversal, t = " + str(traj_reversal_t))
plt.scatter(fluo_reversal_x * pcf, fluo_reversal_y * pcf, s = 20**2, c = "pink", marker = "P", linewidths = 3, label = "fluo reversal, t = " + str(fluo_reversal_t))
plt.xlabel(["x_coord in um"])
plt.ylabel(["y_coord in um"])
plt.text(trajectory_x[0] * pcf, trajectory_y[0] * pcf, "start", fontsize = 15, c = "green")
plt.plot(trajectory_x * pcf, trajectory_y * pcf, label = "smoothed_traj_trajectory")

plt.legend()
