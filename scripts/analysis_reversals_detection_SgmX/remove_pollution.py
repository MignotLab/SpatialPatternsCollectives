#%%
import pandas as pd
import numpy as np
from tqdm import tqdm
import count_images
import dataframe
from scipy.spatial import KDTree as kdtree

#%%
import pandas as pd
import numpy as np
from tqdm import tqdm
import count_images
import dataframe
from scipy.spatial import KDTree as kdtree
import pickle

# find all the properties of selected lagging poles
def find_neighbour_information_learning(selection_polluted, analysis, pcf, width):

    #----------ONLY LOOK AT SELECTION AND BACTERIA WITHOUT PROBLEMS------------------
    # this function receives the selection list, a list of bacteria where the lagging pole detection worked well.
    # now define selection as a condition instead of index list.
    all_indices = analysis.loc[:,"seg_id"]
    cond_selection = [x in selection_polluted for x in all_indices]

    # now find the bacteria with problems to exclude them
    # e.g. wrong pole count, skel too short
    cond_no_problems = np.isnan(analysis.loc[:,"mean_intensity_pole_1"].values) == False
    
    # df will now be the dataframe of the bacteria with no problems
    df = analysis.loc[cond_no_problems , :] 

    # find also the lagging poles of the analysis
    cond1_analysis = analysis.loc[:,"leading_pole"].values == 2
    cond2_analysis = analysis.loc[:,"leading_pole"].values == 1

    #----------FIND ALL LAGGING POLES------------------
    # (we want to analyze only lagging poles for the learning)

    # select all the leading poles
    leading_pole = df.loc[:,"leading_pole"].values
    cond1 = leading_pole == 1
    cond2 = leading_pole == 2
    cond_nan = ~(cond1 | cond2)

    # select all the lagging poles
    lagging_pole = np.copy(leading_pole)
    lagging_pole[cond1] = 2
    lagging_pole[cond2] = 1
    cond1 = lagging_pole == 1
    cond2 = lagging_pole == 2
    cond_nan = ~(cond1 | cond2)

    # select the lagging pole coords of all pole 1 and all pole 2
    # these are the poles where we want to find the affected intensity
    x_pole_1_global = df.loc[cond1, "pole_1_x_local_pollution"].values + df.loc[cond1, "x_min_global"].values
    y_pole_1_global = df.loc[cond1, "pole_1_y_local_pollution"].values + df.loc[cond1, "y_min_global"].values
    x_pole_2_global = df.loc[cond2, "pole_2_x_local_pollution"].values + df.loc[cond2, "x_min_global"].values
    y_pole_2_global = df.loc[cond2, "pole_2_y_local_pollution"].values + df.loc[cond2, "y_min_global"].values

    # merge the two lagging pole lists in the right order via cond
    x_lagging_pole = np.zeros(len(df))
    y_lagging_pole = np.zeros(len(df))
    x_lagging_pole[cond1] = x_pole_1_global
    x_lagging_pole[cond2] = x_pole_2_global
    y_lagging_pole[cond1] = y_pole_1_global
    y_lagging_pole[cond2] = y_pole_2_global

    #----------FIND ALL NEIGHBOURS OF THE LAGGING POLES AND CALCULATE SIGNAL------------------

    # find a list of all possible pole coordinates in general
    x_min = df.loc[:,"x_min_global"].values
    y_min = df.loc[:,"y_min_global"].values
    all_x_pole_1 = df.loc[:,"pole_1_x_local_pollution"].values + x_min
    all_y_pole_1 = df.loc[:,"pole_1_y_local_pollution"].values + y_min
    all_x_pole_2 = df.loc[:,"pole_2_x_local_pollution"].values + x_min
    all_y_pole_2 = df.loc[:,"pole_2_y_local_pollution"].values + y_min

    # build a tree datastructure of all available poles
    tree_1 = kdtree(np.column_stack((all_x_pole_1, all_y_pole_1)))
    tree_2 = kdtree(np.column_stack((all_x_pole_2, all_y_pole_2)))

    # find the neighboured lagging pole's indices and distances 
    neigh_dist_1, neigh_index_1 = tree_1.query( np.column_stack((x_lagging_pole,y_lagging_pole)), k = 20 ) 
    neigh_dist_2, neigh_index_2 = tree_2.query( np.column_stack((x_lagging_pole,y_lagging_pole)), k = 20 ) 

    # extract the closest neighbours
    cond_closest_neigh_1 = (neigh_dist_1 < 2 * width / pcf)  & (neigh_dist_1 > 0)
    cond_closest_neigh_2 = (neigh_dist_2 < 2 * width / pcf)  & (neigh_dist_2 > 0)

    # extract the neighbour's intensities
    intensity_1 = df.loc[:, "mean_intensity_pole_1"].values
    intensity_2 = df.loc[:, "mean_intensity_pole_2"].values

    neigh_index_1_flatten = neigh_index_1.flatten()
    neigh_intensity_1 = intensity_1[neigh_index_1_flatten]
    neigh_intensity_1 = np.reshape(neigh_intensity_1, (len(df), 20))
    neigh_index_2_flatten = neigh_index_2.flatten()
    neigh_intensity_2 = intensity_2[neigh_index_2_flatten]
    neigh_intensity_2 = np.reshape(neigh_intensity_2, (len(df), 20))

    # neglect the neighbours that are not close enough
    neigh_intensity_1[~cond_closest_neigh_1] = 0
    neigh_intensity_2[~cond_closest_neigh_2] = 0
    neigh_dist_1[~cond_closest_neigh_1] = np.inf
    neigh_dist_2[~cond_closest_neigh_2] = np.inf

    # calculate ONE signal. sum(I/d^2)
    signal_1 = np.sum( neigh_intensity_1 / neigh_dist_1**2 , axis = 1 ) # all the effect from pole 1s
    signal_2 = np.sum( neigh_intensity_2 / neigh_dist_2**2 , axis = 1 ) # all the effect from pole 1s
    signal = signal_1 + signal_2
    
    # calculate TWO split signals. sum(1/d^2) and sum(I)
    # signal_dist_1 = np.sum( 1 / neigh_dist_1**2 , axis = 1 ) # all the effect from pole 1s
    # signal_dist_2 = np.sum( 1 / neigh_dist_2**2 , axis = 1 ) # all the effect from pole 1s
    # signal_dist = signal_dist_1 + signal_dist_2

    # signal_int_1 = np.sum( neigh_intensity_1 , axis = 1 ) # all the effect from pole 1s
    # signal_int_2 = np.sum( neigh_intensity_2 , axis = 1 ) # all the effect from pole 1s
    # signal_int = signal_int_1 + signal_int_2
    
    # calculate the other signal where I had an idea in my notes
    # rate = 10
    # signal_1 = np.sum( rate / (neigh_dist_1 - np.sqrt(rate/neigh_dist_1))**2 , axis = 1)
    # signal_2 = np.sum( rate / (neigh_dist_2 - np.sqrt(rate/neigh_dist_2))**2 , axis = 1)
    # signal = signal_1 + signal_2

    #----------CREATE OUTPUT DATAFRAME------------------

    # extract the resulting intensity from the dataframe
    resulting_intensity_1 = df.loc[cond1, "mean_intensity_pole_1"].values
    resulting_intensity_2 = df.loc[cond2, "mean_intensity_pole_2"].values

    # put neighbour signal vs resulting intensity into a dataframe
    pollution_info = pd.DataFrame( columns = ("affected_seg_id","affected_intensity", "combined_signal") )
    seg_ids = np.arange(len(analysis))
    pollution_info.loc[:, "affected_seg_id"] = seg_ids
    pollution_info.loc[ cond_no_problems & cond1_analysis, "affected_intensity" ] = resulting_intensity_1
    pollution_info.loc[ cond_no_problems & cond2_analysis, "affected_intensity" ] = resulting_intensity_2
    
    pollution_info.loc[ cond_no_problems, "combined_signal" ] = signal # save the single signal
    # pollution_info.loc[ cond_no_problems, "combined_signal_dist" ] = signal_dist # save the two seperate signals
    # pollution_info.loc[ cond_no_problems, "combined_signal_int" ] = signal_int


    # drop all the columns that are not part of our selection
    pollution_info = pollution_info.loc[cond_selection, :]

    # drop entries where there is no resulting intensity 
    drop_indices = np.where( np.isnan( pollution_info.loc[:,"affected_intensity"].values.astype(float) ))[0]
    drop_indices = pollution_info.index[drop_indices] # extract the real table's indices with the new indexing after removing nans
    pollution_info = pollution_info.drop(drop_indices , axis = 0)

    # drop the entries where there is no signal (e.g. no neighbours)
    drop_indices = np.where ( pollution_info.loc[:, "combined_signal"] == 0 )
    # drop_indices = np.where ( pollution_info.loc[:, "combined_signal_dist"] == 0 )
    # drop_indices = np.where ( pollution_info.loc[:, "combined_signal_int"] == 0 )

    drop_indices = pollution_info.index[drop_indices] # extract the real table's indices with the new indexing after removing nans
    pollution_info = pollution_info.drop(drop_indices , axis = 0)


    return(pollution_info)


# linear regression of signal vs resulting pollution, preferably of all the lagging poles
def learn_pollution_fct(selection_polluted):

    #---------IMPORT DATA----------------------------------
    # import settings
    import settings
    settings_dict_general = vars(settings.settings_general_fct())
    width = settings_dict_general["width"]
    pcf = settings_dict_general["pcf"]

    # import analysis data
    settings_dict = vars(settings.settings_reversal_analysis_fct())
    path_save_dataframe_dir = settings_dict["path_save_dataframe_dir"]
    path_analysis = path_save_dataframe_dir + "fluo_analysis.csv"
    analysis = pd.read_csv(path_analysis, index_col = 0)

    # restrict to first timeframe
    cond_t0 = analysis.loc[:,"t"] == 0
    analysis = analysis.loc[cond_t0,:]

    #---------FIND NEIGHBOUR INFORMATION FOR EACH BACTERIUM'S LAGGING POLE-------
    pollution_info = find_neighbour_information_learning(selection_polluted, analysis, pcf, width)    
    pollution_info.to_csv(path_save_dataframe_dir + "pollution_info.csv")

    #---------DO A REGRESSION ON BACTERIA INTENSITY DEPENDING ON NEIGHBOUR INFORMATION-------

    # just state which seg ids that were previously given are now excluded
    new_seg_ids = pollution_info.loc[:,"affected_seg_id"].values
    excluded_seg_ids = np.setdiff1d(selection_polluted, new_seg_ids)
    print(str(len(excluded_seg_ids)) + " bacteria were excluded from the training data, since a lagging pole could not be identified with the current detection or there were no neighbours close enough:\n")
    print("That equals " + str(len(excluded_seg_ids) / (len(excluded_seg_ids) + len(new_seg_ids)) * 100) + str(" percent of all bacteria.") )

    # try to find correlation between data
    # affected pole intensity vs combined measure of all neighbours
    import matplotlib.pyplot as plt
    affected_intensity = pollution_info.loc[:,"affected_intensity"].values.astype(float)
    combined_signal = pollution_info.loc[:,"combined_signal"].values.astype(float)
    # combined_signal_dist = pollution_info.loc[:,"combined_signal_dist"].values.astype(float)
    # combined_signal_int = pollution_info.loc[:,"combined_signal_int"].values.astype(float)
    # combined_signal_split = np.column_stack((combined_signal_dist,combined_signal_int))

    corellation = np.corrcoef(combined_signal, affected_intensity)[0,1]
    # corellation_dist = np.corrcoef(combined_signal_dist, affected_intensity)[0,1]
    # corellation_int = np.corrcoef(combined_signal_int, affected_intensity)[0,1]

    # do a linear regression
    import statsmodels.api as sm

    #a general linear model with unknown parameters is provided by this command
    combined_signal_with_constant = sm.add_constant(combined_signal) # do this command to enable the fit to work with a constant beta_0 
    # combined_signal_with_constant_split = sm.add_constant(combined_signal_split) # do this command to enable the fit to work with a constant beta_0 
    model = sm.OLS(affected_intensity, combined_signal_with_constant)
    # model_split = sm.OLS(affected_intensity, combined_signal_with_constant_split)

    #the fitted values B are provided by fitting the model
    results = model.fit()
    print(results.summary())
    regression_parameter = results.params
    plt.figure(1)

    # plot the residuals
    import scipy
    plt.hist(results.resid, bins = 100)
    plt.ylabel("Count")
    plt.xlabel(r"Residual $\varepsilon_i$")
    mean_residuals = np.mean(results.resid)
    plt.axvline(mean_residuals, label = "Mean residual: " + str( np.round(mean_residuals,1)), c = "r")
    plt.legend()
    #normality_p = scipy.stats.normaltest(results.resid).pvalue
    plt.title("Distribution of residuals")
    

    # results = model_split.fit()
    # print(results.summary())
    # regression_parameter = results.params
    # print(regression_parameter)

    #plot the resulting corellation and linear regression
    plt.figure(2)
    plt.title("Correlation = " + str(np.round(corellation,3)))
    plt.scatter( combined_signal, affected_intensity, marker = "x" )
    plt.ylabel("affected intensity")
    plt.xlabel("combined measure: $\sum_{i \in neighbour\ poles} \\frac{intensity_i}{distance_i^2}$")
    x = np.linspace( np.min(combined_signal), np.max(combined_signal), 1000)
    plt.xlim( np.percentile(combined_signal, 1) , np.percentile(combined_signal, 90) ) 

    plt.plot(x, regression_parameter[0] * np.ones(1000) + regression_parameter[1] * x, color = "orange", linewidth = 4)
    plt.show()

    print("\nThe regression constant is: " + str(regression_parameter[0]) + "\nand the coefficient is: " + str(regression_parameter[1]))

    data_save = {"correlation_results": results, 
                 "x_correlation": combined_signal, 
                 "y_correlation": affected_intensity}
    
    import settings
    settings_dict_param_estim = vars(settings.settings_parameter_estimation_fct())
    path_save_image_dir = settings_dict_param_estim["path_save_image_dir"]
    with open(path_save_image_dir + "object_for_pollution_plots.pkl", 'wb') as pickle_file:
        pickle.dump(data_save, pickle_file)

    return(regression_parameter[0], regression_parameter[1])



# find all the properties of any pole for the pollution subtraction process
def find_neighbour_information_subtraction(analysis, pcf, width):

    #----------ONLY LOOK AT SELECTION AND BACTERIA WITHOUT PROBLEMS------------------

    # now find the bacteria with problems to exclude them
    # e.g. wrong pole count, skel too short
    cond_no_problems = np.isnan(analysis.loc[:,"mean_intensity_pole_1"].values) == False
    
    # df will now be the dataframe of the bacteria with no problems
    df = analysis.loc[cond_no_problems , :] 


    #----------FIND ALL NEIGHBOURS OF THE LAGGING POLES AND CALCULATE SIGNAL------------------

    # find a list of all possible pole coordinates in general
    x_min = df.loc[:,"x_min_global"].values
    y_min = df.loc[:,"y_min_global"].values
    all_x_pole_1 = df.loc[:,"pole_1_x_local_pollution"].values + x_min
    all_y_pole_1 = df.loc[:,"pole_1_y_local_pollution"].values + y_min
    all_x_pole_2 = df.loc[:,"pole_2_x_local_pollution"].values + x_min
    all_y_pole_2 = df.loc[:,"pole_2_y_local_pollution"].values + y_min

    # build a tree datastructure of all available poles
    tree_1 = kdtree(np.column_stack((all_x_pole_1, all_y_pole_1)))
    tree_2 = kdtree(np.column_stack((all_x_pole_2, all_y_pole_2)))

    pollution_info = pd.DataFrame( columns = ("affected_seg_id","p1_affected_intensity", "p1_combined_signal","p2_affected_intensity", "p2_combined_signal" ) )
    seg_ids = np.arange(len(analysis))

    for pole_nr in [1,2]:
        
        if pole_nr == 1:
            all_x_pole = all_x_pole_1
            all_y_pole = all_y_pole_1
        else:
            all_x_pole = all_x_pole_2
            all_y_pole = all_y_pole_2

        # find the neighboured pole's indices and distances
        neigh_dist_1, neigh_index_1 = tree_1.query( np.column_stack((all_x_pole, all_y_pole)), k = 20 ) 
        neigh_dist_2, neigh_index_2 = tree_2.query( np.column_stack((all_x_pole, all_y_pole)), k = 20 ) 

        # extract the closest neighbours
        cond_closest_neigh_1 = (neigh_dist_1 < 2 * width / pcf)  & (neigh_dist_1 > 0)
        cond_closest_neigh_2 = (neigh_dist_2 < 2 * width / pcf)  & (neigh_dist_2 > 0)

        # extract the neighbour's intensities
        intensity_1 = df.loc[:, "mean_intensity_pole_1"].values
        intensity_2 = df.loc[:, "mean_intensity_pole_2"].values

        neigh_index_1_flatten = neigh_index_1.flatten()
        neigh_intensity_1 = intensity_1[neigh_index_1_flatten]
        neigh_intensity_1 = np.reshape(neigh_intensity_1, (len(df), 20))
        neigh_index_2_flatten = neigh_index_2.flatten()
        neigh_intensity_2 = intensity_2[neigh_index_2_flatten]
        neigh_intensity_2 = np.reshape(neigh_intensity_2, (len(df), 20))

        # neglect the neighbours that are not close enough
        neigh_intensity_1[~cond_closest_neigh_1] = 0
        neigh_intensity_2[~cond_closest_neigh_2] = 0
        neigh_dist_1[~cond_closest_neigh_1] = np.inf
        neigh_dist_2[~cond_closest_neigh_2] = np.inf

        # calculate the signal
        signal_1 = np.sum( neigh_intensity_1 / neigh_dist_1**2 , axis = 1 ) # all the effect from pole 1s
        signal_2 = np.sum( neigh_intensity_2 / neigh_dist_2**2 , axis = 1 ) # all the effect from pole 1s
        signal = signal_1 + signal_2

        # write stuff into the output dataframe
        # extract the resulting intensity from the dataframe
        resulting_intensity = df.loc[:, "mean_intensity_pole_" + str(pole_nr)].values

        # put neighbour signal vs resulting intensity into a dataframe
        pollution_info.loc[:, "affected_seg_id"] = seg_ids
        pollution_info.loc[ cond_no_problems, "p" + str(pole_nr) + "_affected_intensity" ] = resulting_intensity
        pollution_info.loc[ cond_no_problems, "p" + str(pole_nr) + "_combined_signal" ] = signal


    #----------CREATE OUTPUT DATAFRAME------------------

    # drop entries where there is no resulting intensity 
    drop_indices = np.where( np.isnan( pollution_info.loc[:,"p1_affected_intensity"].values.astype(float) ))[0]
    drop_indices = pollution_info.index[drop_indices] # extract the real table's indices with the new indexing after removing nans
    pollution_info = pollution_info.drop(drop_indices , axis = 0)

    return(pollution_info)







def remove_pollution_fct(regression_constant, regression_coefficient):
    
    #---------IMPORT DATA----------------------------------
    # import settings
    import settings
    settings_general_dict = vars(settings.settings_general_fct())
    width = settings_general_dict["width"]
    pcf = settings_general_dict["pcf"]
    path_seg_dir = settings_general_dict["path_seg_dir"]
    path_fluo_dir = settings_general_dict["path_fluo_dir"]

    settings_dict = vars(settings.settings_fluorescence_detection_fct())
    step_frames = settings_dict["step_frames"]
    frame_end = settings_dict["frame_end"]

    # import analysis data
    settings_dict_2 = vars(settings.settings_reversal_analysis_fct())
    path_save_dataframe_dir = settings_dict_2["path_save_dataframe_dir"]
    path_analysis = path_save_dataframe_dir + "fluo_analysis.csv"
    analysis = pd.read_csv(path_analysis, index_col = 0)
    analysis_corrected = dataframe.create_dataframe_fct()

    # loop over every timeframe

    # only extract of times or all times?
    if frame_end == "all":
        # Count number of images in the directory
        frame_end = count_images.count_and_prepare_images_fct(path_seg_dir, path_fluo_dir)
        # Else: Loop until specified timeframe

    for t in tqdm(range(0, frame_end, step_frames)):
        cond_t = analysis.loc[:,"t"] == t
        analysis_t = analysis.loc[cond_t,:].copy(deep = False)
        analysis_t = analysis_t.reset_index()

        #---------FIND NEIGHBOUR INFORMATION FOR EACH BACTERIUM-------
        pollution_info_t = find_neighbour_information_subtraction(analysis_t, pcf, width)    

        #---------SUBTRACT POLLUTION-------
        analysis_corrected_t = analysis_t.copy(deep = False)
        analysis_corrected_t.loc[:,"mean_intensity_pole_1"] = analysis_t.loc[:,"mean_intensity_pole_1"] - pollution_info_t.loc[:,"p1_combined_signal"] * regression_coefficient - regression_constant
        analysis_corrected_t.loc[:,"mean_intensity_pole_2"] = analysis_t.loc[:,"mean_intensity_pole_2"] - pollution_info_t.loc[:,"p2_combined_signal"] * regression_coefficient - regression_constant

        analysis_corrected = pd.concat( [ analysis_corrected , analysis_corrected_t] )
        analysis_corrected.to_csv(path_save_dataframe_dir + "fluo_analysis_corrected.csv")

    print("Finished removing pollution!")
    return(analysis_corrected)
























































































# # OLD FUNCTIONS WITH FOR LOOPS

# # find all the properties of the lagging pole in the learning process
# def find_neighbour_information_learning_old(selection_polluted, analysis, pollution_info, pcf, width):
#     #---------FIND NEIGHBOUR INFORMATION FOR EACH BACTERIUM-------
#     for seg_index in tqdm(selection_polluted):

#         cond_seg_index = pollution_info.loc[:,"affected_seg_id"] == seg_index
        
#         # consider only those bacteria where a mean intensity could be calculated
#         if np.isnan( analysis.loc[seg_index,"mean_intensity_pole_1"] ) == False:
            
#             # see if the lagging pole is pole 1 or pole 2:
#             leading_pole = analysis.loc[seg_index,"leading_pole"]
#             if leading_pole == 1:
#                 lagging_pole = 2
#                 do_not_consider = False
#             elif leading_pole == 2:
#                 lagging_pole = 1
#                 do_not_consider = False
#             else:
#                 # case that the leading pole is nan
#                 do_not_consider = True

#             if do_not_consider == False:

#                 # extract global pole coords
#                 pole_x_global = analysis.loc[seg_index,"pole_" + str(lagging_pole) + "_x_local"] + analysis.loc[seg_index,"x_min_global"]
#                 pole_y_global = analysis.loc[seg_index,"pole_" + str(lagging_pole) + "_y_local"] + analysis.loc[seg_index,"y_min_global"]

#                 # find poles close to them
#                 # all distances to pole 1s
#                 distances_1 = np.sqrt((pole_x_global - (analysis.loc[:,"pole_1_x_local"] + analysis.loc[:,"x_min_global"]))**2 + \
#                     (pole_y_global - (analysis.loc[:,"pole_1_y_local"] + analysis.loc[:,"y_min_global"]))**2)
#                 # all distances to pole 2s
#                 distances_2 = np.sqrt((pole_x_global - (analysis.loc[:,"pole_2_x_local"] + analysis.loc[:,"x_min_global"]))**2 + \
#                     (pole_y_global - (analysis.loc[:,"pole_2_y_local"] + analysis.loc[:,"y_min_global"]))**2)
#                 # all pole 1 close to them
#                 neighbour_close_table_1 = distances_1 < 2 * width / pcf 
#                 # all pole 2 close to them
#                 neighbour_close_table_2 = distances_2 < 2 * width / pcf 

#                 # find the indices of those poles close to them
#                 neighbour_indices_1 = np.array(neighbour_close_table_1[neighbour_close_table_1 == True].index)
#                 neighbour_indices_2 = np.array(neighbour_close_table_2[neighbour_close_table_2 == True].index)

#                 # now exclude the index of the current seg_id
#                 neighbour_indices_1 =  neighbour_indices_1[neighbour_indices_1 != seg_index]
#                 neighbour_indices_2 =  neighbour_indices_2[neighbour_indices_2 != seg_index]   # subtract the index of the bacterium itsself

#                 # next, check if the neighbours are valid. It could for example be, that the neighbour already have warnings, e.g. no mean fluo intensity
#                 cond_neigbours_1_good = np.isnan( analysis.loc[neighbour_indices_1, "mean_intensity_pole_1"].values  ) == False
#                 cond_neigbours_2_good = np.isnan( analysis.loc[neighbour_indices_2, "mean_intensity_pole_1"].values  ) == False
#                 neighbour_indices_1 = neighbour_indices_1[cond_neigbours_1_good]
#                 neighbour_indices_2 = neighbour_indices_2[cond_neigbours_2_good]

#                 # extract their pole intensities
#                 neighbour_intensities_1 = analysis.loc[neighbour_indices_1, "mean_intensity_pole_1"]
#                 neighbour_intensities_2 = analysis.loc[neighbour_indices_2, "mean_intensity_pole_2"]

#                 # extract their distances
#                 neighbour_distances_1 = distances_1[neighbour_indices_1]
#                 neighbour_distances_2 = distances_2[neighbour_indices_2]

#                 # now merge all of the data obtained for both poles
#                 neighbour_indices = np.concatenate([neighbour_indices_1 , neighbour_indices_2])
#                 neighbour_intensities = np.concatenate([neighbour_intensities_1 , neighbour_intensities_2])
#                 neighbour_distances = np.concatenate([neighbour_distances_1 , neighbour_distances_2])

#                 # calculate the combined measure
#                 if len(neighbour_indices) != 0:
#                     combined_signal = np.sum( neighbour_intensities / neighbour_distances**2 )
                
#                 # write a warning when no neighbours were found
#                 else: 
#                     combined_signal = np.nan
#                     pollution_info.at[pollution_index, "error_msg"] = "No neighbours were found in the given distance."


#                 # save information in dataframe
#                 pollution_index = np.array(pollution_info[cond_seg_index].index[0]).item() # writing an array does not work via conditions, so we use the pollution index
#                 pollution_info.loc[cond_seg_index,"affected_intensity"] = analysis.loc[seg_index,"mean_intensity_pole_" + str(lagging_pole)]
#                 pollution_info.at[pollution_index, "neighbour_seg_id"] = neighbour_indices
#                 pollution_info.at[pollution_index, "neighbour_intensity"] = neighbour_intensities
#                 pollution_info.at[pollution_index, "neighbour_distance"] = neighbour_distances
#                 pollution_info.loc[pollution_index, "combined_signal"] = combined_signal


#             # in the case that the lagging pole could not be easily identified and the entry is skipped
#             # we write a [], so that the drop index function later works
#             else:
#                 pollution_index = np.array(pollution_info[cond_seg_index].index[0]).item() # writing an array does not work via conditions, so we use the pollution index
#                 pollution_info.at[pollution_index, "neighbour_intensity"] = np.array([])
#                 pollution_info.at[pollution_index, "neighbour_seg_id"] = np.array([])
#                 pollution_info.at[pollution_index, "error_msg"] = "The lagging pole could not be identified with the previous procedure."

#         else:
#             # save information in dataframe
#             pollution_index = np.array(pollution_info[cond_seg_index].index[0]).item() # writing an array does not work via conditions, so we use the pollution index
#             pollution_info.at[pollution_index, "neighbour_seg_id"] = np.array([])
#             pollution_info.at[pollution_index, "neighbour_intensity"] = np.array([])
#             pollution_info.at[pollution_index, "neighbour_distance"] = np.array([])
#             pollution_info.at[pollution_index, "combined_signal"] = np.nan
#             pollution_info.at[pollution_index, "error_msg"] = "There was no mean pole intensity."

#     return(pollution_info)





# # find all the properties of any pole for the pollution subtraction process
# def find_neighbour_information_subtraction_old(selection_polluted, analysis, pollution_info, pcf, width):
#     #---------FIND NEIGHBOUR INFORMATION FOR EACH BACTERIUM-------
#     for seg_index in selection_polluted:

#         cond_seg_index = pollution_info.loc[:,"affected_seg_id"] == seg_index
#         # consider only those bacteria where a mean intensity could be calculated
#         if np.isnan( analysis.loc[seg_index,"mean_intensity_pole_1"] ) == False:
            
#             for pole_nr in [1,2]:
#                 # extract global pole coords
#                 pole_x_global = analysis.loc[seg_index,"pole_" + str(pole_nr) + "_x_local"] + analysis.loc[seg_index,"x_min_global"]
#                 pole_y_global = analysis.loc[seg_index,"pole_" + str(pole_nr) + "_y_local"] + analysis.loc[seg_index,"y_min_global"]

#                 # find poles close to them
#                 # all distances to pole 1s
#                 distances_1 = np.sqrt((pole_x_global - (analysis.loc[:,"pole_1_x_local"] + analysis.loc[:,"x_min_global"]))**2 + \
#                     (pole_y_global - (analysis.loc[:,"pole_1_y_local"] + analysis.loc[:,"y_min_global"]))**2)
#                 # all distances to pole 2s
#                 distances_2 = np.sqrt((pole_x_global - (analysis.loc[:,"pole_2_x_local"] + analysis.loc[:,"x_min_global"]))**2 + \
#                     (pole_y_global - (analysis.loc[:,"pole_2_y_local"] + analysis.loc[:,"y_min_global"]))**2)
#                 # all pole 1 close to them
#                 neighbour_close_table_1 = distances_1 < 2 * width / pcf 
#                 # all pole 2 close to them
#                 neighbour_close_table_2 = distances_2 < 2 * width / pcf 

#                 # find the indices of those poles close to them
#                 neighbour_indices_1 = np.array(neighbour_close_table_1[neighbour_close_table_1 == True].index)
#                 neighbour_indices_2 = np.array(neighbour_close_table_2[neighbour_close_table_2 == True].index)

#                 # now exclude the index of the current seg_id
#                 neighbour_indices_1 =  neighbour_indices_1[neighbour_indices_1 != seg_index]
#                 neighbour_indices_2 =  neighbour_indices_2[neighbour_indices_2 != seg_index]   # subtract the index of the bacterium itsself

#                 # next, check if the neighbours are valid. It could for example be, that the neighbour already have warnings, e.g. no mean fluo intensity
#                 cond_neigbours_1_good = np.isnan( analysis.loc[neighbour_indices_1, "mean_intensity_pole_1"].values  ) == False
#                 cond_neigbours_2_good = np.isnan( analysis.loc[neighbour_indices_2, "mean_intensity_pole_1"].values  ) == False
#                 neighbour_indices_1 = neighbour_indices_1[cond_neigbours_1_good]
#                 neighbour_indices_2 = neighbour_indices_2[cond_neigbours_2_good]

#                 # extract their pole intensities
#                 neighbour_intensities_1 = analysis.loc[neighbour_indices_1, "mean_intensity_pole_1"]
#                 neighbour_intensities_2 = analysis.loc[neighbour_indices_2, "mean_intensity_pole_2"]

#                 # extract their distances
#                 neighbour_distances_1 = distances_1[neighbour_indices_1]
#                 neighbour_distances_2 = distances_2[neighbour_indices_2]

#                 # now merge all of the data obtained for both poles
#                 neighbour_indices = np.concatenate([neighbour_indices_1 , neighbour_indices_2])
#                 neighbour_intensities = np.concatenate([neighbour_intensities_1 , neighbour_intensities_2])
#                 neighbour_distances = np.concatenate([neighbour_distances_1 , neighbour_distances_2])


#                 # calculate the combined measure
#                 if len(neighbour_indices) != 0:
#                     combined_signal = 0
#                     for i in range(len(neighbour_distances)):
#                         combined_signal = combined_signal + neighbour_intensities[i] / neighbour_distances[i]**2

#                 # write a warning when no neighbours were found
#                 else: 
#                     combined_signal = 0 # in this case we write it as 0 instead of nan to just subtract zero from the analysis matrix at this point
#                     pollution_info.at[pollution_index, "p" + str(pole_nr) + "_error_msg"] = "No neighbours were found in the given distance."


#                 # save information in dataframe
#                 pollution_index = np.array(pollution_info[cond_seg_index].index[0]).item() # writing an array does not work via conditions, so we use the pollution index
#                 pollution_info.loc[cond_seg_index,"p" + str(pole_nr) + "_affected_intensity"] = analysis.loc[seg_index,"mean_intensity_pole_" + str(pole_nr)]
#                 pollution_info.at[pollution_index, "p" + str(pole_nr) + "_neighbour_seg_id"] = neighbour_indices
#                 pollution_info.at[pollution_index, "p" + str(pole_nr) + "_neighbour_intensity"] = neighbour_intensities
#                 pollution_info.at[pollution_index, "p" + str(pole_nr) + "_neighbour_distance"] = neighbour_distances
#                 pollution_info.loc[pollution_index, "p" + str(pole_nr) + "_combined_signal"] = combined_signal

#         else:
#             # case that no mean pole intensity could be calculated. corresponds to an exclusion in the main file
#             # save information in dataframe
#             pollution_index = np.array(pollution_info[cond_seg_index].index[0]).item() # writing an array does not work via conditions, so we use the pollution index
#             pollution_info.at[pollution_index, "p1_neighbour_seg_id"] = np.array([]).astype(object)
#             pollution_info.at[pollution_index, "p2_neighbour_seg_id"] = np.array([]).astype(object)
#             pollution_info.at[pollution_index, "p1_neighbour_intensity"] = np.array([]).astype(object)
#             pollution_info.at[pollution_index, "p2_neighbour_intensity"] = np.array([]).astype(object)
#             pollution_info.at[pollution_index, "p1_neighbour_distance"] = np.array([]).astype(object)
#             pollution_info.at[pollution_index, "p2_neighbour_distance"] = np.array([]).astype(object)
#             pollution_info.at[pollution_index, "p1_combined_signal"] = np.nan
#             pollution_info.at[pollution_index, "p2_combined_signal"] = np.nan
#             pollution_info.at[pollution_index, "p1_error_msg"] = "There was no mean pole intensity."
#             pollution_info.at[pollution_index, "p2_error_msg"] = "There was no mean pole intensity."


#     return(pollution_info)