#%%

import pandas as pd
import numpy as np

# import settings
import settings
settings_dict = vars(settings.settings_fluorescence_detection_fct())
width = settings_dict["width"]
pcf = settings_dict["pcf"]

# import analysis data
path_dataframes_dir = list(vars(settings.settings_reversal_analysis_fct()).values())[0]
path_analysis = path_dataframes_dir + "analysis.csv"
analysis = pd.read_csv(path_analysis)

# restrict to first timeframe
cond_t0 = analysis.loc[:,"t"] == 0
analysis = analysis.loc[cond_t0,:]

# construct dataframe for each bacteria's information
pollution_training = pd.DataFrame(columns=("affected_seg_id","affected_intensity","neighbour_seg_id","neighbour_intensity","neighbour_distance"))

# select those bacteria that are polluted at their lagging pole
selection_polluted = [1274, 1306, 1109, 1194, 1389, 1528, 1493, 1642, 1280, 226, 1127, 1257, 1268, 1520, 1493, 1642, 1595, 1838, 2690, 1421, 1427, 1425, 1509, 1565, 1572, 1495, 1451, 1157, 1092, 961, 932, 845, 650, 826, 799, 742, 554, 429, 202, 245, 323, 457, 778, 1196, 1263, 1305, 1153, 1053, 1107, 1451, 1347, 1331, 1318, 1255, 1089, 1092, 1026, 1008, 1065, 914, 797, 803, 507, 400, 794, 1512, 1924 ]
#selection_polluted = []
selection_polluted = list(set(selection_polluted)) #remove doubles
selection_polluted = np.sort(selection_polluted) #sort

# we can already fill the affected intensity and seg id
pollution_training.loc[:,"affected_seg_id"] = selection_polluted


for seg_index in selection_polluted:

    # consider only those bacteria where a mean intensity could be calculated
    cond_seg_index = pollution_training.loc[:,"affected_seg_id"] == seg_index
    if np.isnan( analysis.loc[seg_index,"mean_intensity_pole_1"] ) == False:
        
        # see if the lagging pole is pole 1 or pole 2:
        leading_pole = analysis.loc[seg_index,"leading_pole"]
        if leading_pole == 1:
            lagging_pole = 2
            do_not_consider = False
        elif leading_pole == 2:
            lagging_pole = 1
            do_not_consider = False
        else:
            do_not_consider = True

        if do_not_consider == False:

            # extract affected pole intensity
            pollution_training.loc[cond_seg_index,"affected_intensity"] = analysis.loc[seg_index,"mean_intensity_pole_" + str(lagging_pole)]

            # extract global pole coords
            pole_x_global = analysis.loc[seg_index,"pole_" + str(lagging_pole) + "_x_local"] + analysis.loc[seg_index,"x_min_global"]
            pole_y_global = analysis.loc[seg_index,"pole_" + str(lagging_pole) + "_y_local"] + analysis.loc[seg_index,"y_min_global"]

            # find poles close to them
            # all distances to pole 1s
            distances_1 = np.sqrt((pole_x_global - (analysis.loc[:,"pole_1_x_local"] + analysis.loc[:,"x_min_global"]))**2 + \
                (pole_y_global - (analysis.loc[:,"pole_1_y_local"] + analysis.loc[:,"y_min_global"]))**2)
            # all distances to pole 2s
            distances_2 = np.sqrt((pole_x_global - (analysis.loc[:,"pole_2_x_local"] + analysis.loc[:,"x_min_global"]))**2 + \
                (pole_y_global - (analysis.loc[:,"pole_2_y_local"] + analysis.loc[:,"y_min_global"]))**2)
            # all pole 1 close to them
            close_table_1 = distances_1 < 2 * width / pcf 
            # all pole 2 close to them
            close_table_2 = distances_2 < 2 * width / pcf 

            # find the indices of those poles close to them
            close_indices_1 = np.array(close_table_1[close_table_1 == True].index)
            close_indices_2 = np.array(close_table_2[close_table_2 == True].index)

            # now exclude the index of the current seg_id
            close_indices_1 =  close_indices_1[close_indices_1 != seg_index]
            close_indices_2 =  close_indices_2[close_indices_2 != seg_index]   # subtract the index of the bacterium itsself

            # extract their pole intensities
            close_intensities_1 = analysis.loc[close_indices_1, "mean_intensity_pole_1"]
            close_intensities_2 = analysis.loc[close_indices_2, "mean_intensity_pole_2"]

            # extract their distances
            close_distances_1 = distances_1[close_indices_1]
            close_distances_2 = distances_2[close_indices_2]

            # now merge all of the data obtained for both poles
            close_indices = np.concatenate([close_indices_1 , close_indices_2])
            close_intensities = np.concatenate([close_intensities_1 , close_intensities_2])
            close_distances = np.concatenate([close_distances_1 , close_distances_2])

            # save information in dataframe
            pollution_index = np.array(pollution_training[cond_seg_index].index[0]).item() # it does not work with conditions to write an array at this place, so we use the pollution index
            pollution_training.at[pollution_index, "neighbour_seg_id"] = close_indices
            pollution_training.at[pollution_index, "neighbour_intensity"] = close_intensities
            pollution_training.at[pollution_index, "neighbour_distance"] = close_distances

        # in the case that the lagging pole could not be easily identified and the entry is skipped
        # we write a [], so that the drop index function later works
        else:
            pollution_training.at[pollution_index, "neighbour_intensity"] = np.array([])

    else:
        # save information in dataframe
        pollution_training.at[cond_seg_index, "neighbour_seg_id"] = np.array([])
        pollution_training.at[cond_seg_index, "neighbour_intensity"] = np.array([])
        pollution_training.at[cond_seg_index, "neighbour_distance"] = np.array([])


# remove all the entries from the dataframe where no close enough neighbour was detected:
pollution_training = pollution_training.dropna() #drop nans. nans come from all data that has been excluded 
drop_indices = np.where(np.array([len(a) for a in pollution_training.loc[:,"neighbour_intensity"].values]) == 0)[0]
drop_indices = pollution_training.index[drop_indices] # extract the real table's indices with the new indexing after removing nans
pollution_training = pollution_training.drop(drop_indices , axis = 0)

# just state which seg ids that were previously given are now excluded
new_seg_ids = pollution_training.loc[:,"affected_seg_id"].values
excluded_seg_ids = np.setdiff1d(selection_polluted, new_seg_ids)
print("These seg ids were now excluded, since a lagging pole could not be exactly identified or there were no neighbours close enough:\n" + str(excluded_seg_ids))

# try to plot a correlations
import matplotlib.pyplot as plt
affected_intensity = pollution_training.loc[:,"affected_intensity"].values.astype(float)

# affected pole intensity vs polluter's intensity
plt.figure()
one_neighbour_intensity = np.array([a[0] for a in pollution_training.loc[:,"neighbour_intensity"].values])
corellation = np.corrcoef(one_neighbour_intensity, affected_intensity)[0,1]
plt.title("Correlation = " + str(corellation))
plt.scatter(one_neighbour_intensity, affected_intensity )
plt.ylabel("affected intensity")
plt.xlabel("polluter's intensity (only one neighbour taken)")

# affected pole intensity vs polluter's distance
plt.figure()
one_neighbour_distance = np.array([a[0] for a in pollution_training.loc[:,"neighbour_distance"].values])
corellation = np.corrcoef(one_neighbour_distance, affected_intensity)[0,1]
plt.title("Correlation = " + str(corellation))
plt.scatter(one_neighbour_distance, affected_intensity)
plt.ylabel("affected intensity")
plt.xlabel("polluter's distance (only one neighbour taken)")

# affected pole intensity vs combined measure of one neighbour
plt.figure()
combined_measure = one_neighbour_intensity / one_neighbour_distance**2
corellation = np.corrcoef(combined_measure, affected_intensity)[0,1]
plt.title("Correlation = " + str(corellation))
plt.scatter( combined_measure, affected_intensity )
plt.ylabel("affected intensity")
plt.xlabel("combined measure (only one neighbour taken)")

#-----------all neighbours taken----------
# affected pole intensity vs combined measure of one neighbour
plt.figure()

combined_measure = np.array([])
neighbour_distance = np.array(pollution_training.loc[:,"neighbour_distance"].values)
neighbour_intensity = np.array(pollution_training.loc[:,"neighbour_intensity"].values)
for i in range(len(pollution_training.loc[:,"neighbour_distance"].values)):
    combined_measure = np.append( combined_measure, np.sum(neighbour_intensity[i] / neighbour_distance[i]**2) )
corellation = np.corrcoef(combined_measure, affected_intensity)[0,1]
plt.title("Correlation = " + str(corellation))
plt.scatter( combined_measure, affected_intensity )
plt.ylabel("affected intensity")
plt.xlabel("combined measure: $\sum_{neighbour\ poles} intensity * 1/distance^2$")

# %%
# do a linear regression
import statsmodels.api as sm

#a general linear model with unknown parameters is provided by this command
model = sm.OLS(affected_intensity,combined_measure)

#the fitted values B are provided by fitting the model
results = model.fit()
parameter = results.params

#print(results.summary())
plt.title("Correlation = " + str(corellation))
plt.scatter( combined_measure, affected_intensity )
plt.ylabel("affected intensity")
plt.xlabel("combined measure: $\sum_{neighbour\ poles} intensity * 1/distance^2$")
x = np.linspace( np.min(combined_measure), np.max(combined_measure), 1000)
plt.plot(x, parameter * x)



"""
Wichtig ist, bei der Trainingsdata, dass die Leading poles richtig erkannt wurden.
Was passiert sonst? Wir wollen immer nur auf verschmutzte lagging poles schauen.
So können wir sagen, durch das signal von stärke 5 in 10 pixeln entfernung haben wir nun 
eine verschmutzung von 3. Wenn wir aber versehentlich einen leading pole in die daten
einwerfen, dann hätten wir auch mal bei sehr weit entfernten und schwachen nachbarn ein signal
von 5, was einfach durch den leading pole kommt. So kann das programm nicht richtig lernen.
"""

# this parameter now tells us that 

# %%
