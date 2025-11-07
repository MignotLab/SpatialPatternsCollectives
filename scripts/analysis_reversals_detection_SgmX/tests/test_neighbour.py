#%%
import pandas as pd
import numpy as np
from tqdm import tqdm
import count_images
import dataframe
from scipy.spatial import KDTree as kdtree

#----------FIND ALL THE BACTERIA WITHOUT PROBLEMS------------------
# THEN JUST CONTINUE TO WORK WITH THE GOOD BACTERIA
# e.g. wrong pole count, skel too short
cond_no_problems = np.isnan(analysis.loc[:,"mean_intensity_pole_1"].values) == False

# find also the lagging poles of the analysis
cond1_analysis = analysis.loc[:,"leading_pole"].values == 2
cond2_analysis = analysis.loc[:,"leading_pole"].values == 1
df = analysis.loc[cond_no_problems, :]

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
x_pole_1_global = df.loc[cond1, "pole_1_x_local"].values + df.loc[cond1, "x_min_global"].values
y_pole_1_global = df.loc[cond1, "pole_1_y_local"].values + df.loc[cond1, "y_min_global"].values
x_pole_2_global = df.loc[cond2, "pole_2_x_local"].values + df.loc[cond2, "x_min_global"].values
y_pole_2_global = df.loc[cond2, "pole_2_y_local"].values + df.loc[cond2, "y_min_global"].values

# merge the two lagging pole lists in the right order via cond
x_lagging_pole = np.zeros(len(df))
y_lagging_pole = np.zeros(len(df))
x_lagging_pole[cond1] = x_pole_1_global
x_lagging_pole[cond2] = x_pole_2_global
y_lagging_pole[cond1] = y_pole_1_global
y_lagging_pole[cond2] = y_pole_2_global

#----------FIND ALL NEIGHBOURS OF THE LAGGING POLES------------------

# find a list of all possible pole coordinates in general
x_min = df.loc[:,"x_min_global"].values
y_min = df.loc[:,"y_min_global"].values
all_x_pole_1 = df.loc[:,"pole_1_x_local"].values + x_min
all_y_pole_1 = df.loc[:,"pole_1_y_local"].values + y_min
all_x_pole_2 = df.loc[:,"pole_2_x_local"].values + x_min
all_y_pole_2 = df.loc[:,"pole_2_y_local"].values + y_min

# build a tree datastructure of all available poles
tree_1 = kdtree(np.column_stack((all_x_pole_1, all_y_pole_1)))
tree_2 = kdtree(np.column_stack((all_x_pole_2, all_y_pole_2)))

# find the neighboured lagging pole's indices and distances
neigh_dist_1, neigh_index_1 = tree_1.query( np.column_stack((x_lagging_pole,y_lagging_pole)), k = 10 ) 
neigh_dist_2, neigh_index_2 = tree_2.query( np.column_stack((x_lagging_pole,y_lagging_pole)), k = 10 ) 

# extract the closest neighbours
cond_closest_neigh_1 = (neigh_dist_1 < 2 * width / pcf)  & (neigh_dist_1 > 0)
cond_closest_neigh_2 = (neigh_dist_2 < 2 * width / pcf)  & (neigh_dist_2 > 0)

# extract the neighbour's intensities
intensity_1 = df.loc[:, "mean_intensity_pole_1"].values
intensity_2 = df.loc[:, "mean_intensity_pole_2"].values

neigh_index_1_flatten = neigh_index_1.flatten()
neigh_intensity_1 = intensity_1[neigh_index_1_flatten]
neigh_intensity_1 = np.reshape(neigh_intensity_1, (len(df), 10))
neigh_index_2_flatten = neigh_index_2.flatten()
neigh_intensity_2 = intensity_2[neigh_index_2_flatten]
neigh_intensity_2 = np.reshape(neigh_intensity_2, (len(df), 10))

# neglect the neighbours that are not close enough
neigh_intensity_1[~cond_closest_neigh_1] = 0
neigh_intensity_2[~cond_closest_neigh_2] = 0
neigh_dist_1[~cond_closest_neigh_1] = np.inf
neigh_dist_2[~cond_closest_neigh_2] = np.inf

# calculate the signal
signal_1 = np.sum( neigh_intensity_1 / neigh_dist_1**2 , axis = 1 ) # all the effect from pole 1s
signal_2 = np.sum( neigh_intensity_2 / neigh_dist_2**2 , axis = 1 ) # all the effect from pole 1s
signal = signal_1 + signal_2

# extract the resulting intensity from the dataframe
resulting_intensity_1 = df.loc[cond1, "mean_intensity_pole_1"].values
resulting_intensity_2 = df.loc[cond2, "mean_intensity_pole_2"].values

# put everything in the end into a dataframe
pollution_info_new = pd.DataFrame( columns = ("affected_seg_id","signal") )
seg_ids = np.arange(len(analysis))
pollution_info_new.loc[:, "affected_seg_id"] = seg_ids
pollution_info_new.loc[ cond_no_problems & cond1_analysis, "affected_intensity" ] = resulting_intensity_1
pollution_info_new.loc[ cond_no_problems & cond2_analysis, "affected_intensity" ] = resulting_intensity_2
pollution_info_new.loc[ cond_no_problems, "signal" ] = signal
#%%
drop_indices = np.where( np.isnan( pollution_info_new.loc[:,"affected_intensity"].values ))[0]
drop_indices = pollution_info_new.index[drop_indices] # extract the real table's indices with the new indexing after removing nans
pollution_info_new = pollution_info_new.drop(drop_indices , axis = 0)

drop_indices = np.where ( pollution_info_new.loc[:, "signal"] == 0 )
drop_indices = pollution_info_new.index[drop_indices] # extract the real table's indices with the new indexing after removing nans
pollution_info_new = pollution_info_new.drop(drop_indices , axis = 0)


# %%
