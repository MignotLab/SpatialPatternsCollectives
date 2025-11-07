#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import reversal dataframe 
path_fluo_df = "C:/Joni/Uni/Vorlesungen/AMU/Internship_Mignot_Lab/Fluorescence_Detection/output/dataframes/merged_df.csv"
fluo_df = pd.read_csv(path_fluo_df, index_col = 0)

# create an array of global time between reversals
time_between_reversals = np.array([])

# see available track ids
available_track_ids = np.unique(df_merged.loc[:,"track_id"].values)

# find the tbr for one track id
for track_id in available_track_ids:
    cond_track_id = fluo_df.loc[:,"track_id"] == track_id
    reversal_array = fluo_df.loc[cond_track_id,"reversal"].values.astype(bool)
    distance_array = np.arange(reversal_array.shape[0])
    distance_at_reversals = distance_array[reversal_array]
    time_between_reversals_track_id = distance_at_reversals[1:] - distance_at_reversals[:-1]
    time_between_reversals = np.append(time_between_reversals, time_between_reversals_track_id)

# convert to physical units:
# each 1 unit in the reversal array stands for 5 frames
# each frame stands for 2 seconds
time_between_reversals_frames = time_between_reversals * 5
time_between_reversals_seconds = time_between_reversals * 10

#%%
mean_tbr = np.mean(time_between_reversals_frames)
sd_tbr = np.std(time_between_reversals_frames)
percentile = np.percentile(time_between_reversals_frames, 5)

plt.hist(time_between_reversals_frames, bins = 30, density = True)
plt.vlines(percentile, ymin = 0, ymax = 0.006, color = "orange")
print(str(percentile) + " frames")

