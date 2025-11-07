#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plt_trajectory_fct(merged_df, track_id):

    # select all data for this track id and plot
    cond_track_id = merged_df.loc[:,"track_id"] == track_id
    selection = merged_df.loc[cond_track_id,:]
    nr_frames = len(selection)
    plt.figure()
    plt.title("Trajectory of skeleton center over " + str(nr_frames) + " frames of track-id number " + str(track_id))
    plt.plot ( selection.loc[:,"center_x"].values , selection.loc[:,"center_y"].values )

    

plt_trajectory_fct(merged_df, 513)

#512
# %%
