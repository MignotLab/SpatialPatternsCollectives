# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fractions import Fraction as frac
from matplotlib.ticker import FuncFormatter, MultipleLocator

def pi_axis_formatter(val, pos, denomlim=10, pi=r'\pi'):
    """
    format label properly
    for example: 0.6666 pi --> 2π/3
               : 0      pi --> 0
               : 0.50   pi --> π/2  
    """
    minus = "-" if val < 0 else ""
    val = abs(val)
    ratio = frac(val/np.pi).limit_denominator(denomlim)
    n, d = ratio.numerator, ratio.denominator
    
    fmt2 = "%s" % d 
    if n == 0:
        fmt1 = "0"
    elif n == 1:
        fmt1 = pi
    else:
        fmt1 = r"%s%s" % (n,pi)
        
    fmtstring = "$" + minus + (fmt1 if d == 1 else r"{%s}/{%s}" % (fmt1, fmt2)) + "$"
    
    return fmtstring

# %%
path = "W:/jb/a_thesis/python/notebook/Myxo/csv_tracking_data/2021_03_05/5h/cell_tracking/DM31/2021_03_05_DZ2_DM31-3p_617-05p_20X_5h_tb2f=20s_DM31.csv"
df = pd.read_csv(path)
track_id_column = "TrackID"
t_column = "t"
x_column = "x"
y_column = "y"
df.sort_values(by=[track_id_column,t_column],ignore_index=True,inplace=True)
# %%
scale = 748.8/2304 #µm / px
tbf = 1/3 # minutes / frame
x_medium = np.max(df.loc[:,x_column]) / 2
cond_rippling = (df.loc[:,x_column] < x_medium * 0.6).values[:-1]
cond_swarming = (df.loc[:,x_column] > x_medium * 1.4).values[:-1]
cond_same_track = df.loc[:,track_id_column].values[1:] - df.loc[:,track_id_column].values[:-1] == 0

direction_x = df.loc[:,x_column].values[1:] - df.loc[:,x_column].values[:-1]
direction_y = df.loc[:,y_column].values[1:] - df.loc[:,y_column].values[:-1]
velocity = np.linalg.norm(np.array([direction_x[cond_same_track],direction_y[cond_same_track]]), axis=0) * scale / tbf
angle_direction = np.arctan2(direction_y[cond_same_track],direction_x[cond_same_track])
angle_direction[angle_direction < 0] += np.pi
angle_direction += np.pi/2
angle_direction[angle_direction > np.pi] -= np.pi

cond_rippling = cond_rippling[cond_same_track]
cond_swarming = cond_swarming[cond_same_track]
# %%
def plot_hist(data,title,x_label,y_label,fontsize):
    """
    Plot an histogram
    
    """
    fontsize = 30
    width_bin = 2*np.pi / 64
    bins_angle = np.arange(0,np.pi+width_bin,width_bin)
    max_vel = 10
    width_bin_vel = 0.4
    bins_vel = np.arange(0,max_vel+width_bin_vel,width_bin_vel)

    # Orientation
    fig, ax = plt.subplots(figsize=(8,6))
    label_mean_tbr = 'Mean = '+str(np.round(np.nanmean(angle_direction),1))+' rad'
    n = ax.hist(angle_direction,bins=bins_angle,label=label_mean_tbr, density=True, alpha=0.4, histtype='bar', ec='black',color="royalblue")
    ax.set_xlabel("Orientation (rad)",fontsize=fontsize)
    ax.set_ylabel("Density",fontsize=fontsize)
    ax.set_xlim(0,np.pi)
    # setting ticks labels
    ax.xaxis.set_major_formatter(FuncFormatter(pi_axis_formatter))
    # setting ticks at proper numbers
    ax.xaxis.set_major_locator(MultipleLocator(base=np.pi/4))
    ax.tick_params(labelsize=fontsize)
    # ax.legend(loc='best',fontsize=fontsize/1.5)
    # Velocity
    fig, ax = plt.subplots(figsize=(8,6))
    label_mean_tbr = 'Mean = '+str(np.round(np.nanmean(velocity),1))+' $\mu$m/min'
    n = ax.hist(velocity,bins=bins_vel,label=label_mean_tbr, density=True, alpha=0.4, histtype='bar', ec='black',color="royalblue")
    ax.set_xlabel("Velocity ($\mu$m/min)",fontsize=fontsize)
    ax.set_ylabel("Density",fontsize=fontsize)
    ax.set_xlim(0,max_vel)
    ax.tick_params(labelsize=fontsize)
    ax.legend(loc='best',fontsize=fontsize/1.5)

    # Orientation
    angle_direction_rip = angle_direction[cond_rippling]
    fig, ax = plt.subplots(figsize=(8,6))
    label_mean_tbr = 'Mean = '+str(np.round(np.nanmean(angle_direction_rip),1))+' rad'
    n = ax.hist(angle_direction_rip,bins=bins_angle,label=label_mean_tbr, density=True, alpha=0.4, histtype='bar', ec='black',color="royalblue")
    ax.set_title("Rippling",fontsize=fontsize*1.5)
    ax.set_xlabel("Orientation (rad)",fontsize=fontsize)
    ax.set_ylabel("Density",fontsize=fontsize)
    ax.set_xlim(0,np.pi)
    # setting ticks labels
    ax.xaxis.set_major_formatter(FuncFormatter(pi_axis_formatter))
    # setting ticks at proper numbers
    ax.xaxis.set_major_locator(MultipleLocator(base=np.pi/4))
    ax.tick_params(labelsize=fontsize)
    # ax.legend(loc='best',fontsize=fontsize/1.5)
    # Velocity
    velocity_rip = velocity[cond_rippling]
    fig, ax = plt.subplots(figsize=(8,6))
    label_mean_tbr = 'Mean = '+str(np.round(np.nanmean(velocity_rip),1))+' $\mu$m/min'
    n = ax.hist(velocity_rip,bins=bins_vel,label=label_mean_tbr, density=True, alpha=0.4, histtype='bar', ec='black',color="royalblue")
    ax.set_title("Rippling",fontsize=fontsize*1.5)
    ax.set_xlabel("Velocity ($\mu$m/min)",fontsize=fontsize)
    ax.set_ylabel("Density",fontsize=fontsize)
    ax.set_xlim(0,max_vel)
    ax.tick_params(labelsize=fontsize)
    ax.legend(loc='best',fontsize=fontsize/1.5)

    # Orientation
    angle_direction_swarm = angle_direction[cond_swarming]
    fig, ax = plt.subplots(figsize=(8,6))
    label_mean_tbr = 'Mean = '+str(np.round(np.nanmean(angle_direction_swarm),1))+' rad'
    n = ax.hist(angle_direction_swarm,bins=bins_angle,label=label_mean_tbr, density=True, alpha=0.4, histtype='bar', ec='black',color="royalblue")
    ax.set_title("Swarming",fontsize=fontsize*1.5)
    ax.set_xlabel("Orientation (rad)",fontsize=fontsize)
    ax.set_ylabel("Density",fontsize=fontsize)
    ax.set_xlim(0,np.pi)
    # setting ticks labels
    ax.xaxis.set_major_formatter(FuncFormatter(pi_axis_formatter))
    # setting ticks at proper numbers
    ax.xaxis.set_major_locator(MultipleLocator(base=np.pi/4))
    ax.tick_params(labelsize=fontsize)
    # ax.legend(loc='best',fontsize=fontsize/1.5)
    # Velocity
    velocity_swarm = velocity[cond_swarming]
    fig, ax = plt.subplots(figsize=(8,6))
    label_mean_tbr = 'Mean = '+str(np.round(np.nanmean(velocity_swarm),1))+' $\mu$m/min'
    n = ax.hist(velocity_swarm,bins=bins_vel,label=label_mean_tbr, density=True, alpha=0.4, histtype='bar', ec='black',color="royalblue")
    ax.set_title("Swarming",fontsize=fontsize*1.5)
    ax.set_xlabel("Velocity ($\mu$m/min)",fontsize=fontsize)
    ax.set_ylabel("Density",fontsize=fontsize)
    ax.set_xlim(0,max_vel)
    ax.tick_params(labelsize=fontsize)
    ax.legend(loc='best',fontsize=fontsize/1.5)
# %%
# Number of cells
cond_rippling = (df.loc[:,x_column] < x_medium * 0.6).values
cond_swarming = (df.loc[:,x_column] > x_medium * 1.4).values
nb_frames = len(np.unique(df.loc[:,t_column]))
print("nb_frames: ", nb_frames)
print("mean_nb_bact_rippling: ", len(df.loc[cond_rippling,x_column]) / nb_frames)
print("mean_nb_bact_swarming: ", len(df.loc[cond_swarming,x_column]) / nb_frames)
