#%% import
import numpy as np
import matplotlib.pyplot as plt
import error_message
import skeleton_analysis

# ------------------------------------------------------------------
# DETECTS POLES
# ------------------------------------------------------------------

def pole_identification_fct(bact_index, fluo, skel, segment_contour, fluo_max_coords, min_skel_length, df, plot_onoff):

    #-------------------DETECT POLES, CENTER, PATH COUNT---------------------------
    pole_1_coords, pole_2_coords, center_coords, n_paths, skel_length, pole_1_coords_pollution, pole_2_coords_pollution = skeleton_analysis.skan_pole_detection_fct(skel, segment_contour, bact_index, df)
    
    # write coordinates into the dataframe if there is only 1 path
    if n_paths == 1:
        # write local coordinates
        df.loc[bact_index,"pole_1_x_local"] = pole_1_coords[1]
        df.loc[bact_index,"pole_1_y_local"] = pole_1_coords[0]
        df.loc[bact_index,"pole_2_x_local"] = pole_2_coords[1]
        df.loc[bact_index,"pole_2_y_local"] = pole_2_coords[0]
        df.loc[bact_index,"center_x_local"] = center_coords[1]
        df.loc[bact_index,"center_y_local"] = center_coords[0]
        df.loc[bact_index,"pole_1_x_local_pollution"] = pole_1_coords_pollution[1]
        df.loc[bact_index,"pole_1_y_local_pollution"] = pole_1_coords_pollution[0]
        df.loc[bact_index,"pole_2_x_local_pollution"] = pole_2_coords_pollution[1]
        df.loc[bact_index,"pole_2_y_local_pollution"] = pole_2_coords_pollution[0]
        # write local coordinates of the fluorescent cluster
        df.loc[bact_index,"fluo_max_x_local"] = fluo_max_coords[1]
        df.loc[bact_index,"fluo_max_y_local"] = fluo_max_coords[0]

    #-------------------PLOT POLES------------------
    if plot_onoff == "on":
        # show the poles + end points
        fig = plt.figure()
        ax = plt.axes()
        plt.title("skeleton and poles")

        # plot sceleton in black, everything around grey
        plt.imshow(skel,  cmap = plt.cm.Greys)

        # prepare colormap and plot variable s.t values outside the bacteria are white
        cmap_fluo = plt.cm.viridis  # Can be any colormap that you want after the cm
        cmap_fluo.set_bad(color='white')
        fluo_plot = np.ma.masked_where(fluo == 0, fluo)

        # plot fluo as an overlay on the skeleton
        im = plt.imshow(fluo_plot, alpha = 0.8, cmap = cmap_fluo)

        # plot poles only if there are 2
        if n_paths == 1:
            plt.scatter( [ pole_1_coords[1], pole_2_coords[1], center_coords[1] ] , [ pole_1_coords[0], pole_2_coords[0], center_coords[0] ], c = "orange" )
            #plt.scatter(pole_2_coords[1],pole_2_coords[0])
            #plt.scatter(center_coords[1],center_coords[0])

        # set colorbar to be the same height as the plot
        cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
        plt.colorbar(im, cax=cax) # Similar to fig.colorbar(im, cax = cax)

        
            
    #-------------------REMARK WRONG POLE COUNTS------------------
    # if we have more than two poles = more than one path, we have multiple bacteria in one segment. This is a segmentation mistake, so they are remarked and excluded.
    if n_paths == 0:
        only_two_poles = "0 or 1"
        df.loc[bact_index,"mean_intensity_pole_1"] = np.nan
        df.loc[bact_index,"mean_intensity_pole_2"] = np.nan
        error_message.create_error_msg_fct(df, bact_index, "0 or 1 poles", affected_poles= [1,2])

    elif n_paths > 1:
        only_two_poles = "3 or more"
        df.loc[bact_index,"mean_intensity_pole_1"] = np.nan
        df.loc[bact_index,"mean_intensity_pole_2"] = np.nan
        error_message.create_error_msg_fct(df, bact_index, "3 or more poles", affected_poles= [1,2])
    
    else:
        only_two_poles = "yes"

    #-------------------REMARK LOW SKELETON LENGTH------------------
    if skel_length < min_skel_length:
        sufficient_skel_length = False
        df.loc[bact_index,"mean_intensity_pole_1"] = np.nan
        df.loc[bact_index,"mean_intensity_pole_2"] = np.nan
        error_message.create_error_msg_fct(df, bact_index, "skel too small", affected_poles= [1,2])

    else:
        sufficient_skel_length = True

    
    return(only_two_poles, sufficient_skel_length)