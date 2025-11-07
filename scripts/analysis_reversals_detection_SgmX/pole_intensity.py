import numpy as np
import matplotlib.pyplot as plt
import error_message
import settings

# ------------------------------------------------------------------
# CALCULATES FLUORESCENCE INTENSITY AROUND EACH POLE
# ------------------------------------------------------------------

def pole_intensity_fct(fluo, fluo_thresh, width, pcf, df, bact_index, plot_onoff):

    #----------------------EXTRACT POLE POSITIONS-----------------------------
    # get the just calculated pole positions from the dataframe
    pole_1_xpos = df.loc[bact_index,"pole_1_x_local"]
    pole_1_ypos = df.loc[bact_index,"pole_1_y_local"]
    pole_2_xpos = df.loc[bact_index,"pole_2_x_local"]
    pole_2_ypos = df.loc[bact_index,"pole_2_y_local"]

    
    #----------------------DETERMINE SUMMATION AREA-----------------------------
    # we do our summation over a square with the width of the bacteria size (if this square is not shrinkened by the frame size)
    width_px = np.ceil(width / pcf).astype(int) #convert width from um into pixel
    radius = int(width_px/2) #define a radius of summation

    square_extract_pole_1 = fluo[np.maximum(pole_1_ypos - radius, 0) : pole_1_ypos + radius + 1, np.maximum(pole_1_xpos - radius, 0) : pole_1_xpos + radius + 1]
    square_extract_pole_2 = fluo[np.maximum(pole_2_ypos - radius, 0) : pole_2_ypos + radius + 1, np.maximum(pole_2_xpos - radius, 0) : pole_2_xpos + radius + 1]

    pole_1_fluo_sum_area = square_extract_pole_1
    pole_2_fluo_sum_area = square_extract_pole_2


    #----------------------SUM OVER THIS AREA AREA-----------------------------
    #build the mean over all fluorizing elements in this area:
    #fluorizing elements are those who are above a certain threshold. 

    #pole 1
    pole_1_sum = 0
    pole_1_n = 0
    for i in range(np.shape(pole_1_fluo_sum_area)[0]):
        for j in range(np.shape(pole_1_fluo_sum_area)[1]):
            if pole_1_fluo_sum_area[i,j] != 0 and pole_1_fluo_sum_area[i,j] > fluo_thresh: # the fluo matrix is always zero if the segment is zero as well. So pixels outside the bacteria are marked with 0. So in the fluo view we can only find the fluorescence on the bacteria, but not next to it.
                pole_1_sum = pole_1_sum + pole_1_fluo_sum_area[i,j]
                pole_1_n = pole_1_n + 1 #count the number of cells that are nonzero for the average
    df.loc[bact_index, "pole_1_n"] = pole_1_n
    if pole_1_n != 0:
        df.loc[bact_index,"mean_intensity_pole_1"] = pole_1_sum / pole_1_n  
    else:
       # print("\nError: Fluo threshold not hit anywhere --> summation over zero pixels\n")
        df.loc[bact_index,"mean_intensity_pole_1"] = np.nan
        error_message.create_error_msg_fct(df, bact_index, "fluo thresh", affected_poles= [1])

    #pole 2
    pole_2_sum = 0
    pole_2_n = 0
    for i in range(np.shape(pole_2_fluo_sum_area)[0]):
        for j in range(np.shape(pole_2_fluo_sum_area)[1]):
            if pole_2_fluo_sum_area[i,j] != 0 and pole_2_fluo_sum_area[i,j] > fluo_thresh: 
                pole_2_sum = pole_2_sum + pole_2_fluo_sum_area[i,j]
                pole_2_n = pole_2_n + 1 #count the number of cells that are nonzero for the average
    df.loc[bact_index, "pole_2_n"] = pole_2_n
    if pole_2_n != 0:
        df.loc[bact_index,"mean_intensity_pole_2"] = pole_2_sum / pole_2_n   
    else:
      #  print("Error: Fluo threshold not hit anywhere --> summation over zero pixels")
        df.loc[bact_index,"mean_intensity_pole_2"] = np.nan
        error_message.create_error_msg_fct(df, bact_index, "fluo thresh", affected_poles= [2])


    #----------------------PLOT AREA AND SUM FOR A BACTERIA-----------------------------
    #plot poles and calculation domains if needed
    if plot_onoff == "on":
        #show the selected areas and calculated means
        fig = plt.figure()

        # prepare colormap and plot variable s.t values outside the bacteria are white
        cmap_fluo = plt.cm.viridis  # Can be any colormap that you want after the cm
        cmap_fluo.set_bad(color='white')
        square_extract_pole_1_plot = np.ma.masked_where(square_extract_pole_1 == 0, square_extract_pole_1)
        square_extract_pole_2_plot = np.ma.masked_where(square_extract_pole_2 == 0, square_extract_pole_2)
        fluo_plot = np.ma.masked_where(fluo == 0, fluo)

        # plot pole extracts
        plt.title("Mean of pole 1: " + str(round(df.loc[bact_index,"mean_intensity_pole_1"],2)) + " Fluo threshold = " + str(fluo_thresh))
        plt.imshow(square_extract_pole_1_plot, alpha = 0.7, cmap = cmap_fluo)
        plt.clim(np.min(fluo),np.max(fluo)) #set the colorbar range to the one of the fluo image
        plt.colorbar()
        fig2 = plt.figure()
        plt.title("Mean of pole 2: " + str(round(df.loc[bact_index,"mean_intensity_pole_2"],2)) + " Fluo threshold = " + str(fluo_thresh))
        plt.imshow(square_extract_pole_2_plot, alpha = 0.7, cmap = cmap_fluo)
        plt.clim(np.min(fluo),np.max(fluo))
        plt.colorbar()

        # prepare entire bacteria analysis
        square_extracts_on_fluosize = np.zeros((np.shape(fluo)))
        square_extracts_on_fluosize[np.maximum(pole_1_ypos - radius, 0) : pole_1_ypos + radius + 1, np.maximum(pole_1_xpos - radius, 0) : pole_1_xpos + radius + 1] = 1 #mask pole 1
        square_extracts_on_fluosize[np.maximum(pole_2_ypos - radius, 0) : pole_2_ypos + radius + 1, np.maximum(pole_2_xpos - radius, 0) : pole_2_xpos + radius + 1] = 1 #mask pole 2

        # plot entire bacteria analysis
        fig3 = plt.figure()
        plt.imshow(fluo_plot, cmap = cmap_fluo)
        plt.title("Bacteria number: " + str(bact_index) + ", Mean of pole 1: " + str(round(df.loc[bact_index,"mean_intensity_pole_1"],2)) + ", Mean of pole 2: " + str(round(df.loc[bact_index,"mean_intensity_pole_2"],2)))
        plt.imshow(square_extracts_on_fluosize, alpha = 0.2)
        plt.show()

    return(pole_1_n, pole_2_n)

