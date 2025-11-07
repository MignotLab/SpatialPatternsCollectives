#%% import
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from skimage.segmentation import find_boundaries
from time import time as time_measure
from time import sleep as sleep
import pandas as pd
import PIL
import os

# plot all the zones on all the bacteria

def plot_fluo_analysis_fct(label_img_seg, label_img_fluo, regions_fluo, mean_noise, sd_noise, width, pcf, df, path_save_image_dir, time, reversals, plot_indices, show_plot, compression, compression_quality, mark_selection, iteration = None):

    if show_plot == "off":
        plt.close("all")
    
    a = time_measure()
    #--------------------CONVERT CONSTANTS------------------------
    width_um = width
    width_px = np.ceil(width / pcf).astype(int)
    radius_px = int(width_px/2)

    #--------------------MARKER SIZE SETTINGS------------------------
    # we will set the size of the image in inches first. Matplotlib will use a marker size of 1 / 72 inches as characteristic unit.
    size_inches = 100 # this sets the size of the image in inches. Inches are an important measure for fontsize etc. If it is too low, the minimum font/line size is reached. (e.g. at size_inches = 1 for an image of 3200x3200)
    # get the total height of the image
    image_width_px = np.shape(label_img_seg)[1]
    # We  want to set the font with a fixed amount of pixel size.
    # All the recommenadtions for sizes in pixels were tested on a 3200x3200 image. 
    # If the font gets too small for an image with higher resolution, scale it up with this factor
    font_scale_factor = 1
    # So if the size of the image changes, or the size in inches, we need to convert the font size to keep a fixed amount of pixel
    conversion_factor_px = font_scale_factor * 72 * size_inches / image_width_px
    # now, we can set the size of everything in pixel, by writing: conversion_factor * pixel_size

    # the size of the markers on the bacteria will be given in mycrometers.
    conversion_factor_um = font_scale_factor * 72 * size_inches / (image_width_px * pcf)
    # now we can set the size of everything in mu m, by writing conversion_factor_um * mum_size

    #--------------------INITIALIZE LISTS------------------------
    square_extracts_on_fluosize_full = np.zeros(np.shape(label_img_fluo))
    x_min_list = np.array([]) #list for min coords where two poles exist
    x_max_list = np.array([])
    y_min_list = np.array([])
    y_max_list = np.array([])


    #--------------------SELECT 2 POLED BACTERIA INDICES------------------------
    # find those indices of our selection where TWO poles COULD be calculated. Why? Only for these we can plot the calculation boxes.
    cond_selection = np.isnan(df.loc[:,"mean_intensity_pole_1"].values.astype(float)) == False
    
    # these indices of the table are now either the track ids or the 
    # seg ids, depending on how the table is sorted.

    # if we are in the reversal analysis, the table is sorted by track ids
    # this is because some seg ids are deleted because they have "nan" track ids
    # thus we have to select the correct table entries by giving the track indices
    # in the other case, the fluorescence analysis, we just give the seg indices
    if reversals == "yes":
        track_index_selection_two_poles = df.loc[cond_selection, "track_id"]
        seg_index_selection_two_poles = df.loc[cond_selection, "seg_id"]
        bact_index_selection_two_poles = track_index_selection_two_poles.values
    else:
        seg_index_selection_two_poles = df.loc[cond_selection, "seg_id"]
        bact_index_selection_two_poles = seg_index_selection_two_poles.values
    b = time_measure()
    #print("ini")
    #print(b-a)

    
    c = time_measure()
    # -----------------MARK CALCULATION SQUARES---------------
    # -----------------AND EXTRACT MIN AND MAX COORDS FOR EACH BACTERIA---------------
    # look if this selection is NULL - maybe we selected only boundary or wrong-pole cells

    if np.size(bact_index_selection_two_poles) == 0:
        print("A plot can not be created, since all selected cells are excluded, i.e. at the boundary or have a pole count != 2, ...")
    else:
        # we loop over all indices where two poles could be calculated. 
        for bact_index in bact_index_selection_two_poles:

            # we select the indices of the table that correspond to the desired indices,
            # so either to the track id or to the seg id.
            # Because it could be that the table does not include a certain track id
            # or seg id, thus the table is not completely sorted after those.
            # Sometimes, if a track id is missing, the table index is 53 while the track id is 54.
            # So to not work anymore with this misleading index of the table, we work now
            # only with the desired indices.
            # We can call the table at the desired index via cond_bact_index
            # Whenever we specifically need the indices, we select seg_index 
            if reversals == "yes":
                cond_bact_index = df.loc[:,"track_id"] == bact_index
                seg_index = df.loc[cond_bact_index, "seg_id"].values[0]   # for the regionprops, we want to make sure that the seg id is selected
            else: 
                cond_bact_index = df.loc[:,"seg_id"] == bact_index
                seg_index = df.loc[cond_bact_index, "seg_id"].values[0]   # for the regionprops, we want to make sure that the seg id is selected

            # extract image parameters
            fluo = regions_fluo[seg_index].image_intensity
            square_extracts_on_fluosize = np.zeros((np.shape(fluo)))
            
            # extract pole positions
            pole_1_xpos, pole_1_ypos, pole_2_xpos, pole_2_ypos = map(int,df.loc[cond_bact_index,"pole_1_x_local":"pole_2_y_local"].values[0])
            
            # we draw all the calculation squares into a matrix:
            # first find the cut-off size of the calculation squares
            square_extracts_on_fluosize[np.maximum(pole_1_ypos - radius_px, 0) : pole_1_ypos + radius_px + 1, np.maximum(pole_1_xpos - radius_px, 0) : pole_1_xpos + radius_px + 1] = 1 #mask pole 1 has same size parameters as mask in intensity calculation
            square_extracts_on_fluosize[np.maximum(pole_2_ypos - radius_px, 0) : pole_2_ypos + radius_px + 1, np.maximum(pole_2_xpos - radius_px, 0) : pole_2_xpos + radius_px + 1] = 1 #mask pole 2
            # then find the minimal and max. coordinates
            x_min = df.loc[cond_bact_index, "x_min_global"].values[0] #extract min and max coordinates of x and y. Min coordinates are helpful to do translation from local to global COS for example
            x_max = df.loc[cond_bact_index, "x_max_global"].values[0]
            y_min = df.loc[cond_bact_index, "y_min_global"].values[0]
            y_max = df.loc[cond_bact_index, "y_max_global"].values[0]
            # last, we draw them on the right positions
            cond_small_squares = square_extracts_on_fluosize == 1
            square_extracts_on_fluosize_full[y_min:y_max+1,x_min:x_max+1][cond_small_squares] += 1 #we insert our mask with the size of one bacteria into the big picture containing all bacteria. But we only take the part of the mask where it is one, to prevent overwriting something with the zeros of the mask
                
            # we memorize all the mininmal and maximal coordinates
            # to later translate from local to global coordinate system
            x_min_list = np.append(x_min_list, x_min)
            x_max_list = np.append(x_max_list, x_max)
            y_min_list = np.append(y_min_list, y_min)
            y_max_list = np.append(y_max_list, y_max)
            
            #finish loop and go to next bacteria

        # polish the square extract's list:
        # since some masks added up to values of two, we set them now to two to get a clearer picture
        square_extracts_on_fluosize_full[square_extracts_on_fluosize_full >1] = 1 

        # now also extract the boundary of each bacteria
        contours = find_boundaries(label_img_seg, mode = "inner")
        d = time_measure()
       # print("calc squares")
      #  print(d-c)
        e = time_measure()

        # -----------------FIND FLUO PLOT RANGE---------------
        # find a typical maximum of a bacteria to make the plot more visible. 

        # We take 10 bacteria and average the max.
        indices_selection = np.random.choice(bact_index_selection_two_poles, 100)
        local_max_list = np.array([])
        for j in indices_selection:

            if reversals == "yes":
                cond_bact_index = df.loc[:,"track_id"] == j
                seg_index = df.loc[cond_bact_index, "seg_id"].values[0]   # for the regionprops, we want to make sure that the seg id is selected
            else: 
                cond_bact_index = df.loc[:,"seg_id"] == j
                seg_index = df.loc[cond_bact_index, "seg_id"].values[0]   # for the regionprops, we want to make sure that the seg id is selected

            local_max = np.max(regions_fluo[seg_index].image_intensity)
            local_max_list = np.append(local_max_list, local_max)
        mean_max = np.mean(local_max_list)
        f = time_measure()
     #   print("fluo range setup")
     #   print(f-e)
        g = time_measure()

        # -----------------PLOT BACTERIA AND CALCULATION SQUARES---------------
        # plot settings         
        sizes_px = np.shape(label_img_seg)   
        fig = plt.figure()  
        fig.set_size_inches(size_inches , size_inches * sizes_px[0] / sizes_px[1], forward = False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        # do the plot as an overlap over the segmented image, the fluo image and the square extracts    
        # segments and contours
        ax.imshow(2*(label_img_seg >0).astype(int) - contours.astype(int), cmap = plt.cm.gist_gray, extent = (0,np.shape(label_img_fluo)[1],np.shape(label_img_fluo)[0],0)) #black and white image. Background black (noise not important there but bacteria white to make fluo visible) Contours are grey in between
        #ax.imshow(label_img_fluo, vmin = 100, vmax = 500,  alpha = 0.7, extent = (0,np.shape(label_img_fluo)[1],np.shape(label_img_fluo)[0],0)) #extend = left, right, bottom, up
        # fluo
        ax.imshow(label_img_fluo, vmin = mean_noise - 8*sd_noise, vmax = mean_max,  alpha = 0.7, extent = (0,np.shape(label_img_fluo)[1],np.shape(label_img_fluo)[0],0)) #extend = left, right, bottom, up
        # summation squares
        ax.imshow(square_extracts_on_fluosize_full, alpha = 0.2, extent = (0,np.shape(label_img_fluo)[1],np.shape(label_img_fluo)[0],0))
        h = time_measure()
      #  print("plot bact and calc squares")
      #  print(h-g)
        k = time_measure()

        # -----------------PLOT LEADING POLES-----------------------
        # select the leading pole coordinates and plot them in the right order.
        # What is the right order? The order according to indices in the dataframe
        # Why is it important? The lists of the global coordinates,x_min_list_leading_pole
        # were created according to this old order.
        start2 = time_measure()
        # conditions for selection: current time and all cells with leading pole 1 or 2
        cond_t = df.loc[:,"t"] == time
        lead_pole_1_cond = df.loc[:,"leading_pole"] == 1
        lead_pole_2_cond = df.loc[:,"leading_pole"] == 2

        # we extract the pole coordinates from those cells and save it as a dataframe
        coords_lead_pole_1_global = df.loc[cond_t & lead_pole_1_cond,["pole_1_x_local","pole_1_y_local"]] + df.loc[cond_t & lead_pole_1_cond,["x_min_global","y_min_global"]].values
        coords_lead_pole_2_global = df.loc[cond_t & lead_pole_2_cond,["pole_2_x_local","pole_2_y_local"]] + df.loc[cond_t & lead_pole_2_cond,["x_min_global","y_min_global"]].values

        # these dataframes are merged together in a form "pole_1_x","pole_1_y","pole_2_x","pole_2_y"
        coords_lead_pole_merged = pd.concat([coords_lead_pole_1_global, coords_lead_pole_2_global], ignore_index = False, sort = False) 

        # sort by indices to find the original order again
        coords_lead_pole_merged = coords_lead_pole_merged.sort_index()

        # save this, now that it is in the right order, as a numpy array and drop the nans
        coords_lead_pole_merged = coords_lead_pole_merged.values.astype(float)
        cond_nan = np.isnan(coords_lead_pole_merged) == False

        # the numpy array has to be reshaped  and the nans dropped
        coords_lead_pole_merged = np.reshape( coords_lead_pole_merged[cond_nan] , ( int( len(coords_lead_pole_merged[cond_nan]) / 2 ), 2 ) )
        # scatter plot of the global coords
        plt.scatter(coords_lead_pole_merged[:,0], coords_lead_pole_merged[:,1], s = ((radius_px * 2 + 1) * conversion_factor_px)**2, facecolors='none', edgecolors='orange', linewidths = 1 * conversion_factor_px)
        end2 = time_measure()
      #  print("plot leading poles:")
        time2 = end2-start2
     #   print(time2)


    
        # -----------------PLOT SEGMENTATION OR TRACKING NUMBERS-----------------------
        # additionally draw the numbers on the plot. If there is an error, print in red
        # remark that the one poled or boundary excluded are NOT plotted here
        if plot_indices == "on":
            
            m = time_measure()
            i = 0

            for bact_index in bact_index_selection_two_poles:
                
                if reversals == "yes":
                    cond_bact_index = df.loc[:,"track_id"] == bact_index
                    seg_index = df.loc[cond_bact_index, "seg_id"].values[0]   # for the regionprops, we want to make sure that the seg id is selected
                else: 
                    cond_bact_index = df.loc[:,"seg_id"] == bact_index
                    seg_index = df.loc[cond_bact_index, "seg_id"].values[0]   # for the regionprops, we want to make sure that the seg id is selected

                # find local pole coordinates
                center_xpos, center_ypos = map(int,df.loc[cond_bact_index,"center_x_local":"center_y_local"].values[0]) #extract pole positions

                # no error --> text in white
                if df.loc[cond_bact_index,"error_message_pole_1"].values[0] == "no error" and df.loc[cond_bact_index,"error_message_pole_2"].values[0] == "no error":
                    plt.text( x_min_list[i] + center_xpos , y_min_list[i] + center_ypos , str(bact_index), color = "white", fontsize = np.maximum(8 * conversion_factor_px , 0.6 * width_um * conversion_factor_um) ) #font size is either 8 pixels or half of the bacteria's width
                # error --> text in red
                else: 
                    plt.text( x_min_list[i] + center_xpos , y_min_list[i] + center_ypos , "x" + str(bact_index) + "x", color = "lightpink", fontsize = 8 * conversion_factor_px )

                i = i+1
            n = time_measure()
       #     print("plot numbers")
        #    print(n-m)
        ax.axis("off")

        # -----------------PLOT REVERSALS---------------------------
        if reversals == "yes":

            cond_reversals = df.loc[:, "reversal"] == 1
            reversal_indices = df.loc[cond_reversals, "track_id"].values 
            for rev_index in reversal_indices:
                
                cond_rev_index = df.loc[:,"track_id"] == rev_index
                seg_index = df.loc[cond_rev_index, "seg_id"].values[0]   # for the regionprops, we want to make sure that the seg id is selected

                # find local pole coordinates
                center_xpos, center_ypos = map(int,df.loc[cond_rev_index,"center_x_local":"center_y_local"].values[0]) #extract pole positions
                
                # find translation value to global pole coordinates
                x_min_new = df.loc[cond_rev_index,"x_min_global"] #extract min and max coordinates of x and y
                y_min_new = df.loc[cond_rev_index,"y_min_global"]

                # add reversal patch
                ax.add_patch(Circle((x_min_new + center_xpos, y_min_new + center_ypos), radius = 0.2 * width * conversion_factor_um, color = "red", alpha = 0.8))
                ax.add_patch(Circle((x_min_new + center_xpos, y_min_new + center_ypos), radius = 3 * width * conversion_factor_um, color = "red", alpha = 0.3))


        o = time_measure()

        # -----------------MARK SELECTION---------------------------
        if "yes" in mark_selection:

            cond_marking = df.loc[:, "selection"] == True
            marking_indices = df.loc[cond_marking, "seg_id"].values 
            for marking_index in marking_indices:
                
                cond_marking_index = df.loc[:,"seg_id"] == marking_index
                seg_index = df.loc[cond_marking_index, "seg_id"].values[0]   # for the regionprops, we want to make sure that the seg id is selected

                # find local pole coordinates
                center_xpos, center_ypos = map(int,df.loc[cond_marking_index,"center_x_local":"center_y_local"].values[0]) #extract pole positions
                
                # find translation value to global pole coordinates
                x_min_new = df.loc[cond_marking_index,"x_min_global"] #extract min and max coordinates of x and y
                y_min_new = df.loc[cond_marking_index,"y_min_global"]

                # add reversal patch
                ax.add_patch(Circle((x_min_new + center_xpos, y_min_new + center_ypos), radius = 0.2 * width * conversion_factor_um, color = "red", alpha = 0.8))
                ax.add_patch(Circle((x_min_new + center_xpos, y_min_new + center_ypos), radius = 3 * width * conversion_factor_um, color = "red", alpha = 0.3))


        # -----------------SAVE FIGURE---------------------------
        if reversals == "yes":
            save_name = path_save_image_dir + "reversal_analysis/" + str(time) + ".tif"
        elif mark_selection == "yes":
            if iteration is not None:
                save_name = path_save_image_dir + "visualize_selection/" + str(time) + "_iter=" + str(iteration) + ".tif"
            else:
                save_name = path_save_image_dir + "visualize_selection/" + str(time) + ".tif"
        # case that a change in leading pole of the selection: old algo without parameter refinement and depollution vs new algo with everything
        elif mark_selection == "yes_before":
            save_name = path_save_image_dir + "algorithm_comparison/old_" + str(time) + ".tif"
        elif mark_selection == "yes_after":
            save_name = path_save_image_dir + "algorithm_comparison/new_" + str(time) + ".tif"
        else:
            save_name = path_save_image_dir + "fluo_analysis/" + str(time) + ".tif"
            
        fig.savefig(save_name, dpi = sizes_px[1] / size_inches)
        
        
        # -----------------COMPRESS FIGURE---------------------------
        if compression == "on":
            im = PIL.Image.open(save_name) #open
            rgb_im = im.convert('RGB') # convert to RGB (drop alpha channel)
            rgb_im.save(save_name.rstrip(".tif") + ".jpg" , "JPEG", quality= compression_quality) #compress
            im.close()
            rgb_im.close()
            os.remove(save_name) #remove normal file
        p = time_measure()

        if show_plot == "off":
            plt.close("all")
        else: 
            plt.show()
     #   print("save figure")
     #   print(p-o)

    
    
        # # OLD METHOD TO PLOT THE LEADING POLES VIA PATCHES. IS MUCH SLOWER (10-100x)
        # start1 = time_measure()
        # # loop again over all bacteria with 2 poles
        # for bact_index in bact_index_selection_two_poles:

        #     if reversals == "yes":
        #         cond_bact_index = df.loc[:,"track_id"] == bact_index
        #         seg_index = df.loc[cond_bact_index, "seg_id"].values[0]   # for the regionprops, we want to make sure that the seg id is selected
        #     else: 
        #         cond_bact_index = df.loc[:,"seg_id"] == bact_index
        #         seg_index = df.loc[cond_bact_index, "seg_id"].values[0]   # for the regionprops, we want to make sure that the seg id is selected


        #     # find local pole coordinates
        #     pole_1_xpos, pole_1_ypos, pole_2_xpos, pole_2_ypos = map(int,df.loc[cond_bact_index,"pole_1_x":"pole_2_y"].values[0]) #extract pole positions


        #     # find translation value to global pole coordinates
        #     x_min_new = np.min(regions_fluo[seg_index].coords[:,1]) #extract min and max coordinates of x and y
        #     y_min_new = np.min(regions_fluo[seg_index].coords[:,0]) #extract min and max coordinates of x and y

        #     # plot leading poles by adding a circle
        #     if df.loc[cond_bact_index,"leading_pole"].values[0] == 1:
        #         ax.add_patch(Circle((x_min_new + pole_1_xpos, y_min_new + pole_1_ypos), radius = 0.5, color = "green"))
        #     elif df.loc[cond_bact_index,"leading_pole"].values[0] == 2:
        #         ax.add_patch(Circle((x_min_new + pole_2_xpos, y_min_new + pole_2_ypos), radius = 0.5, color = "green"))
        # end1 = time_measure()
        # l = time_measure()
        # print("plot leading poles")
        # print(l-k)
        





    # useful plotting code
    # vmin = 100
    # vmax = 500
    # a = label_img_fluo / vmax
    # a[a>1] = 1
    # b = label_img_seg > 0
    # plt.figure(figsize = (32,32), dpi = 100)
    # plt.imshow(a*30+b)
# %%