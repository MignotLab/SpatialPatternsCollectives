#%% import
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from skimage.io import imread

# plot all the zones on all the bacteria

def plot_everything_without_main(path_seg, path_fluo, df):
    
    # load the data that is usually provided from the other functions 
    label_img_seg = imread(path_seg)
    label_img_fluo = imread(path_fluo)
    regions_fluo = regionprops(label_image=label_img_seg, intensity_image=label_img_fluo)

    # first of all we have to recalculate the bacteria radius
    # 1 pixel = 0.06 mu m
    width = 0.7 #of a bacteria in mu m
    pcf = 0.06 #mu m - pixel conversion factor
    width_px = np.ceil(width / pcf).astype(int)
    radius = int(width_px/2)
    
    # initialize lists
    square_extracts_on_fluosize_full = np.zeros(np.shape(label_img_fluo))
    x_min_list = np.array([])
    x_max_list = np.array([])
    y_min_list = np.array([])
    y_max_list = np.array([])
    text_index_list = np.array([])

    # find those indices where two poles could be calculated. ATTENTION: the output is weird without the [0] at the end
    bact_index_selection_two_poles = np.where(np.isnan(df.loc[:,"pole_1_x"].values.astype(float)) == False)[0]

    # we loop over all indices where two poles could be calculated. We can just look where pole 1 x has a non-nan entry
    for bact_index in bact_index_selection_two_poles:
        # extract image parameters
        fluo = regions_fluo[bact_index].image_intensity
        square_extracts_on_fluosize = np.zeros((np.shape(fluo)))
        
        # extract pole positions
        pole_1_xpos, pole_1_ypos, pole_2_xpos, pole_2_ypos = map(int,df.loc[bact_index,"pole_1_x":"pole_2_y"].values)
        
        # we draw all the calculation squares into a matrix
        square_extracts_on_fluosize[np.maximum(pole_1_ypos - radius, 0) : pole_1_ypos + radius + 1, np.maximum(pole_1_xpos - radius, 0) : pole_1_xpos + radius + 1] = 1 #mask pole 1
        square_extracts_on_fluosize[np.maximum(pole_2_ypos - radius, 0) : pole_2_ypos + radius + 1, np.maximum(pole_2_xpos - radius, 0) : pole_2_xpos + radius + 1] = 1 #mask pole 2
        x_min = np.min(regions_fluo[bact_index].coords[:,1])
        x_max = np.max(regions_fluo[bact_index].coords[:,1])
        y_min = np.min(regions_fluo[bact_index].coords[:,0])
        y_max = np.max(regions_fluo[bact_index].coords[:,0])
        square_extracts_on_fluosize_full[y_min:y_max+1,x_min:x_max+1] = square_extracts_on_fluosize
        
        # we memorize all the mininmal and maximal coordinates to use them later to draw numbers on the bacteria
        x_min_list = np.append(x_min_list, x_min)
        x_max_list = np.append(x_max_list, x_max)
        y_min_list = np.append(y_min_list, y_min)
        y_max_list = np.append(y_max_list, y_max)
        text_index_list = np.append(text_index_list, bact_index)


    # # print(bact_index_selection)
    # for bact_index in bact_index_selection:
    #     if df.loc[bact_index, "error_message_pole_1"] == "no error" and df.loc[bact_index, "error_message_pole_2"] == "no error":
    #         # print(bact_index)
    #         fluo = regions_fluo[bact_index].image_intensity
    #         square_extracts_on_fluosize = np.zeros((np.shape(fluo)))
    #         # print(df.loc[bact_index,:])
    #         # print("whole selection")
    #         # print(df.loc[bact_index,"pole_1_x":"pole_2_y"].values)
    #         # print("array selection")
    #         # print(df.loc[bact_index,["pole_1_x","pole_1_y","pole_2_x","pole_2_y"]].values[0])
    #         pole_1_xpos, pole_1_ypos, pole_2_xpos, pole_2_ypos = map(int,df.loc[bact_index,"pole_1_x":"pole_2_y"].values)
    #         if np.isnan(pole_1_xpos) != True:
    #             square_extracts_on_fluosize[np.maximum(pole_1_ypos - radius, 0) : pole_1_ypos + radius + 1, np.maximum(pole_1_xpos - radius, 0) : pole_1_xpos + radius + 1] = 1 #mask pole 1
    #             square_extracts_on_fluosize[np.maximum(pole_2_ypos - radius, 0) : pole_2_ypos + radius + 1, np.maximum(pole_2_xpos - radius, 0) : pole_2_xpos + radius + 1] = 1 #mask pole 2
    #             x_min = np.min(regions_fluo[bact_index].coords[:,1])
    #             x_max = np.max(regions_fluo[bact_index].coords[:,1])
    #             y_min = np.min(regions_fluo[bact_index].coords[:,0])
    #             y_max = np.max(regions_fluo[bact_index].coords[:,0])
    #             square_extracts_on_fluosize_full[y_min:y_max+1,x_min:x_max+1] = square_extracts_on_fluosize
    #             x_min_list = np.append(x_min_list, x_min)
    #             x_max_list = np.append(x_max_list, x_max)
    #             y_min_list = np.append(y_min_list, y_min)
    #             y_max_list = np.append(y_max_list, y_max)
    #             text_index_list = np.append(text_index_list, bact_index)
    
    # do the plot as an overlap over the segmented image, the fluo image and the square extracts    
    plt.figure(figsize = (32,32), dpi = 100)
    plt.title("All bacteria with summation areas")
    plt.imshow(label_img_seg >0, cmap = plt.cm.gist_gray, extent = (0,np.shape(label_img_fluo)[1],np.shape(label_img_fluo)[0],0)) #extend = left, right, bottom, up
    plt.imshow(label_img_fluo, vmin = 100, vmax = 500,  alpha = 0.7, extent = (0,np.shape(label_img_fluo)[1],np.shape(label_img_fluo)[0],0))
    plt.imshow(square_extracts_on_fluosize_full, alpha = 0.4, extent = (0,np.shape(label_img_fluo)[1],np.shape(label_img_fluo)[0],0))
    
    # additionally draw the numbers on the plot
    i = 0
    for text_index in text_index_list.astype(int):
        plt.text( np.round((x_max_list[i] + x_min_list[i])/2) , np.round((y_max_list[i] + y_min_list[i])/2) , str(text_index), color = "white" )
        i = i+1
        #plt.text( text_index, text_index , text_index, color = "white" )

    plt.axis('off')

    # useful plotting code
    # vmin = 100
    # vmax = 500
    # a = label_img_fluo / vmax
    # a[a>1] = 1
    # b = label_img_seg > 0
    # plt.figure(figsize = (32,32), dpi = 100)
    # plt.imshow(a*30+b)
# %%
