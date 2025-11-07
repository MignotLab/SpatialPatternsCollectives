#%% import
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import skeletonize
from skimage.io import imread, imsave
from skimage.restoration import estimate_sigma
import matplotlib.pyplot as plt
from scipy.spatial import KDTree as kdtree

# %% function definition for main loop

# plot all the zones on all the bacteria

def plot_everything(pole_coord_list, label_img_seg, label_img_fluo, regions_fluo, df, bact_index_selection):
    # 1 pixel = 0.06 mu m
    width = 0.7 #of a bacteria in mu m
    pcf = 0.06 #mu m - pixel conversion factor
    width_px = np.ceil(width / pcf).astype(int)
    radius = int(width_px/2)
    
    square_extracts_on_fluosize_full = np.zeros(np.shape(label_img_fluo))
    x_min_list = np.array([])
    x_max_list = np.array([])
    y_min_list = np.array([])
    y_max_list = np.array([])
    text_index_list = np.array([])

    for bact_index in bact_index_selection:
        if df.loc[bact_index, "error_message_pole_1"] == "no error" and df.loc[bact_index, "error_message_pole_2"] == "no error":
            fluo = regions_fluo[bact_index].image_intensity
            square_extracts_on_fluosize = np.zeros((np.shape(fluo)))
            pole_1_xpos, pole_1_ypos, pole_2_xpos, pole_2_ypos = map(int,pole_coord_list[:,bact_index])
            if np.isnan(pole_1_xpos) != True:
                square_extracts_on_fluosize[np.maximum(pole_1_ypos - radius, 0) : pole_1_ypos + radius + 1, np.maximum(pole_1_xpos - radius, 0) : pole_1_xpos + radius + 1] = 1 #mask pole 1
                square_extracts_on_fluosize[np.maximum(pole_2_ypos - radius, 0) : pole_2_ypos + radius + 1, np.maximum(pole_2_xpos - radius, 0) : pole_2_xpos + radius + 1] = 1 #mask pole 2
                x_min = np.min(regions_fluo[bact_index].coords[:,1])
                x_max = np.max(regions_fluo[bact_index].coords[:,1])
                y_min = np.min(regions_fluo[bact_index].coords[:,0])
                y_max = np.max(regions_fluo[bact_index].coords[:,0])
                square_extracts_on_fluosize_full[y_min:y_max+1,x_min:x_max+1] = square_extracts_on_fluosize
                x_min_list = np.append(x_min_list, x_min)
                x_max_list = np.append(x_max_list, x_max)
                y_min_list = np.append(y_min_list, y_min)
                y_max_list = np.append(y_max_list, y_max)
                text_index_list = np.append(text_index_list, bact_index)
    
    
    plt.figure(figsize = (32,32), dpi = 100)
    plt.title("All bacteria with summation areas")
    plt.imshow(label_img_seg >0, cmap = plt.cm.gist_gray, extent = (0,np.shape(label_img_fluo)[1],np.shape(label_img_fluo)[0],0)) #extend = left, right, bottom, up
    plt.imshow(label_img_fluo, vmin = 100, vmax = 500,  alpha = 0.7, extent = (0,np.shape(label_img_fluo)[1],np.shape(label_img_fluo)[0],0))
    plt.imshow(square_extracts_on_fluosize_full, alpha = 0.4, extent = (0,np.shape(label_img_fluo)[1],np.shape(label_img_fluo)[0],0))
    
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

        

# algorithm to check if a bacteria lies at the boundary, i.e. has a pixel that is "bord_thresh"- pixels away from the image boundary
def boundary_check(bact_index, bord_thresh, regions_fluo, label_img_fluo):
    
    # find the bacterial borders
    x_min = np.min(regions_fluo[bact_index].coords[:,1])
    x_max = np.max(regions_fluo[bact_index].coords[:,1])
    y_min = np.min(regions_fluo[bact_index].coords[:,0])
    y_max = np.max(regions_fluo[bact_index].coords[:,0])

    # --> exclude bacterias at the boundary
    if x_min < 0 + bord_thresh or x_max >= np.shape(label_img_fluo)[1] - bord_thresh or y_min < 0 + bord_thresh or y_max >= np.shape(label_img_fluo)[0] - bord_thresh:
        at_boundary = "yes"

    else:
        at_boundary = "no"

    return(at_boundary)

def pole_identification(fluo, skel, plot_onoff):
    # find the poles = end nodes. They are the pixels with just one neighbour)
    # for this we have to sum over all the nodes of the image and see which ones have 1 neighbour only AND lie on the skeleton

    # initialize matrix with values of the skeleton over which we can sum
    skel_int = skel.astype(int) #convert the boolean skel to integers that can be added
    skel_big = np.zeros((fluo.shape[0]+2, fluo.shape[1]+2)) #create an artificial outer layer of zeros to perform the addition over all cells without problems (sometimes the end nodes lie at the boundary of skel)
    skel_big[1:-1,1:-1] = skel_int

    # make the sum over the neighbourhoods
    neighbour_sum = np.zeros((fluo.shape[0]+2,fluo.shape[1]+2))
    for i in range(1,skel_big.shape[0]-1): #zeilen
        for j in range(1,skel_big.shape[1]-1): #spalten
            neighbour_sum[i,j] = skel_big[i-1,j-1] + skel_big[i-1,j] + skel_big[i-1,j+1] + \
                skel_big[i,j-1] + skel_big[i,j+1] +  \
                skel_big[i+1,j-1] + skel_big[i+1,j] + skel_big[i+1,j+1] 

    # find pole positions as positions where skeleton cells only have one neighbour
    pole_matrix = np.logical_and(neighbour_sum == 1,skel_big == 1)
    pole_positions = np.where(np.logical_and(neighbour_sum[1:-1] == 1,skel_big[1:-1] == 1) == True) #we restrict our matrices again to the size of the original skeleton to obtain the correct pole coordinates

    if plot_onoff == "on":
        # show the poles + end points
        plt.figure()
        plt.title("skeleton and poles")
        plt.imshow(fluo, alpha = 0.95)
        plt.colorbar()
        plt.imshow(skel + 2* np.logical_and(neighbour_sum[1:-1,1:-1] == 1,skel == 1).astype(int), alpha = 0.45,  cmap = plt.cm.Greys)
        #plt.imshow(fluo + 1.001*np.max(fluo)*skel + 1.002*np.max(fluo) * np.logical_and(neighbour_sum[1:-1,1:-1] == 1,skel == 1).astype(int) )
    
    # if we have more than two poles, we have multiple bacteria in one segment. This is a segmentation mistake, so they are remarked and excluded.
    # can less then 2 segments happen? Not clear yet!
    #if np.sum(pole_matrix) != 2:
    #    only_two_poles = "no"
    if np.sum(pole_matrix) < 2:
        only_two_poles = "0 or 1"

    elif np.sum(pole_matrix) > 2:
        only_two_poles = "3 or more"

    else:
        only_two_poles = "yes"
    
    return(pole_positions, only_two_poles)

def pole_intensity(pole_positions, fluo, fluo_thresh_factor, df, plot_onoff, label_img_fluo, bact_index, sigma):
    # extract the pole positions
    pole_1_xpos = pole_positions[1][0]
    pole_1_ypos = pole_positions[0][0]
    pole_2_xpos = pole_positions[1][1]
    pole_2_ypos = pole_positions[0][1]
    
    # 1 pixel = 0.06 mu m
    width = 0.7 #of a bacteria in mu m
    pcf = 0.06 #mu m - pixel conversion factor
    width_px = np.ceil(width / pcf).astype(int)
    radius = int(width_px/2)

    # we want to take the fluo mean over a circle of radius = bacteria_width:
    # # firstly, create an even bigger matrix extending by the radius in all directions in case that the summation would be restricted by the boundary
    # fluo_bigger = np.zeros( (segment.shape[0] + 2 * radius, segment.shape[1] + 2 * radius) ) #create the big one
    # fluo_bigger[radius: segment.shape[0] +radius, radius: segment.shape[1] + radius] = fluo #fill the fluo values in

    # # #create a circle mask
    # X = np.arange(-radius,radius+1)
    # [X, Y] = np.meshgrid(X,X)
    # distance = np.sqrt(X**2 + Y**2)
    # circle_mask = distance <= radius

    square_extract_pole_1 = fluo[np.maximum(pole_1_ypos - radius, 0) : pole_1_ypos + radius + 1, np.maximum(pole_1_xpos - radius, 0) : pole_1_xpos + radius + 1]
    square_extract_pole_2 = fluo[np.maximum(pole_2_ypos - radius, 0) : pole_2_ypos + radius + 1, np.maximum(pole_2_xpos - radius, 0) : pole_2_xpos + radius + 1]

    
    # #define a rectangular extract where the loop is going over and the circle comes inside:
    # square_extract_pole_1 = fluo_bigger[ pole_1_ypos : pole_1_ypos + 2 * radius + 1 ,  pole_1_xpos : pole_1_xpos + 2 * radius + 1]
    # square_extract_pole_2 = fluo_bigger[ pole_2_ypos : pole_2_ypos + 2 * radius + 1,  pole_2_xpos : pole_2_xpos + 2 * radius + 1]

    # #create the regarded area. This is the rectangle extract with the circle inside. All the values outside the circle are set to zero.
    # pole_1_fluo_sum_area = circle_mask * square_extract_pole_1
    # pole_2_fluo_sum_area = circle_mask * square_extract_pole_2

    pole_1_fluo_sum_area = square_extract_pole_1
    pole_2_fluo_sum_area = square_extract_pole_2

    #build the mean over all fluorizing elements in this area:
    #fluorizing elements are those who are above a certain threshold. This excludes the non-zero elements outside the bacteria and the background noise
    #this worked quite well but could be further tested. The script for it is below
    noise = np.mean(label_img_fluo)
    fluo_thresh = fluo_thresh_factor * noise #no exclusion for fluo

    #pole 1
    pole_1_sum = 0
    pole_1_n = 0
    for i in range(np.shape(pole_1_fluo_sum_area)[0]):
        for j in range(np.shape(pole_1_fluo_sum_area)[1]):
            if pole_1_fluo_sum_area[i,j] > fluo_thresh: #average noise is 150, strong fluo is 800
                pole_1_sum = pole_1_sum + pole_1_fluo_sum_area[i,j]
                pole_1_n = pole_1_n + 1 #count the number of cells that are nonzero for the average
    if pole_1_n != 0:
        mean_intensity_pole_1 = pole_1_sum / pole_1_n  
        error_message_pole_1 = "no error"

    else:
        mean_intensity_pole_1 = "no error"
        error_message_pole_1 = "Fluo threshold not hit anywhere --> summation over zero pixels"


    #pole 2
    pole_2_sum = 0
    pole_2_n = 0
    for i in range(np.shape(pole_2_fluo_sum_area)[0]):
        for j in range(np.shape(pole_2_fluo_sum_area)[1]):
            if pole_2_fluo_sum_area[i,j] > fluo_thresh:
                pole_2_sum = pole_2_sum + pole_2_fluo_sum_area[i,j]
                pole_2_n = pole_2_n + 1 #count the number of cells that are nonzero for the average
    if pole_2_n != 0:
        mean_intensity_pole_2 = pole_2_sum / pole_2_n   
        error_message_pole_2 = "no error"

    else:
        mean_intensity_pole_2 = "no error"
        error_message_pole_2 =  "Fluo threshold not hit anywhere --> summation over zero pixels"

    #compare pole intensity to the noise: 
    if abs(mean_intensity_pole_2 - mean_intensity_pole_1) / np.maximum(mean_intensity_pole_1,mean_intensity_pole_2) <= 3*sigma / np.maximum(mean_intensity_pole_1,mean_intensity_pole_2):
        error_message_pole_1 = "Pole difference as big as noise"
        error_message_pole_2 = "Pole difference as big as noise"       
    else:  
        error_message_pole_1 = "no error"
        error_message_pole_2 = "no error"

    #plot poles and calculation domains
    if plot_onoff == "on":
        #show the selected areas and calculated means
        plt.figure()
        plt.title("Mean of pole 1: " + str(round(mean_intensity_pole_1,2)) + " Fluo threshold = " + str(fluo_thresh))
        #plt.imshow(circle_mask, alpha = 0.7)
        plt.imshow(square_extract_pole_1, alpha = 0.7)
        plt.colorbar()
        #plt.imshow(pole_1_fluo_sum_area, alpha = 0.4,  cmap = plt.cm.Reds)
        plt.figure()
        plt.title("Mean of pole 2: " + str(round(mean_intensity_pole_2,2)) + " Fluo threshold = " + str(fluo_thresh))
        #plt.imshow(circle_mask, alpha = 0.7)
        plt.imshow(square_extract_pole_2, alpha = 0.7)
        plt.colorbar()
        #plt.imshow(pole_2_fluo_sum_area, alpha = 0.4,  cmap = plt.cm.Reds)

        square_extracts_on_fluosize = np.zeros((np.shape(fluo)))
        square_extracts_on_fluosize[np.maximum(pole_1_ypos - radius, 0) : pole_1_ypos + radius + 1, np.maximum(pole_1_xpos - radius, 0) : pole_1_xpos + radius + 1] = 1 #mask pole 1
        square_extracts_on_fluosize[np.maximum(pole_2_ypos - radius, 0) : pole_2_ypos + radius + 1, np.maximum(pole_2_xpos - radius, 0) : pole_2_xpos + radius + 1] = 1 #mask pole 2

        plt.figure()
        plt.imshow(fluo)
        plt.title("Bacteria number: " + str(bact_index) + ", Mean of pole 1: " + str(round(mean_intensity_pole_1,2)) + ", Mean of pole 2: " + str(round(mean_intensity_pole_2,2)))
        plt.imshow(square_extracts_on_fluosize, alpha = 0.2)
        plt.show()


    return(mean_intensity_pole_1, mean_intensity_pole_2, error_message_pole_1, error_message_pole_2, pole_1_xpos, pole_1_ypos, pole_2_xpos, pole_2_ypos)

# main loop function. Can also be used to plot stuff
def fluorescence_detection(label_img_fluo, regions_fluo, bact_index, bord_thresh, fluo_thresh_factor, sigma, pole_coord_list, df, plot_onoff):
    print("\n\nbact index: " + str(bact_index) + "\n\n")

    # extract image information about the current bacteria
    bact_fluo = regions_fluo[bact_index]            # create the region object of a bacteria which can extract a lot of measures about it
    segment = regions_fluo[bact_index].image        # extract the picture of the bacteria containing the segmentation information (ID)
    fluo = regions_fluo[bact_index].image_intensity # extract the fluorescence picture of the bacteria
    skel = skeletonize(bact_fluo.image)             # create a skeleton of the bacteria

    # detect if bacteria lies at the border. If yes, exclude it.
    at_boundary = boundary_check(bact_index = bact_index, bord_thresh = bord_thresh, regions_fluo = regions_fluo, label_img_fluo = label_img_fluo)

    if at_boundary == "yes":
        df.loc[bact_index,"error_message_pole_1"] = "Bacteria too close to the boundary"
        df.loc[bact_index,"error_message_pole_2"] = "Bacteria too close to the boundary"
        df.loc[bact_index,"mean_intensity_pole_1"] = np.nan
        df.loc[bact_index,"mean_intensity_pole_2"] = np.nan
        print("Bacteria " + str(bact_index) + " is too close to the boundary! No calculations done and no plot")

    # else, calculate mean fluorenscence around this pole   
    else:
        # calculate mean fluorenscence around this pole
        # what with the situation as in bacteria 0 ? Where we have more then two ends? is just sorting by distance okay?

        # find the poles = end nodes. They are the pixels with just one neighbour)
        pole_positions, only_two_poles = pole_identification(fluo = fluo, skel = skel, plot_onoff = plot_onoff)

        # if we have more than two poles, we have multiple bacteria in one segment. This is a segmentation mistake, so they are remarked and excluded.
        # can less then 2 segments happen? Not clear yet!
        # if only_two_poles == "no":
        #     df.loc[bact_index,"mean_intensity_pole_1"] = np.nan
        #     df.loc[bact_index,"mean_intensity_pole_2"] = np.nan
        #     df.loc[bact_index,"error_message_pole_1"] = "More than two Poles --> wrong segmentation"
        #     df.loc[bact_index,"error_message_pole_2"] = "More than two Poles --> wrong segmentation"
        if only_two_poles == "0 or 1":
            df.loc[bact_index,"mean_intensity_pole_1"] = np.nan
            df.loc[bact_index,"mean_intensity_pole_2"] = np.nan
            df.loc[bact_index,"error_message_pole_1"] = "0 or 1 poles detected"
            df.loc[bact_index,"error_message_pole_2"] = "0 or 1 poles detected"
            print("0 or 1 pole. No further calculations done and no further plots")

        elif only_two_poles == "3 or more":
            df.loc[bact_index,"mean_intensity_pole_1"] = np.nan
            df.loc[bact_index,"mean_intensity_pole_2"] = np.nan
            df.loc[bact_index,"error_message_pole_1"] = "3 or more poles detected"
            df.loc[bact_index,"error_message_pole_2"] = "3 or more poles detected"
            print("3 or more poles. No further calculations done and no further plots")

        # in the case that we have 2 poles, we can do our normal calculation
        else:
            # note that there was no error for pole calculation
            df.loc[bact_index,"error_message_pole_1"] = "no error"
            df.loc[bact_index,"error_message_pole_2"] = "no error"

            # calculate pole intentsity and memorize pole coordinates
            mean_intensity_pole_1, mean_intensity_pole_2, error_message_pole_1, error_message_pole_2, pole_1_xpos, pole_1_ypos, pole_2_xpos, pole_2_ypos = pole_intensity(pole_positions, fluo, fluo_thresh_factor, df, plot_onoff = plot_onoff, label_img_fluo = label_img_fluo, bact_index = bact_index, sigma = sigma)
            pole_coord_list[:,bact_index] = [pole_1_xpos, pole_1_ypos, pole_2_xpos, pole_2_ypos]

            # case differentiation if the mean intensity actually has a value or not
            if error_message_pole_1 == "Fluo threshold not hit anywhere --> summation over zero pixels":
                df.loc[bact_index,"mean_intensity_pole_1"] = np.nan
                df.loc[bact_index,"error_message_pole_1"] = "Fluo threshold not hit anywhere --> summation over zero pixels"
            elif error_message_pole_1 == "No big difference between the two poles (at least not 3*sd of noise)":
                df.loc[bact_index,"mean_intensity_pole_1"] = mean_intensity_pole_1
                df.loc[bact_index,"error_message_pole_1"] = "Pole difference as big as noise"
            else:
                df.loc[bact_index,"mean_intensity_pole_1"] = mean_intensity_pole_1  

            if error_message_pole_2 == "Fluo threshold not hit anywhere --> summation over zero pixels":
                df.loc[bact_index,"mean_intensity_pole_2"] = np.nan
                df.loc[bact_index,"error_message_pole_2"] = "Fluo threshold not hit anywhere --> summation over zero pixels"
            elif error_message_pole_1 == "No big difference between the two poles (at least not 3*sd of noise)":
                df.loc[bact_index,"mean_intensity_pole_2"] = mean_intensity_pole_2
                df.loc[bact_index,"error_message_pole_2"] = "Pole difference as big as noise"
            else:
                df.loc[bact_index,"mean_intensity_pole_2"] = mean_intensity_pole_2 



    plt.show()

    return(df,pole_coord_list)


# MAIN FUNCTION: Pole identification and Fluorencence mean calculation around each pole

def main(path_seg, path_fluo, bord_thresh, fluo_thresh_factor, plot_onoff, selection_or_all):
    
    # load image data. For both it is an x,y - intensity table (ID is the intensity)
    label_img_seg = imread(path_seg)
    label_img_fluo = imread(path_fluo)

    # extract the noise standard deviation
    sigma = estimate_sigma(label_img_fluo, channel_axis=None, average_sigmas=True)

    # initialize regions.
    # regions measure all kind of different properties of labeled image regions = bacteria
    regions = regionprops(label_img_seg) #only the segmented image has information about the regions
    regions_fluo = regionprops(label_image=label_img_seg, intensity_image=label_img_fluo) #the fluorescenece is selected on the regions, but as an extra intensity map

    # create a dataframe for the bacterial information
    global df
    df = pd.DataFrame(columns=("t","id","mean_intensity_pole_1","mean_intensity_pole_2","error_message_pole_1","error_message_pole_2"))
    time = np.zeros(len(regions_fluo), dtype= int) #we observe only t = 0
    indices = np.arange(len(regions_fluo), dtype = int)
    df.loc[:,"t"] = time
    df.loc[:,"id"] = indices

    #simulate just a selection bacteria
    if selection_or_all == "selection":

        bact_index_selection = list(map(int, input("Enter Bact Index or Multiple Indices (separate by space bar):  ").strip().split()))
        #bact_index_list = int(input("Enter Bact Index or Multiple Indices (separate by space bar):  "))
        pole_coord_list = np.empty((4,len(regions)))
        pole_coord_list[:] = np.nan
        for bact_index in bact_index_selection:
            df, pole_coord_list = fluorescence_detection(label_img_fluo = label_img_fluo, regions_fluo = regions_fluo, bact_index = bact_index, bord_thresh = bord_thresh, fluo_thresh_factor = fluo_thresh_factor, sigma = sigma, pole_coord_list = pole_coord_list, df = df, plot_onoff= plot_onoff)

    #loop over all bacteria
    else: 

        # Main Loop: Pole identification and Fluorencence mean calculation around each pole
        # loop over all bacteria
        # bord_thresh is the border exclusion threshold in pixel. No exclusion with 0.
        bact_index_selection = np.arange(len(regions)) #select all
        pole_coord_list = np.empty((4,len(regions)))
        pole_coord_list[:] = np.nan
        for bact_index in bact_index_selection:
            df, pole_coord_list = fluorescence_detection(label_img_fluo = label_img_fluo, regions_fluo = regions_fluo, bact_index = bact_index, bord_thresh = bord_thresh, fluo_thresh_factor = fluo_thresh_factor, sigma = sigma, pole_coord_list = pole_coord_list, df = df, plot_onoff= plot_onoff)

        
    print("finished!")

    #plot all the bacteria
    plot_everything(pole_coord_list, label_img_seg, label_img_fluo, regions_fluo, df, bact_index_selection)
    plt.show()

    return(df)

#%% Run Program

# set parameters
%matplotlib qt
plot_onoff = "off" #on or off
selection_or_all = "selection" #selection or all
bord_thresh = 5
fluo_thresh_factor = 0 # We count all the pixels with value factor * noise. E.g. if fact = 0, we count all the cells bigger than zero.

# initialize path of segmented phase contrast image and fluoresent image
path_seg = "C:/Joni/Uni/Vorlesungen/AMU/Internship Mignot Lab/python_codes/fluorescent_detection/movie/seg.tif"
path_fluo = "C:/Joni/Uni/Vorlesungen/AMU/Internship Mignot Lab/python_codes/fluorescent_detection/movie/fluo.tif"

df = main(path_seg, path_fluo, bord_thresh, fluo_thresh_factor, plot_onoff, selection_or_all)

#if you just want to run the analysis for one bacterium, type
#fluorescence_detection(regions_fluo, bact_index, bord_thresh, fluo_thresh_factor, df, plot_onoff):

#%% Graph from JBs notes

def plot_mean_distribution(df):
    a = df.loc[:,("mean_intensity_pole_1","mean_intensity_pole_2")].values
    a = a.astype(float)
    a[np.isnan(a)] = 0
    cond = a[:,0] > a[:,1]
    a[cond] =  np.roll(a[cond,:],shift=1,axis=1)
    sort_id = np.argsort(a[:,1])
    b = a[sort_id,:]

    x = np.arange(len(b[:,1]))

   # plt.title("ELEMENTS EXCLUDED: " + str(np.sum(b[:,1] == 0)))
    plt.plot(x,b[:,1], label = "leading pole") 
    plt.plot(x,b[:,0], label = "lagging pole")
    plt.legend()

    excluded_total = str(np.sum(b[:,1] == 0))
    cond_boundary = df.loc[:,"error_message_pole_1"] == "Bacteria too close to the boundary"
    excluded_boundary = str(len(df.loc[cond_boundary,:]))
    cond_fluo_1 = df.loc[:,"error_message_pole_1"] == "Fluo threshold not hit anywhere --> summation over zero pixels"
    cond_fluo_2 = df.loc[:,"error_message_pole_2"] == "Fluo threshold not hit anywhere --> summation over zero pixels"
    cond_fluo =  np.logical_and(cond_fluo_1, cond_fluo_2) 
    excluded_fluo = str(len(df.loc[cond_fluo,:]))
    cond_pole_1 = df.loc[:,"error_message_pole_1"] == "0 or 1 poles detected"
    cond_pole_2 = df.loc[:,"error_message_pole_1"] == "3 or more poles detected"
    cond_pole = np.logical_or(cond_pole_1, cond_pole_2) 
    excluded_pole = str(len(df.loc[cond_pole,:]))

    #look how many poles have a relative deviation smaller than 3 standard deviations.
    sigma = estimate_sigma(label_img_fluo, channel_axis=None, average_sigmas=True) 
    number_similar_poles = str(np.sum(abs(b[:,0] - b[:,1]) / b[:,1] <= 3*sigma / b[:,1])) #exchange 0.1 with like 3* standard deviation or sth like this
    location= np.where(abs(b[:,0] - b[:,1]) / b[:,1] <= 3*sigma / b[:,1])
    plt.plot(x[location],b[location,0][0], "o")

    plt.xlabel("Sorted bacteria index. (Sorted, not the same as normal bacteria index) \n\nNumber of similar poles: " + number_similar_poles + "\n\nELEMENTS EXCLUDED: " + excluded_total + " = \n" + "Wrong Pole Count: " + excluded_pole + "\nBoundary Threshold: " + excluded_boundary + "\nFluorescence Threshold: " + excluded_fluo)
    plt.ylabel("Mean fluorescence value. ")

    plt.show()
^^
# initialize path of segmented phase contrast image and fluoresent image
path_seg = "C:/Joni/Uni/Vorlesungen/AMU/Internship Mignot Lab/python_codes/fluorescent_detection/movie/seg.tif"
path_fluo = "C:/Joni/Uni/Vorlesungen/AMU/Internship Mignot Lab/python_codes/fluorescent_detection/movie/fluo.tif"

fluo_thresh_factor_off = 0
fluo_thresh_factor_on = 1.5
bord_thresh_off = 0
bord_thresh_on = 3

df_b_cut_off_f_cut_off = main(path_seg, path_fluo, bord_thresh = bord_thresh_off, fluo_thresh_factor = fluo_thresh_factor_off, plot_onoff = "off", selection_or_all = "all")
print("Bacteria mean over index.\n Leading pole sorted vs the lagging pole.\n If both means close to each other \n--> no leading pole can be determined")
plt.title("Border treshold (off): " + str(bord_thresh_off) + ". Fluo treshold Factor (off): " + str(fluo_thresh_factor_off))
plot_mean_distribution(df_b_cut_off_f_cut_off)


df_b_cut_on_f_cut_off = main(path_seg, path_fluo, bord_thresh = bord_thresh_on, fluo_thresh_factor = fluo_thresh_factor_off, plot_onoff = "off", selection_or_all = "all")
plt.title("Border treshold (on): " + str(bord_thresh_on) + ". Fluo treshold Factor (off): " + str(fluo_thresh_factor_off))
plot_mean_distribution(df_b_cut_on_f_cut_off)

df_b_cut_off_f_cut_on = main(path_seg, path_fluo, bord_thresh = bord_thresh_off, fluo_thresh_factor = fluo_thresh_factor_on, plot_onoff = "off", selection_or_all = "all") 
plt.title("Border treshold (off): " + str(bord_thresh_off) + ". Fluo treshold Factor (on): " + str(fluo_thresh_factor_on))
plot_mean_distribution(df_b_cut_off_f_cut_on)


df_b_cut_on_f_cut_on = main(path_seg, path_fluo, bord_thresh = bord_thresh_on, fluo_thresh_factor = fluo_thresh_factor_on, plot_onoff = "off", selection_or_all = "all") 
plt.title("Border treshold (on): " + str(bord_thresh_on) + ". Fluo treshold Factor (on): " + str(fluo_thresh_factor_on))
plot_mean_distribution(df_b_cut_on_f_cut_on)




# what could be optimized in terms of speed
#- take the sum just over the skeleton and not over everything

#create the graph from jbs notes
# --> here we can see the effect of threshold exclusions

#what has to be tested:
#find out if the skeletonization properly works! for bacteria number ... for excample, the skeleton went out of the bacteria and thus the calculation sum was not correct

#create the complete plot with calculation areas on each bacteria

#plot the area distribution

# implememt an algo to check if they are close to each other. Exclude the threshold

# exclude small sizes: minsize = 3um and with circularity

#in the later code, exclude low lifetime of tracking

# %%
