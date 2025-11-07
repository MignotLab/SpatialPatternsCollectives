
#%% ----------CREATE FAKE IMAGE---------
import matplotlib.pyplot as plt #otherwise PIL does not work ^^
import settings
from skimage.measure import regionprops
import boundary_check
import pole_identification
import extract_bacteria_regionprops
import dataframe
import numpy as np
from tqdm import tqdm
import import_image

#test if an array is contained in a list of arrays (https://stackoverflow.com/questions/23979146/check-if-numpy-array-is-in-list-of-numpy-arrays)
def arreqclose_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if elem.size == myarr.size and np.allclose(elem, myarr)), False)


#---------------SETTINGS ------------------------
def create_fake_image_function():
    # general settings
    settings_dict_general = vars(settings.settings_general_fct())
    path_seg_dir = settings_dict_general["path_seg_dir"]
    #path_fluo_dir = settings_dict_general["path_fluo_dir"]
    path_fluo_dir =  "C:/Joni/Uni/Vorlesungen/AMU/Internship_Mignot_Lab/Fluorescence_Detection/input/images/full_movies/2023_03_16_flora_sgmx_fluo/fluo/"
    width = settings_dict_general["width"] 
    pcf = settings_dict_general["pcf"] 

    # specific settings
    settings_dict = vars(settings.settings_create_fake_image_fct())
    path_fake_image_dir = settings_dict["path_fake_image_dir"]
    settings_dict_fluorescence_analysis = vars(settings.settings_fluorescence_detection_fct())
    bord_thresh = settings_dict_fluorescence_analysis["bord_thresh"]
    min_skel_length = settings_dict_fluorescence_analysis["min_skel_length"]

    mean_bacteria_noise = settings_dict["mean_bacteria_noise"] #values after subtraction from the background values
    sd_bacteria_noise = settings_dict["sd_bacteria_noise"] #values after subtraction from the background values
    mean_background_noise = settings_dict["mean_background_noise"] 
    sd_background_noise = settings_dict["sd_background_noise"]
    spot_gaussian_intensity_strong = settings_dict["spot_gaussian_intensity_strong"] # amplitude of the gaussian fake light
    spot_gaussian_intensity_weak = spot_gaussian_intensity_strong / 2 # amplitude of specially generated weak lights
    spot_gaussian_sd = settings_dict["spot_gaussian_sd"] # sd of the gaussian bell
    protein_diffusion_intensity = settings_dict["protein_diffusion_intensity"]
    protein_diffusion_gaussian_sd_proportion_of_half_length = settings_dict["protein_diffusion_gaussian_sd_proportion_of_half_length"]
    protein_diffusion_permeability_bacteria_self = settings_dict["protein_diffusion_permeability_bacteria_self"]
    protein_diffusion_permeability_bacteria_others = settings_dict["protein_diffusion_permeability_bacteria_others"]
    protein_diffusion_permeability_outside = settings_dict["protein_diffusion_permeability_outside"]


    # non-imported settings
    width_px = np.ceil(width / pcf).astype(int)
    radius = int(width_px/2)
    plot_onoff = "off"


    #----------IMPORT SEGMENTATION IMAGE AND CHECK POLE LOCATION / BOUNDARY FILTER ------------------------

    # import image props
    path_seg, path_fluo, label_img_seg, label_img_fluo, mean_noise, sd_noise, mean_bacteria_intensity, sd_bact_intensity = import_image.import_denoise_normalize(0, path_seg_dir, path_fluo_dir)

    # start pole detection and filters just as in the analysis

    # set dataframe FOR ONE FRAME as global variable and import main loop, which used df
    global df #we declare the dataframe as a global variable, that we can edit in all the subroutines
    global df_suppl

    # extract region properties
    regions_fluo = regionprops(label_image=label_img_seg, intensity_image=label_img_fluo) #the fluorescenece is selected on the regions, but as an extra intensity map

    # create a dataframe FOR ONE TIME for the bacterial information
    df = dataframe.initialize_dataframe_fct(length = len(regions_fluo), time = 0)

    bact_index_all = np.arange(len(regions_fluo)) #select all


    # start boundary and pole check
    print("Boundary Check and Pole detection...")
    for bact_index in tqdm(bact_index_all):

        # -------------------EXTRACT SINGLE BACTERIA IMAGE INFORMATION-------------------------
        # extract image information about the current bacteria
        bact_region = regions_fluo[bact_index]    # create the region object of a bacteria which can extract a lot of measures about it
        segment, fluo, skel = extract_bacteria_regionprops.extract_bacteria_regionprops_fct(region = bact_region, df = df, bact_index = bact_index)

        # -------------------BOUNDARY FILTER-------------------------
        # detect if bacteria lies at the border. If yes, exclude it.
        at_boundary = boundary_check.boundary_check_fct(bact_index = bact_index, bord_thresh = bord_thresh, regions_fluo = regions_fluo, label_img_fluo = label_img_fluo, df = df)

        if at_boundary == "yes":
        #  print("Bacteria " + str(bact_index) + " is too close to the boundary! No calculations done and no plots for it")
            pass
        
        # else, calculate mean fluorenscence around this pole   
        else:
            # -------------------FIND POLE POSITIONS-------------------------
            # calculate mean fluorenscence around this pole

            # find the poles = end nodes
            only_two_poles, sufficient_skel_length = pole_identification.pole_identification_fct(bact_index = bact_index, fluo = fluo, skel = skel, min_skel_length = min_skel_length, df = df, plot_onoff = plot_onoff)

            # if we have less than two poles, it usually is because bacteria are too small. 
            if only_two_poles == "0 or 1":
                #print("0 or 1 pole. No further calculations done and no further plots for it")
                pass
            
            # if we have more than two poles, we have multiple bacteria in one segment. This is a segmentation mistake, so they are remarked and excluded
            elif only_two_poles == "3 or more":
                #print("3 or more poles. No further calculations done and no further plots for it")
                pass

             # if the skeleton is too small, it is probably just dirt or a segmentation error
            elif sufficient_skel_length == False:
                pass
            
            # in the case that we have 2 poles, we can generate artificial pole lights
            else:
                df.loc[bact_index, "selection"] = "yes"

                # -------------------DECIDE LEADING POLES RANDOMLY-------------------------
                # (zero means that leading pole is off)
                df.loc[bact_index, "leading_pole"] = np.random.choice([1,2,0], p = [0.45, 0.45, 0.1]) 


                
    #---------------CREATE FAKE IMAGE ------------------------

    # initialize fake fluorescence image        
    label_img_fluo_fake_mask = np.copy(label_img_seg) > 0
    label_img_fluo_fake = np.zeros(np.shape(label_img_seg))


    # select those bacteria that are not bugged, i.e. have two poles and dont lie at the boundary
    cond_selection = df.loc[:, "selection"] == "yes"
    bact_index_selection = df.loc[cond_selection, "seg_id"].values
    shape = np.shape(label_img_fluo)
    m = shape[0]
    n = shape[1]

    #---------------GENERAL RECIPE ------------------------

    # loop over each of them to 
    # 0. shift pole a bit to the end of the head
    # 1. create a fake gaussian light at the pole
    # 2. create bacteria background noise (fluorescent protein gradient or homogenous noise)
    # 3. create general background noise

    #---------------LOOP OVER ALL BACTERIA ------------------------

    print("Creating fake Fluorescence...")
    for bact_index in tqdm(bact_index_selection):
        
        # randomly choose if pole is strong or weak
        pole_on_or_off = np.random.choice(["on", "weak"], p = [0.9, 0.1])

        # if the pole is on or weak, create fake fluorescence
        if df.loc[bact_index, "leading_pole"] != 0:

            # create data (assume spot sd constant)
            if pole_on_or_off == "on":
                spot_intensity = spot_gaussian_intensity_strong
            elif pole_on_or_off == "weak":
                spot_intensity = spot_gaussian_intensity_weak

            # select the randomly chosen leading pole
            leading_pole = df.loc[bact_index,"leading_pole"]

            # just memorize all the bacteria's coordinates
            bact_coords = regions_fluo[bact_index].coords

            #---------------FAKE LEADING POLE LIGHT ------------------------
            # select pole coordinates
            pole_x_global = df.loc[bact_index, ["pole_" + str(leading_pole) + "_x_local"]].values[0] + df.loc[bact_index, "x_min_global"]
            pole_y_global = df.loc[bact_index, ["pole_" + str(leading_pole) + "_y_local"]].values[0] + df.loc[bact_index, "y_min_global"]

            # select global center coords 
            center_x_global = df.loc[bact_index, "center_x_local"] + df.loc[bact_index, "x_min_global"]
            center_y_global = df.loc[bact_index, "center_y_local"] + df.loc[bact_index, "y_min_global"]

            # define a random shift in the direction of the bacteria head:
            # firstly draw the direction from center to pole as an approximate direction of orientation
            orientation_vector = np.array([ pole_y_global - center_y_global , pole_x_global - center_x_global ])
            length_orientation_vector = np.linalg.norm(orientation_vector)
            if length_orientation_vector != 0:
                orientation_vector_unit = orientation_vector / length_orientation_vector
    
                # now that we have the right direction, we can just draw the x-and y amplification randomly:
                # prepare drawing possibilities
                shift_x_choice = np.arange(0,+round(0.5*radius) + 1, 1)
                shift_y_choice = np.arange(0,+round(0.5*radius) + 1, 1)  
                
                # draw the pole shift.
                # we just draw another time, in case the pole lands outside the bacteria
                while arreqclose_in_list( np.array([pole_y_global,pole_x_global ]) , bact_coords ) == False:
                    shift_x = orientation_vector_unit[1] * np.random.choice( shift_x_choice, p = shift_x_choice**2 / np.sum(shift_x_choice**2) ) #the probability to have higher values is higher
                    shift_y = orientation_vector_unit[0] * np.random.choice( shift_y_choice, p = shift_y_choice**2 / np.sum(shift_y_choice**2) )

                    # now apply the shift
                    pole_x_global = np.round( pole_x_global + shift_x ).astype(int)
                    pole_y_global = np.round( pole_y_global + shift_y ).astype(int)

                # select x and y coordinates on which we should work.
                # we only make changes to the picture around the poles in a radius of width = 2*radius
                y = np.r_[np.maximum(pole_y_global - 2 * radius, 0) : np.minimum(pole_y_global + 2 * radius + 1, m)]
                x = np.r_[np.maximum(pole_x_global - 2 * radius, 0) : np.minimum(pole_x_global + 2 * radius + 1, n)]
                Y,X = np.meshgrid(y,x)
                square_extract_pole = label_img_fluo_fake[Y,X]
                square_extract_pole = spot_intensity * np.exp( - ( ( X - pole_x_global )**2 / ( 2 * spot_gaussian_sd**2 ) + (Y - pole_y_global)**2 / ( 2 * spot_gaussian_sd**2 ) ) )

                # add fake gaussian lights on the fake fluo image
                label_img_fluo_fake[Y,X] = label_img_fluo_fake[Y,X] + square_extract_pole

            # else: just dont change the fluorescence

            #---------------BACTERIA BACKGROUND NOISE ------------------------
            # CASE 1: gradient from pole to center
            # add protein diffusion
            # a bacteria has a fluorescent gradient, which is spreads a bit to other bacteria bodies
            # and also (a bit weaker) to the environment

            # here, we work on another domain: from the pole to the center of the bacterium
            # we find out the distance between the pole and center, this times a given factor will be the standard deviation
            # of a second gaussian, that has different permeabilities: a high permeability in the bacteria isself, a small one in the neighbours and a low one outside
            distance_p_c = np.round( np.linalg.norm( [pole_y_global - center_y_global , pole_x_global - center_x_global]  ) ).astype(int)
            protein_diffusion_sd = distance_p_c * protein_diffusion_gaussian_sd_proportion_of_half_length

            if distance_p_c != 0:

                # select x and y coordinates on which we should work.
                # here, we work on another domain: from the pole to the center of the bacterium
                y = np.r_[np.maximum(pole_y_global - 2 * distance_p_c, 0) : np.minimum(pole_y_global + 2 * distance_p_c + 1, m)]
                x = np.r_[np.maximum(pole_x_global - 2 * distance_p_c, 0) : np.minimum(pole_x_global + 2 * distance_p_c + 1, n)]
                Y,X = np.meshgrid(y,x)
                square_extract_pole_big = label_img_fluo_fake[Y,X]
                square_extract_pole_big = protein_diffusion_intensity * np.exp( - ( ( X - pole_x_global )**2 / ( 2 * protein_diffusion_sd**2 ) + (Y - pole_y_global)**2 / ( 2 * protein_diffusion_sd**2 ) ) )
                
                # multiply the regions inside bacteria and outside with different permeabilities
                inside_this_bacteria = label_img_seg[Y,X] == bact_index
                square_extract_pole_big[inside_this_bacteria] = protein_diffusion_permeability_bacteria_self * square_extract_pole_big[inside_this_bacteria]
                inside_other_bacteria = label_img_seg[Y,X] != 0 & bact_index
                square_extract_pole_big[inside_other_bacteria] = protein_diffusion_permeability_bacteria_others * square_extract_pole_big[inside_other_bacteria]
                outside_bacteria =  label_img_seg[Y,X] == 0
                square_extract_pole_big[outside_bacteria] = protein_diffusion_permeability_outside * square_extract_pole_big[outside_bacteria]

                # add fake protein diffusion fluorescence
                label_img_fluo_fake[Y,X] = label_img_fluo_fake[Y,X] + square_extract_pole_big

            # else: just dont change the fluorescence

            # CASE 2: "homogenous" noise
            # bact_coords = regions_fluo[bact_index].coords
            # for coords_pair in bact_coords:
                # bact_background_noise = np.random.normal(mean_bacteria_noise , sd_bacteria_noise)
                # label_img_fluo_fake[coords_pair[0], coords_pair[1]] = label_img_fluo_fake[coords_pair[0], coords_pair[1]] + bact_background_noise

        # if the poles are off, we just have a homogenous background noise
        else:
            bact_coords = regions_fluo[bact_index].coords
            for coords_pair in bact_coords:
                bact_background_noise = np.random.normal(mean_bacteria_noise , sd_bacteria_noise)
                label_img_fluo_fake[coords_pair[0], coords_pair[1]] = label_img_fluo_fake[coords_pair[0], coords_pair[1]] + bact_background_noise


    # if you include randomness: since some gaussians were drawn as too strong for a 16 bit image, they have to be corrected to the maximum values of the original pictures
    label_img_fluo_fake[label_img_fluo_fake > 65535] = 65535
    label_img_fluo_fake[label_img_fluo_fake < 0] = 0

    #---------------GENERAL BACKGROUND NOISE ------------------------
    # add background noise
    label_img_fluo_fake = label_img_fluo_fake + np.random.normal(mean_background_noise , sd_background_noise, size = np.shape(label_img_fluo_fake))


    #---------------VIZUALIZE FAKE IMAGE------------------------

    plt.close("all")
    plt.figure(1)
    plt.title("Fake image")
    label_img_seg_binary = label_img_seg > 0
    #plt.imshow(label_img_seg_binary, cmap = "gray")
    plt.imshow(label_img_fluo_fake, vmin = 0, vmax = 1000, alpha = 0.8) 
    plt.colorbar()
    # plt.scatter(pole_x_global, pole_y_global)


    #import denormalized fluo image
    import PIL
    path_fluo = path_fluo_dir + "0.tif"
    label_img_fluo_unedited = np.array(PIL.Image.open(path_fluo))

    plt.figure(2)
    plt.title("Real image")
    #plt.imshow(label_img_seg_binary, cmap = "gray")
    plt.imshow(label_img_fluo_unedited, vmin = 0, vmax = 1000, alpha = 0.8) 
    plt.colorbar()
    #plt.scatter(pole_x_global, pole_y_global)

    # save fluo image:
    path_fake_image =  path_fake_image_dir + '0.tif'
    img = PIL.Image.fromarray(label_img_fluo_fake)
    img.save(path_fake_image)
    fake_image_test = np.array(PIL.Image.open(path_fake_image))

    # return the ideal dataframe where we know all the correct leading poles
    return(df)