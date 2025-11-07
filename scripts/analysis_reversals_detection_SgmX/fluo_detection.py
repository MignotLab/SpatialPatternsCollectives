#import my functions
import boundary_check
import pole_identification
import pole_intensity
import leading_pole_detection_in_fluo_detection
import extract_bacteria_regionprops



# main function. Performs all the operations necessary for checking the goodness of one bacteria and calculating + plotting information about it.



def fluorescence_detection_fct(label_img_fluo, regions_fluo, bact_index, bord_thresh, min_skel_length, fluo_thresh, lead_pole_factor, noise_tol_factor, pole_on_thresh, width, pcf, sd_noise, df, plot_onoff, time):
    #print("\n\ntime: " + str(time) + ",   bact index: " + str(bact_index) + "\n\n")

    # -------------------EXTRACT SINGLE BACTERIA IMAGE INFORMATION-------------------------
    # extract image information about the current bacteria
    bact_region = regions_fluo[bact_index]    # create the region object of a bacteria which can extract a lot of measures about it
    segment, segment_contour, fluo, skel, fluo_max_coords = extract_bacteria_regionprops.extract_bacteria_regionprops_fct(region=bact_region,
                                                                                                                          df=df,
                                                                                                                          bact_index=bact_index)

    # -------------------BOUNDARY FILTER-------------------------
    # detect if bacteria lies at the border. If yes, exclude it.
    at_boundary = boundary_check.boundary_check_fct(bact_index=bact_index, 
                                                    bord_thresh=bord_thresh, 
                                                    regions_fluo=regions_fluo, 
                                                    label_img_fluo=label_img_fluo,
                                                    df=df)

    if at_boundary == "yes":
     #  print("Bacteria " + str(bact_index) + " is too close to the boundary! No calculations done and no plots for it")
        pass
    
    # else, calculate mean fluorenscence around this pole   
    else:
        # -------------------FIND POLE POSITIONS-------------------------
        # calculate mean fluorenscence around this pole

        # find the poles = end nodes
        only_two_poles, sufficient_skel_length = pole_identification.pole_identification_fct(bact_index=bact_index, 
                                                                                             fluo=fluo, 
                                                                                             skel=skel, 
                                                                                             segment_contour=segment_contour, 
                                                                                             fluo_max_coords=fluo_max_coords, 
                                                                                             min_skel_length=min_skel_length, 
                                                                                             df=df, 
                                                                                             plot_onoff=plot_onoff)

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
        
        # in the case that we have 2 poles, we can do our normal calculation
        else:
            # -------------------CALCULATE POLE INTENSITY-------------------------
            # note that there was no error for pole identification --> we can continue our calculation

            # calculate pole intentsity 
            pole_1_n, pole_2_n = pole_intensity.pole_intensity_fct(fluo=fluo, 
                                                                   fluo_thresh=fluo_thresh, 
                                                                   width=width, 
                                                                   pcf=pcf, 
                                                                   df=df, 
                                                                   bact_index=bact_index, 
                                                                   plot_onoff=plot_onoff)

            ## -------------------DECIDE LEADING POLE-------------------------
            # THIS IS NOW DONE IN AN EXTRA FUNCTION SINCE IT HAS TO BE ITERATED
            ## if both pole means exist (at least 1 pixel was included for mean calculation), we can decide a leading pole
            #if pole_1_n != 0 and pole_2_n != 0:
            #    # detect leading pole
            #    leading_pole_detection.detect_lead_pole_fct(df = df, mean_intensity_pole_1 = df.loc[bact_index,"mean_intensity_pole_1"], mean_intensity_pole_2 = df.loc[bact_index,"mean_intensity_pole_2"], pole_1_n = pole_1_n, pole_2_n = pole_2_n, lead_pole_factor = lead_pole_factor, noise_tol_factor = noise_tol_factor, pole_on_thresh = pole_on_thresh, sd_noise = sd_noise, bact_index = bact_index)

    return(df)