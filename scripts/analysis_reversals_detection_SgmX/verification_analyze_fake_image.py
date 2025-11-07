#%% IDEA
#  firstly change the fluo and the seg path in settings_general

    # then create a fake image

    # run all of the cells in main until reversal analysis

    # --> loop this process

import os
import shutil
import verification_create_fake_image
import numpy as np
import matplotlib.pyplot as plt
%matplotlib qt

#%% initiate list of convergence control
repetition_values = list()

for repetition in range(21):

    import settings
    # copy the first segmentation image to the fake image seg folder
    seg_path_old = vars(settings.settings_general_fct())["path_seg_dir_old"] + "0.tif"
    seg_path_new = vars(settings.settings_general_fct())["path_seg_dir"] + "0.tif"
    shutil.copyfile(seg_path_old, seg_path_new)

    # create fake image
    print("---CREATING FAKE IMAGE---")
    df_fluo_fake_ideal = verification_create_fake_image.create_fake_image_function()

    # define wrong equal and cautious amount depending on iterations
    n_equal = np.array([])
    n_cautious = np.array([])
    n_wrong = np.array([])
    n_wrong_not_cautious = np.array([])
    n_off_as_pole = np.array([])
    n_off_as_off = np.array([])

    # run analysis (just copy the current main file until reversal analysis)

#    #%% ----------PARAMETER ESTIMATION SHORT---------

    # the pre-processing part counts also as iteration = 0
    iteration = 0

    # settings
    import matplotlib.pyplot as plt
    %matplotlib inline
    plt.close("all")
    import settings
    settings_dict_param_estim = vars(settings.settings_parameter_estimation_fct())

    # Before Everything: Prepare and rename images
    import count_images
    image_count = count_images.count_and_prepare_images_fct(settings_dict_param_estim["step_frames"])

    # First, we have to run the fluo analysis one time with default params
    import main_loop
    print("---PARAMETER ESTIMATION STARTED---\n")
    print("Fluorescence Analysis...")
    df_fluo_test, df_suppl_test = main_loop.main_loop_fct(settings_dict_param_estim)
    # --> saves analysis in extra folder "parameter_estimation/"

    # We also have to run the leading pole detection
    import leading_pole_detection_single
    print("Leading Pole Detection with Default Parameters...")
    df_fluo_test = leading_pole_detection_single.leading_pole_detection_fct(settings_dict_param_estim, iteration = iteration)
    # --> overwrites/extends analysis in extra folder "parameter_estimation/"

    # now create one test image to see if the results are acceptable
    #import plt_fluo_analysis
    #print("\nTest image plot...")
    #plt_fluo_analysis.plt_fluo_analysis_fct(settings_dict_param_estim, iteration)

    # estimate parameters
    import pre_leading_pole_distribution
    print("Parameter Estimation...")
    pole_on_thresh = pre_leading_pole_distribution.get_pole_on_thresh_fct(settings_dict_param_estim["path_save_image_dir"], settings_dict_param_estim["path_save_dataframe_dir"], settings_dict_param_estim["lead_pole_factor"], settings_dict_param_estim["pole_on_thresh"], settings_dict_param_estim["noise_tol_factor"], information_on_off = "on", iteration = iteration)
    settings_dict_param_estim["pole_on_thresh"] = pole_on_thresh


  #  #%% ----------FLUORESCENCE ANALYSIS---------

    # settings 
    import matplotlib.pyplot as plt
    %matplotlib qt
    plt.close("all")
    import settings
    settings_dict_fluo_detec = vars(settings.settings_fluorescence_detection_fct())
    settings_dict_fluo_detec["pole_on_thresh"] = settings_dict_param_estim["pole_on_thresh"] #get values from parameter estimation

    # run main loop
    import main_loop
    print("---FLUORESCENCE ANALYSIS STARTED---\n")
    df_fluo, df_suppl = main_loop.main_loop_fct(settings_dict_fluo_detec)
    # --> saves analysis in default folder 

 #   #%% ----------LEADING POLE DETECTION---------

    # this is the very first leading pole detection
    iteration = 0

    # settings
    import settings
    settings_dict_fluo_detec = vars(settings.settings_fluorescence_detection_fct())
    settings_dict_fluo_detec["pole_on_thresh"] = settings_dict_param_estim["pole_on_thresh"] #get values from parameter estimation

    print("---LEADING POLE DETECTION STARTED---\n")
    import leading_pole_detection_single
    df_fluo = leading_pole_detection_single.leading_pole_detection_fct(settings_dict_fluo_detec, iteration = iteration)
    # --> overwrites analysis in default folder 

 #   #%% ----------COMPARE ANALYSIS TO REAL LEADING POLES---------
    # select those cells that have no problems with pole determination etc
    import verification_compare_detection
    n_equal, n_cautious, n_wrong, n_wrong_not_cautious, n_off_as_pole, n_off_as_off, indices_wrong, indices_wrong_not_cautious = verification_compare_detection.compare_detection_fct(df_fluo_fake_ideal, df_fluo, n_equal, n_cautious, n_wrong, n_wrong_not_cautious, n_off_as_pole, n_off_as_off)

 #   #%% Mark the wrongly detected bacteria on a plot
    # import plt_mark_bacteria
    # df_fluo.loc[indices_wrong, "selection"] = True
    # plt_mark_bacteria.plt_mark_bacteria_fct(df_fluo, before_or_after_correction ="", iteration = iteration)
    #%
 #   #%% ----------PLOT FLUORESCENCE ANALYSIS---------

    # # settings
    # import matplotlib.pyplot as plt
    # %matplotlib qt
    # plt.close("all")
    # import settings
    # settings_dict_fluo_plot = vars(settings.settings_plot_fluorescence_detection_fct())

    # # run plot
    # import plt_fluo_analysis
    # print("---FLUORESCENCE PLOT STARTED---\n")
    # plt_fluo_analysis.plt_fluo_analysis_fct(settings_dict_fluo_plot)


 #   #%% ----------REMOVE POLLUTION------------

    # settings
    import remove_pollution
    %matplotlib inline
    import pandas as pd
    import settings
    settings_dict_fluo_detec = vars(settings.settings_fluorescence_detection_fct())
    path_save_dataframe_dir = settings_dict_fluo_detec["path_save_dataframe_dir"]
    path_analysis = path_save_dataframe_dir + "fluo_analysis.csv"
    df_fluo = pd.read_csv(path_analysis)

    # select bacteria for learning:
    # CASE 1: if the detection works not well, please select polluted bacteria's lagging poles where the detection worked correctly
    #selection_polluted = [1274, 1306, 1109, 1194, 1389, 1528, 1493, 1642, 1280, 226, 1127, 1257, 1268, 1520, 1493, 1642, 1595, 1838, 2690, 1421, 1427, 1425, 1509, 1565, 1572, 1495, 1451, 1157, 1092, 961, 932, 845, 650, 826, 799, 742, 554, 429, 202, 245, 323, 457, 778, 1196, 1263, 1305, 1153, 1053, 1107, 1451, 1347, 1331, 1318, 1255, 1089, 1092, 1026, 1008, 1065, 914, 797, 803, 507, 400, 794, 1512, 1924 ]

    # CASE 2: if the leading pole detection works quite well we can also use all the bacteria and dont have to select them
    cond_t0 = df_fluo.loc[:,"t"] == 0 #for the learning, we just take the first frame
    selection_polluted = df_fluo.loc[cond_t0,"seg_id"].values

    # remove pollution
    print("---POLLUTION REMOVAL STARTED---\n")
    print("---LEARNING POLLUTION---\n")
    regression_constant, regression_coefficient = remove_pollution.learn_pollution_fct(selection_polluted)
    print("---REMOVING POLLUTION---\n")
    df_fluo_corrected = remove_pollution.remove_pollution_fct(regression_constant, regression_coefficient)
    # --> saves analysis in analysis_corrected in default folder 


 #   #%% ----------NEW LEADING POLE DETECTION------------
    # ----------SEE THE EFFECT OF POLLUTION REMOVAL------------

    iteration = 1

    # settings
    import settings
    settings_dict_fluo_detec = vars(settings.settings_fluorescence_detection_fct())
    settings_dict_fluo_detec["pole_on_thresh"] = settings_dict_param_estim["pole_on_thresh"] #get values from parameter estimation

    # run new leading pole detection
    print("---2nd LEADING POLE DETECTION STARTED---\n")
    import leading_pole_detection_single
    df_fluo_corrected = leading_pole_detection_single.leading_pole_detection_fct(settings_dict_fluo_detec, iteration = iteration)
    # --> opens analysis_corrected in default folder 
    # --> overwrites analysis_corrected in default folder 

    ## compare the correction to the old algorithm before pollution removal
    settings_dict_fluo_detec = vars(settings.settings_fluorescence_detection_fct())
    choice_compare_correction = "no"
    if choice_compare_correction == "yes":
        # compare the correction to the old algorithm before pollution removal
        import compare_correction
        compare_correction.compare_correction_fct(settings_dict_fluo_detec["path_save_dataframe_dir"])

#    #%% ----------COMPARE ANALYSIS TO REAL LEADING POLES---------
    # select those cells that have no problems with pole determination etc
    import verification_compare_detection
    n_equal, n_cautious, n_wrong, n_wrong_not_cautious, n_off_as_pole, n_off_as_off, indices_wrong, indices_wrong_not_cautious = verification_compare_detection.compare_detection_fct(df_fluo_fake_ideal, df_fluo_corrected, n_equal, n_cautious, n_wrong, n_wrong_not_cautious, n_off_as_pole, n_off_as_off)

#    #%% Mark the wrongly detected bacteria on a plot
    # import plt_mark_bacteria
    # df_fluo_corrected.loc[indices_wrong, "selection"] = True
    # plt_mark_bacteria.plt_mark_bacteria_fct(df_fluo_corrected, before_or_after_correction ="", iteration = iteration)

#    #%% ----------LET POLE-ON-THRESH CONVERGE------------
    #ITERATE PARAMETER ESTIMATION AND LEADING POLE DETECTION

    print("---ITERATE LEADING POLE DETECTION AND PARAMETER ESTIMATION---")

    # settings
    import matplotlib.pyplot as plt
    %matplotlib inline
    plt.close("all")
    import settings
    # use the threshold from the previous iteration to separate the distributions
    settings_dict_param_estim_new = vars(settings.settings_parameter_estimation_fct())
    settings_dict_param_estim_new["pole_on_thresh"] = settings_dict_param_estim["pole_on_thresh"]
    # I have to change the path again now to the normal path, in order load the correct analysis data
    settings_dict_fluo_detec = vars(settings.settings_fluorescence_detection_fct())
    settings_dict_param_estim_new["path_save_dataframe_dir"] = settings_dict_fluo_detec["path_save_dataframe_dir"]
    settings_dict_fluo_detec = vars(settings.settings_fluorescence_detection_fct())

    # initialize values for the parameters to get into the loop
    pole_on_thresh = 1000
    pole_on_thresh_old = settings_dict_param_estim["pole_on_thresh"]

    # import functions for the iteration
    import leading_pole_detection_single
    import pre_leading_pole_distribution

    while (pole_on_thresh - pole_on_thresh_old) > 0.001:

        # remark old pole_on_thresh
        pole_on_thresh_old = pole_on_thresh

        #------------------PARAMETER ESTIMATION-------------------
        pole_on_thresh = pre_leading_pole_distribution.get_pole_on_thresh_fct(settings_dict_param_estim_new["path_save_image_dir"], settings_dict_param_estim_new["path_save_dataframe_dir"], settings_dict_param_estim_new["lead_pole_factor"], settings_dict_param_estim_new["pole_on_thresh"], settings_dict_param_estim_new["noise_tol_factor"], information_on_off = "off", iteration = iteration)
        settings_dict_param_estim_new["pole_on_thresh"] = pole_on_thresh #overwrite the old thresh
        # --> opens analysis_corrected in default folder
        # --> writes nothing but outputs pole_on_thresh

        # the next parameter estimation will be run with the new pole_on_thresh
        settings_dict_param_estim_new["pole_on_thresh"] = pole_on_thresh

        #------------------LEADING POLE DETECTION-------------------
        #------------------(with new pole_on_thresh)-------------------
        # load the new pole_on_thresh into the leading pole detection params
        settings_dict_fluo_detec["pole_on_thresh"] = settings_dict_param_estim_new["pole_on_thresh"] #get values from parameter estimation

        # run new leading pole detection
        df_fluo_corrected = leading_pole_detection_single.leading_pole_detection_fct(settings_dict_fluo_detec, iteration = iteration)
        # --> opens analysis_corrected in default folder 
        # --> overwrites analysis_corrected in default folder 

        # select those cells that have no problems with pole determination etc
        import verification_compare_detection
        n_equal, n_cautious, n_wrong, n_wrong_not_cautious, n_off_as_pole, n_off_as_off, indices_wrong, indices_wrong_not_cautious = verification_compare_detection.compare_detection_fct(df_fluo_fake_ideal, df_fluo_corrected, n_equal, n_cautious, n_wrong, n_wrong_not_cautious, n_off_as_pole, n_off_as_off)


        iteration = iteration + 1

    print("Number of iterations: " + str(iteration))
    print("Pole_on_thresh: " + str(pole_on_thresh))

    choice_compare_correction = "no"
    if choice_compare_correction == "yes":
        print("---VIZUALIZING CHANGES DUE TO THE POLE CORRECTION---")
        # compare the correction to the old algorithm before pollution removal
        import compare_correction
        compare_correction.compare_correction_fct(settings_dict_fluo_detec["path_save_dataframe_dir"])


    # #%% Mark the wrongly detected bacteria on a plot
    # # I COULD DO THIS LATER IN A LOOP FOR ALL CORRECTIONS
    # # IF I WOULD STORE THE DF FOR EACH CORRECTION
    # import plt_mark_bacteria
    # df_fluo_corrected.loc[indices_wrong_not_cautious, "selection"] = True # dont count cautious putting to nan as a mistake
    # #df_fluo_corrected.loc[indices_wrong, "selection"] = True # select real mistakes
    # plt_mark_bacteria.plt_mark_bacteria_fct(df_fluo_corrected, before_or_after_correction ="", iteration = iteration)

#    #%% note the tuples of 

    tuple_indicators = {"n_equal": n_equal, 
                        "n_cautious": n_cautious,
                        "n_wrong": n_wrong, 
                        "n_wrong_not_cautious": n_wrong_not_cautious, 
                        "n_off_as_pole": n_off_as_pole, 
                        "n_off_as_off": n_off_as_off, 
                        "indices_wrong": indices_wrong, 
                        "indices_wrong_not_cautious": indices_wrong_not_cautious }

    repetition_values.append(tuple_indicators)


        #         tuple_indicators = n_equal, n_cautious, n_wrong
        #         repetition_values.append(tuple_indicators)
        # # %%

    # %%
