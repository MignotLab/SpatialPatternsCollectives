#%%
import sys
import count_images

# choose which analysis
which_analysis = "normal_analysis" #if you choose fake image analysis, also set frame_end to 0 anywhere



if which_analysis == "normal_analysis":

# 1
    # FOR THE NORMAL ANALYSIS
    class settings_general_fct():
        def __init__(self):
            #-------------------PATHS------------------
            self.path_program_dir = "data_low_density/"
            self.path_tracking = self.path_program_dir+"input/dataframes/tracking.csv"
            self.path_output_dir = self.path_program_dir+"output/"
            self.path_seg_dir = self.path_program_dir+"input/images/seg/"
            self.path_fluo_dir = self.path_program_dir+"input/images/fluo/"

            #-------------------GIVEN PARAMETERS------------------
            self.width = 0.6     # bacteria width in µm (0.7 µm to 1µm to be sure to take the SgmX cluster in the squares centered on the ends of the skeleton)
            self.pcf = 0.0646028 # micrometer - pixel conversion factor (µm/pixel ratio)

# 2
elif which_analysis == "fake_image_analysis":
    # FOR THE FAKE IMAGE ANALYSIS
    class settings_general_fct():
        def __init__(self):
            #-------------------PATHS------------------
            self.path_program_dir = "data_low_density/"
            self.path_tracking = self.path_program_dir+"input/dataframes/tracking.csv"
            self.path_output_dir = self.path_program_dir+"output/"
            self.path_fluo_dir = self.path_output_dir + "images/fake_image/"
            self.path_seg_dir = self.path_program_dir+"input/images/fake_image_seg/"
            self.path_seg_dir_old = self.path_program_dir+"input/images/seg/"
            
            #-------------------GIVEN PARAMETERS------------------
            self.width = 0.6     # bacteria width in µm
            self.pcf = 0.0646028 # micrometer - pixel conversion factor (pixel/µm ratio)
                   
# 3
class settings_create_fake_image_fct():
    def __init__(self):

        # paths and general parameters
        self.path_program_dir = vars(settings_general_fct())["path_program_dir"]
        self.path_output_dir = vars(settings_general_fct())["path_output_dir"]

        self.path_dataframes_dir = self.path_output_dir + "dataframes/fake_image/"
        self.path_fake_image_dir = self.path_output_dir + "images/fake_image/"
        
        # picture intensity character settings.
        # you can define two presets, a normal one (= exact recreation) and a strong one (exagerrated)

        choice_pollution = "normal"
        if choice_pollution == "normal":
            # picture intensity character settings
            self.spot_gaussian_intensity_strong = 600 # amplitude factor of the gaussian fake light
            self.spot_gaussian_sd = 3 # sd of the gaussian bell

        elif choice_pollution == "strong":
            # picture intensity character settings
            self.spot_gaussian_intensity_strong = 800 # amplitude factor of the gaussian fake light
            self.spot_gaussian_sd = 8 # sd of the gaussian bell

        self.mean_bacteria_noise = 30 # noise in a bacterium, after subtracting the mean background noise 
        self.sd_bacteria_noise = 18 # sd in a bacterium 
        self.mean_background_noise = 129 # general image noise mean
        self.sd_background_noise = 11 # general image noise sd
        self.spot_gaussian_intensity_weak = self.spot_gaussian_intensity_strong / 2 # amplitude of specially generated weak lights
        self.protein_diffusion_intensity = self.spot_gaussian_intensity_strong / 24
        self.protein_diffusion_gaussian_sd_proportion_of_half_length = 1
        self.protein_diffusion_permeability_bacteria_self = 1
        self.protein_diffusion_permeability_bacteria_others = 0.4
        self.protein_diffusion_permeability_outside = 0.1

# 4
class settings_parameter_estimation_fct():
    def __init__(self):
        #-------------------PATHS------------------
        # path for everything
        self.path_program_dir = vars(settings_general_fct())["path_program_dir"]
        self.path_output_dir = vars(settings_general_fct())["path_output_dir"]

        self.path_save_image_dir = self.path_output_dir + "images/parameter_estimation/"
        self.path_save_dataframe_dir = self.path_output_dir + "dataframes/parameter_estimation/"
        

        #-------------------SETTINGS------------------
        # plot single bacteria information. on or off
        self.plot_single_onoff = "off" #on or off. plot for every single bacteria. Good to test if setting are working
        # plot final image with all bacteria IN THE FUNCTION (faster). on or off
        self.plot_final_onoff = "off"
        # show THIS PLOT plot or just save it. on or off
        self.show_plot = "on"
        # plot segmentation indices. on or off
        self.plot_indices = "on"
        # enable compression
        self.compression = "off"
        # determine compression quality
        self.compression_quality = 100 #percent
        # select segmentation indices to investigate
        self.selection_or_all = "all" #selection or all
        # select frames
        self.frame_end = 1 # !HAS TO STAY 1 HERE!
        # choose step between frames. has to be divisible by 5
        self.step_frames = 5 
        if self.step_frames % 5 != 0:
            sys.exit("Error. The step between frames has to be divisible by 5.")

        #-------------------DETECTION PARAMETERS------------------
        # import these mostly from the fluorescence detection
        settings_dict_fluo_detec = vars(settings_fluorescence_detection_fct())
        self.lead_pole_factor = settings_dict_fluo_detec["lead_pole_factor"] # leading pole factor. How much stronger must a pole be to be the leading pole? I suggest between 1.2-1.4
        self.noise_tol_factor = settings_dict_fluo_detec["noise_tol_factor"]    # the noise tolerance factor. We accept a pole difference if it is bigger than e.g. 3*the noise standard deviation
        self.bord_thresh = settings_dict_fluo_detec["bord_thresh"]         # bacteria which are this amount of pixel close to the boundary are excluded
        self.fluo_thresh = settings_dict_fluo_detec["fluo_thresh"] # We count all the pixels above this value. E.g. if fact = -inf, we count all the cells on the bacteria. Is useful if the noise can not be removed. generally between [-inf, +10] you have to test using df_single (see below)
        self.min_skel_length = settings_dict_fluo_detec["min_skel_length"] 
        self.pole_on_thresh = 0 # is set to a default value of zero in the parameter estimation   # is the threshold for when a pole is "above background noise". Has to be hand-tested, using one test run and the function leading_pole_distr
        
# 5
class settings_fluorescence_detection_fct():
    def __init__(self):
        #-------------------PATHS------------------
        # path for everything
        self.path_program_dir = vars(settings_general_fct())["path_program_dir"]
        self.path_output_dir = vars(settings_general_fct())["path_output_dir"]

        self.path_save_image_dir = self.path_output_dir + "images/"
        self.path_save_dataframe_dir = self.path_output_dir + "dataframes/"
        

        #-------------------SETTINGS------------------

        # You can plot already at this step, while executing the fluorescence analysis. 
        # This is just kept as an option, but not very useful.

        # plot single bacteria information for all the selcted bacteria. on or off
        self.plot_single_onoff = "off" #on or off. plot for every single bacteria. Good to test if setting are working
        # plot final image with all bacteria IN THE FUNCTION (faster). on or off
        self.plot_final_onoff = "off"
        # show THIS PLOT plot in python. on or off (Otherwise it is just saved)
        self.show_plot = "off"
        # plot segmentation indices. on or off
        self.plot_indices = "on"
        # enable compression
        self.compression = "on"
        # determine compression quality
        self.compression_quality = 80 #percent
        # select segmentation indices to investigate
        self.selection_or_all = "all" #selection or all
        # select frames
        self.frame_end = "all" # For how many frames should the fluorescence analysis run? Last frame's number or "all"
        # choose step between frames. has to be divisible by 5
        self.step_frames = 5 
        
        if self.step_frames % 5 != 0:
            sys.exit("Error. The step between frames has to be divisible by 5.")
        if self.frame_end == "all":
            # Count number of images in the directory
            self.frame_end = count_images.count_and_prepare_images_fct( self.step_frames )
        

        #-------------------DETECTION PARAMETERS------------------
        pcf = vars(settings_general_fct())["pcf"]
        self.lead_pole_factor = 1.4  # leading pole factor. How much stronger must a pole be to be the leading pole? I suggest between 1.2-1.4.
        self.noise_tol_factor = 3    # the noise tolerance factor. We accept a pole difference if it is bigger than e.g. 3*the noise standard deviation
        self.bord_thresh = 5         # bacteria which are this amount of pixel close to the boundary are excluded
        self.min_skel_length = int(2 / pcf)  # in mum, minimum length for a skeleton to be recoginzed as a bacterium
        self.fluo_thresh =  -100     # Outdated: This method is not too useful, still kept in the code. We count all the pixels above this value. E.g. if fact = -inf, we count all the cells on the bacteria. Is useful if the noise can not be removed. generally between [-inf, +10] you have to test using df_single (see below)
        self.pole_on_thresh = 1.4    # Outdated: Setting a value here is meaningless, it is going to be automatically estimated by the program. It is the threshold for when a pole is "above background noise".

# 6
class settings_plot_fluorescence_detection_fct():
    def __init__(self):
        #-------------------PATHS------------------
        # path for everything
        self.path_program_dir = vars(settings_general_fct())["path_program_dir"]
        self.path_output_dir = vars(settings_general_fct())["path_output_dir"]

        self.path_save_image_dir = self.path_output_dir + "images/"
        self.path_save_dataframe_dir = self.path_output_dir + "dataframes/"

        #-------------------SETTINGS------------------
        # plot segmentation indices. on or off
        self.plot_indices = "on"
        # show plot in python or just save it. on or off
        self.show_plot = "off"
        # enable compression
        self.compression = "on"
        # determine compression quality
        self.compression_quality = 80 #percent
        # select frames
        self.frame_end = "all" # For how many frames should be the fluorescence analysis be plotted? Last frame's number or "all"
        # choose step between frames. has to be divisible by 5
        self.step_frames = 5 
        
        if self.step_frames % 5 != 0:
            sys.exit("Error. The step between frames has to be divisible by 5.")
        if self.frame_end == "all":
            # Count number of images in the directory
            self.frame_end = count_images.count_and_prepare_images_fct( self.step_frames )
        
# 7
class settings_reversal_analysis_fct():
    def __init__(self):
        #-------------------PATHS------------------
        self.path_program_dir = vars(settings_general_fct())["path_program_dir"]
        self.path_output_dir = vars(settings_general_fct())["path_output_dir"]

        # path for loading and saving the analysis dataframe
        self.path_save_dataframe_dir = self.path_output_dir + "dataframes/"

        self.choice_filter_out_short_reversals = True
        self.forbidden_frame_distance_factor = 2 # marks the time distance of reversals from which we are sure that they are illegal, counted in distance_factor * step_frames, so a distance factor of 3 with a step size of 5 makes reversals in +15 frames illegal

# 8
class settings_plot_reversal_analysis_fct():
    def __init__(self):

        #-------------------PATHS------------------
        self.path_program_dir = vars(settings_general_fct())["path_program_dir"]
        self.path_output_dir = vars(settings_general_fct())["path_output_dir"]

        self.path_save_dataframe_dir = self.path_output_dir + "dataframes/"
        self.path_save_image_dir = self.path_output_dir + "images/"

        #-------------------SETTINGS------------------
        # plot segmentation indices. on or off
        self.plot_indices = "on"
        # show plot or just save it. on or off
        self.show_plot = "off"
        # enable compression
        self.compression = "off"
        # determine compression quality
        self.compression_quality = 80 #percent
        # select frames
        self.frame_end = "all" # For how many frames should the reversal analysis be plotted. Last frame's number or "all"
        # choose step between frames. has to be divisible by 5
        self.step_frames = 5 
        
        if self.step_frames % 5 != 0:
            sys.exit("Error. The step between frames has to be divisible by 5.")
        if self.frame_end == "all":
            # Count number of images in the directory
            self.frame_end = count_images.count_and_prepare_images_fct( self.step_frames )
        

# class settings_plt_single_bact_fct():
#     def __init__(self):
#         #-------------------PATHS------------------
#         # segmented image directory
#         # self.path_seg_dir = "data_low_density/images/image_sequence/seg/"
#         self.path_seg_dir = "data_low_density/images/full_movies/2023_03_16_flora_sgmx_fluo/seg/2023_03_16_DZ2_SgmX_YFP/"
#         # fluorescence image directory 
#         # self.path_fluo_dir = "data_low_density/images/image_sequence/fluo/"
#         self.path_fluo_dir = "data_low_density/images/full_movies/2023_03_16_flora_sgmx_fluo/fluo/"
#         # save created images directory
#         self.path_save_dir = "data_low_density/images/created_images/single_bact_"
#         # path for saving and loading dataframes
#         self.path_dataframes_dir = "data_low_density/dataframes/single_bact_"
        

#         #-------------------SETTINGS------------------
#         # plot single bacteria information. on or off
#         self.plot_single_onoff = "on" #on or off. plot for every single bacteria. Good to test if setting are working
#         # plot final image with all bacteria IN THE FUNCTION (faster). on or off
#         self.plot_final_onoff = "on"
#         # show THIS PLOT plot or just save it. on or off
#         self.show_plot = "off"
#         # plot segmentation indices. on or off
#         self.plot_indices = "on"
#         # enable compression
#         self.compression = "on"
#         # determine compression quality
#         self.compression_quality = 80 #percent
#         # select segmentation indices to investigate
#         self.selection_or_all = "selection" #selection or all
#         # select frames
#         self.frame_end = 5 #how many frames should be plotted. Number or "all"
#         # choose step between frames. has to be divisible by 5
#         self.step_frames = 5 
#         if self.step_frames % 5 != 0:
#             sys.exit("Error. The step between frames has to be divisible by 5.")
        
#         #-------------------DETECTION PARAMETERS------------------
#         self.lead_pole_factor = 1.4  # leading pole factor. How much stronger must a pole be to be the leading pole? I suggest between 1.2-1.4
#         self.noise_tol_factor = 3    # the noise tolerance factor. We accept a pole difference if it is bigger than e.g. 3*the noise standard deviation
#         self.bord_thresh = 5         # bacteria which are this amount of pixel close to the boundary are excluded
#         self.fluo_thresh =  -100     # We count all the pixels above this value. E.g. if fact = -inf, we count all the cells on the bacteria. Is useful if the noise can not be removed. generally between [-inf, +10] you have to test using df_single (see below)
#         self.pole_on_thresh = 1.4    # is the threshold for when a pole is "above background noise". Has to be hand-tested, using one test run and the function leading_pole_distr

#         #-------------------GIVEN PARAMETERS------------------
#         self.width = 0.7     # bacteria width in um
#         self.pcf = 0.06      # micrometer - pixel conversion factor
        

#%%