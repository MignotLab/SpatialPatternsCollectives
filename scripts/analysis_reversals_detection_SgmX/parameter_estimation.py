#%% 
import main_loop
import pre_leading_pole_distribution
import matplotlib.pyplot as plt


def parameter_estimation_fct(settings_dict):
    
    # load unfinetuned parameters
    path_dataframes_dir = settings_dict["path_dataframes_dir"]
    lead_pole_factor = settings_dict["lead_pole_factor"]
    pole_on_thresh = settings_dict["pole_on_thresh"]
    noise_tol_factor = settings_dict["noise_tol_factor"]

    # Run analysis for unfinetuned parameters and show plot
    print("Running analysis for non-finetuned parameters:\n") 
    settings_dict["plot_indices"] = "off" #turn index plotting off in this step, since all bacteria show error msgs
    plt.figure
    df_seg_all_t, df_suppl_all_t = main_loop.main_loop_fct(settings_dict)
    plt.show()
    settings_dict["plot_indices"] = "on" #turn index plotting off in this step, since all bacteria show error msgs
    waiting = input("\nTake your time and look carefully at the image in your files. Concentrate your view on a high-density-area. You can see that, due to pollution, the pole detection did not always properly.\n\
We want to define a threshold, above which we can be quite sure that a bacteria is on and not polluted. (pole_on_thresh)\n\
This threshold will increase the performance in high density areas. \nPress any key to continue.")
    # mit gui hier ein bild zeigen von polluted bacterium

    # show leading pole histogram
    plt.figure()
    pre_leading_pole_distribution.pole_intensity_histogram(path_dataframes_dir, lead_pole_factor, pole_on_thresh, noise_tol_factor)
    plt.show()
    waiting = input("\nPress any key to continue.")


    # update parameters and show new plots. Might be repeated several times
    choice_repeat = "y"
    while choice_repeat == "y":

        # User should enter new pole_on_thresh parameters: 
        pole_on_thresh = input("Please enter lead_pole_factor: ")

        # Maybe the user wants to change another parameter
        choice_params = input("\nChange anything else? (N,y)")
        while choice_params == "y":
            settings_dict[choice_params] = input("Enter the value you would like to give: ")
            choice_params = input("\nChange anything else? If no, enter 'no'. If yes, please enter the parameter name: ")

        # repeat analysis with updated parameters
        print("\nChanges are visualized / Analysis is repeated...")
        df_seg_all_t, df_suppl_all_t = main_loop.main_loop_fct(settings_dict)

        # repeat the whole procedure?
        choice_repeat = input("\nLook at the image in your files. Does it now look better? If not, you should repeat the procedure. Repeat? (N,y)")

# %%
