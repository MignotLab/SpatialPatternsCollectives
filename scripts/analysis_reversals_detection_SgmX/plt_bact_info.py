#%% this function just recalculates everything for one bacteria
def plot_bacterial_information(path_seg_dir, path_fluo_dir, path_dataframes_dir, path_save_dir, bord_thresh, fluo_thresh, lead_pole_factor, noise_tol_factor, pole_on_thresh, width, pcf):
    import matplotlib.pyplot as plt

    global df_single #we declare the dataframe as a global variable, that we can edit in all the subroutines
    import main_loop

    selection_or_all = "selection"
    plot_single_onoff = "on"
    plot_final_onoff = "on"
    t_end = 5

    df_single = main_loop.main_loop_fct(path_seg_dir, path_fluo_dir, path_dataframes_dir, path_save_dir, bord_thresh, fluo_thresh, lead_pole_factor, noise_tol_factor, pole_on_thresh, width, pcf, plot_single_onoff, plot_final_onoff, selection_or_all, t_end)

    return(df_single)
# %%
