import numpy as np
import pandas as pd
import settings

# compare the old to the new results
def compare_correction_fct(path_save_dataframe_dir):

    path_analysis = path_save_dataframe_dir + "fluo_analysis.csv"
    analysis = pd.read_csv(path_analysis)
    path_analysis_corrected = path_save_dataframe_dir + "fluo_analysis_corrected.csv"
    analysis_corrected = pd.read_csv(path_analysis_corrected)

    analysis = analysis.reset_index(drop = True)
    analysis = analysis.drop(columns = ["level_0","Unnamed: 0"], errors = "ignore")
    analysis_corrected = analysis_corrected.reset_index(drop = True)
    analysis_corrected = analysis_corrected.reset_index(drop = True)
    analysis_corrected = analysis_corrected.drop(columns = ["level_0","Unnamed: 0"], errors = "ignore")
    def equalp(x, y):
        return (x == y) | (np.isnan(x) & np.isnan(y))
    change_indices = np.where( equalp(x = analysis.loc[:,"leading_pole"].values, y = analysis_corrected.loc[:,"leading_pole"].values) == False )[0]
    print( str(len(change_indices)) + " poles have been switched." )
    old = analysis.loc[change_indices,:]
    new = analysis_corrected.loc[change_indices,:]
    analysis_corrected.loc[change_indices,"changed_leading_pole"] = True

    # visualize this selection
    # firstly write the selection information.
    # we copy the dataframe we want and add the selection column at the right indices
    # plot new 
    df_with_selection_information = analysis_corrected.copy(deep = False)
    df_with_selection_information.loc[change_indices,"selection"] = True
    import plt_mark_bacteria
    plt_mark_bacteria.plt_mark_bacteria_fct(df_with_selection_information, before_or_after_correction = "_after")

    # plot old
    df_with_selection_information = analysis.copy(deep = False)
    df_with_selection_information.loc[change_indices,"selection"] = True
    import plt_mark_bacteria
    plt_mark_bacteria.plt_mark_bacteria_fct(df_with_selection_information, before_or_after_correction = "_before")

    