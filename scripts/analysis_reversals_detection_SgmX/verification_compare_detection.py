import numpy as np
def compare_detection_fct(df_fluo_fake_ideal, df_fluo, n_equal, n_cautious, n_wrong, n_wrong_not_cautious, n_off_as_pole, n_off_as_off):

    # select all bacteria which have no problems with poles / boundary / skel length
    # in the fake case its just those where no leading pole was positioned
    cond_fake_no_problems = np.isnan(df_fluo_fake_ideal.loc[:, "leading_pole"].values.astype(float)) == False
    amount_fake_no_problems = np.sum(cond_fake_no_problems)

    # in the real case its all those where no fluorescence value was calculated
    cond_analysis_no_problems = np.isnan(df_fluo.loc[:, "mean_intensity_pole_1"].values.astype(float)) == False
    amount_analysis_no_problems = np.sum(cond_analysis_no_problems)

    ideal_pole_list = df_fluo_fake_ideal.loc[cond_fake_no_problems,"leading_pole"]
    analysis_pole_list = df_fluo.loc[cond_analysis_no_problems,"leading_pole"]

    # to compare the ideal pole list with the analysis, we have to make all entries that are 0 to np.nan
    ideal_pole_list[ideal_pole_list == 0] = np.nan

    # now there are two main cases
    # 1. the correct pole was detected ( 1 -> 1, 2 -> 2, nan -> nan)
    equal_list = (ideal_pole_list == analysis_pole_list) | (ideal_pole_list.isna() & analysis_pole_list.isna())
    n_equal = np.append( n_equal, np.sum(equal_list.values.astype(float)) )

    # 2. the incorrect pole was detected  ( 1 -> 2, 2 -> 1, 1/2 -> nan, nan -> 1/2)
    # --> Mistakes and insecurities
    wrong_list = equal_list == False
    n_wrong = np.append( n_wrong , np.sum(wrong_list.values.astype(float)) )
    cond_wrong = wrong_list == True
    indices_wrong = wrong_list[cond_wrong].index

    # --> These two cases add up to all the possibilities

    # EXTRA CASES
    # 3. no pole was detected (program is too cautious) ( 1 -> nan, 2 -> nan )
    cautious_list = (analysis_pole_list.isna() == True) & (ideal_pole_list.isna() == False)
    n_cautious = np.append( n_cautious, np.sum(cautious_list.values.astype(float)) )

    # 4. if we dont count cautiousness as a mistake
    # --> Just mistakes
    wrong_not_cautious_list = wrong_list & (cautious_list == False)
    n_wrong_not_cautious = np.append( n_wrong_not_cautious , np.sum(wrong_not_cautious_list.values.astype(float)) )
    cond_wrong_not_cautious = wrong_not_cautious_list == True
    indices_wrong_not_cautious = wrong_not_cautious_list[cond_wrong_not_cautious].index

    # 5. Off poles are counted as a leaing pole ( nan -> 1, nan -> 2 )
    off_as_pole_list = (ideal_pole_list.isna() == True) & (analysis_pole_list.isna() == False)
    n_off_as_pole = np.append( n_off_as_pole , np.sum(off_as_pole_list.values.astype(float)) )

    # 6. Off poles are correctly counted as off ( nan -> nan, nan -> nan )
    off_as_off_list = (ideal_pole_list.isna() == True) & (analysis_pole_list.isna() == True)
    n_off_as_off = np.append( n_off_as_off , np.sum(off_as_off_list.values.astype(float)) )

    return(n_equal, n_cautious, n_wrong, n_wrong_not_cautious, n_off_as_pole, n_off_as_off, indices_wrong, indices_wrong_not_cautious)

