# Graph from JBs notes

#%% import
import numpy as np
import pandas as pd
from skimage.io import imread, imsave
from skimage.restoration import estimate_sigma
import matplotlib.pyplot as plt



def plot_mean_distribution(df, path_fluo):
    
    label_img_fluo = imread(path_fluo)

    #sort the data
    plt.figure()
    a = df.loc[:,("mean_intensity_pole_1","mean_intensity_pole_2")].values
    a = a.astype(float)
    a[np.isnan(a)] = 0
    cond = a[:,0] > a[:,1]
    a[cond] =  np.roll(a[cond,:],shift=1,axis=1)
    sort_id = np.argsort(a[:,1])
    b = a[sort_id,:]

    x = np.arange(len(b[:,1]))

    #plot the graph
    plt.title("ELEMENTS EXCLUDED: " + str(np.sum(b[:,1] == 0)))
    plt.plot(x,b[:,1], label = "leading pole") 
    plt.plot(x,b[:,0], label = "lagging pole")
    plt.legend()

    #mark the excluded ones #HERE NOT == BUT CONTAINS RATHER
    # excluded_total = str(np.sum(b[:,1] == 0))
    # cond_boundary = df.loc[:,"error_message_pole_1"] == "Excluded: Bacteria too close to the boundary"
    # excluded_boundary = str(len(df.loc[cond_boundary,:]))
    # cond_fluo_1 = df.loc[:,"error_message_pole_1"] == "Error: Fluo threshold not hit anywhere --> summation over zero pixels"
    # cond_fluo_2 = df.loc[:,"error_message_pole_2"] == "Error: Fluo threshold not hit anywhere --> summation over zero pixels"
    # cond_fluo =  np.logical_and(cond_fluo_1, cond_fluo_2) 
    # excluded_fluo = str(len(df.loc[cond_fluo,:]))
    # cond_pole_1 = df.loc[:,"error_message_pole_1"] == "Excluded: 0 or 1 poles detected"
    # cond_pole_2 = df.loc[:,"error_message_pole_1"] == "Excluded: 3 or more poles detected"
    # cond_pole = np.logical_or(cond_pole_1, cond_pole_2) 
    # excluded_pole = str(len(df.loc[cond_pole,:]))
    #cond_off
    #cond_on
    #cond_small_pole_dif
    #cond_noise_dominates

    # #look how many poles have a relative deviation smaller than 3 standard deviations.
    # sigma = estimate_sigma(label_img_fluo, channel_axis=None, average_sigmas=True) 
    # number_similar_poles = str(np.sum(abs(b[:,0] - b[:,1]) / b[:,1] <= 3*sigma / b[:,1])) #exchange 0.1 with like 3* standard deviation or sth like this
    # location= np.where(abs(b[:,0] - b[:,1]) / b[:,1] <= 3*sigma / b[:,1])
    # plt.plot(x[location],b[location,0][0], "o")

    # #plt.xlabel("Sorted bacteria index. (Sorted, not the same as normal bacteria index) \n\nNumber of similar poles: " + number_similar_poles + "\n\nELEMENTS EXCLUDED: " + excluded_total + " = \n" + "Wrong Pole Count: " + excluded_pole + "\nBoundary Threshold: " + excluded_boundary + "\nFluorescence Threshold: " + excluded_fluo)
    # #plt.ylabel("Mean fluorescence value. ")
    # plt.xlabel("Sorted bacteria index. (Sorted, not the same as normal bacteria index) \n\nNumber of similar poles: " + number_similar_poles + "\n\nELEMENTS EXCLUDED: " + excluded_total + " = \n" + "Wrong Pole Count: " + excluded_pole + "\nBoundary Threshold: " + excluded_boundary + "\nFluorescence Threshold: " + excluded_fluo)
    # plt.ylabel("Mean fluorescence value. ")

    plt.show()



# initialize path of segmented phase contrast image and fluoresent image
path_seg = "C:/Joni/Uni/Vorlesungen/AMU/Internship Mignot Lab/python_codes/fluorescent_detection/movie/seg.tif"
path_fluo = "C:/Joni/Uni/Vorlesungen/AMU/Internship Mignot Lab/python_codes/fluorescent_detection/movie/fluo.tif"
path_save = "C:/Joni/Uni/Vorlesungen/AMU/Internship Mignot Lab/python_codes/fluorescent_detection/movie/created_images/"

# plot values (stay like this)
plot_single_onoff = "off" #on or off. plot for every single bacteria. Good to test if setting are working
plot_final_onoff = "off"
selection_or_all = "all" #selection or all

# fixed parameters, best value to be evaluated at each simulation
lead_pole_factor = 1 #leading pole factor. How much stronger must a pole be to be the leading pole?
noise_tol_factor = 3 #the noise tolerance factor. We accept a pole difference if it is bigger than e.g. 3*the noise standard deviation. Could remain at this value.
bact_noise_factor = 0 # mean_noise + bnf* abs(mean_noise) is the threshold for when a pole is "above background noise". Has to be hand-tested.

# these values exclude bacteria and are to be tested. I suppose that the fluo thresh factor is now useless
fluo_thresh_factor_off = -1000
fluo_thresh_factor_on = 1 #look if mb not useful anymore
bord_thresh_off = 0
bord_thresh_on = 3


import main_loop
df_b_cut_off_f_cut_off = main_loop.main_loop(path_seg, path_fluo, bord_thresh = bord_thresh_off, fluo_thresh_factor = fluo_thresh_factor_off, lead_pole_factor = lead_pole_factor, noise_tol_factor = noise_tol_factor, bact_noise_factor = bact_noise_factor, plot_single_onoff = plot_single_onoff, plot_final_onoff = plot_final_onoff, selection_or_all = selection_or_all, path_save = path_save)
print("Bacteria mean over index.\n Leading pole sorted vs the lagging pole.\n If both means close to each other \n--> no leading pole can be determined")
plt.title("Border treshold (off): " + str(bord_thresh_off) + ". Fluo treshold Factor (off): " + str(fluo_thresh_factor_off))
plot_mean_distribution(df_b_cut_off_f_cut_off, path_fluo)

# df_b_cut_on_f_cut_off = main_lo.main_loop(path_seg, path_fluo, bord_thresh = bord_thresh_on, fluo_thresh_factor = fluo_thresh_factor_off, lead_pole_factor = lead_pole_factor, noise_tol_factor = noise_tol_factor, bact_noise_factor = bact_noise_factor, plot_single_onoff = plot_single_onoff, plot_final_onoff = plot_final_onoff, selection_or_all = selection_or_all, path_save = path_save)
# plt.title("Border treshold (on): " + str(bord_thresh_on) + ". Fluo treshold Factor (off): " + str(fluo_thresh_factor_off))
# plot_mean_distribution(df_b_cut_on_f_cut_off, path_fluo)

# df_b_cut_off_f_cut_on = main_lo.main_loop(path_seg, path_fluo, bord_thresh = bord_thresh_off, fluo_thresh_factor = fluo_thresh_factor_on, lead_pole_factor = lead_pole_factor, noise_tol_factor = noise_tol_factor, bact_noise_factor = bact_noise_factor, plot_single_onoff = plot_single_onoff, plot_final_onoff = plot_final_onoff, selection_or_all = selection_or_all, path_save = path_save)
# plt.title("Border treshold (off): " + str(bord_thresh_off) + ". Fluo treshold Factor (on): " + str(fluo_thresh_factor_on))
# plot_mean_distribution(df_b_cut_off_f_cut_on, path_fluo)

# df_b_cut_on_f_cut_on = main_lo.main_loop(path_seg, path_fluo, bord_thresh = bord_thresh_on, fluo_thresh_factor = fluo_thresh_factor_on, lead_pole_factor = lead_pole_factor, noise_tol_factor = noise_tol_factor, bact_noise_factor = bact_noise_factor, plot_single_onoff = plot_single_onoff, plot_final_onoff = plot_final_onoff, selection_or_all = selection_or_all, path_save = path_save)
# plt.title("Border treshold (on): " + str(bord_thresh_on) + ". Fluo treshold Factor (on): " + str(fluo_thresh_factor_on))
# plot_mean_distribution(df_b_cut_on_f_cut_on, path_fluo)

# %%
