#%%

import numpy as np
#import numpy own functions
import numpy as np
import pandas as pd
from skimage.measure import regionprops
import matplotlib.pyplot as plt

#import my functions
import fluo_detection
import plt_everything_old
import import_image

#import numpy own functions
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt

#import my functions
import boundary_check
import pole_identification
import pole_intensity
import leading_pole_detection_in_fluo_detection



path_seg = "C:/Joni/Uni/Vorlesungen/AMU/Internship Mignot Lab/python_codes/fluorescent_detection/movie/seg.tif"
path_fluo = "C:/Joni/Uni/Vorlesungen/AMU/Internship Mignot Lab/python_codes/fluorescent_detection/movie/fluo.tif"
path_save = "C:/Joni/Uni/Vorlesungen/AMU/Internship Mignot Lab/python_codes/fluorescent_detection/movie/created_images/"

# set parameters
%matplotlib qt
#plt.close("all")
plot_single_onoff = "off" #on or off. plot for every single bacteria. Good to test if setting are working
plot_final_onoff = "off"
selection_or_all = "all" #selection or all
lead_pole_factor = 1.2 #leading pole factor. How much stronger must a pole be to be the leading pole?
noise_tol_factor = 3 #the noise tolerance factor. We accept a pole difference if it is bigger than e.g. 3*the noise standard deviation
bord_thresh = 5
fluo_thresh =  -100 #  We count all the pixels above this value. E.g. if fact = -inf, we count all the cells on the bacteria. Is useful if the noise can not be removed. generally between [-inf, +10] you have to test using df_single (see below)
pole_on_thresh = 1.4 # is the threshold for when a pole is "above background noise". Has to be hand-tested, using one test run and the function leading_pole_distr

#%%
bact_index = 800

label_img_seg, label_img_fluo, mean_noise, sd_noise = import_image.import_denoise_normalize(path_seg, path_fluo)

# initialize regions. Regions measure all kind of different properties of labeled image regions = bacteria
regions = regionprops(label_img_seg) #only the segmented image has information about the regions
regions_fluo = regionprops(label_image=label_img_seg, intensity_image=label_img_fluo) #the fluorescenece is selected on the regions, but as an extra intensity map


# extract image information about the current bacteria
bact_fluo = regions_fluo[bact_index]            # create the region object of a bacteria which can extract a lot of measures about it
segment = regions_fluo[bact_index].image        # extract the picture of the bacteria containing the segmentation information (ID)
fluo = regions_fluo[bact_index].image_intensity # extract the fluorescence picture of the bacteria
skel = skeletonize(bact_fluo.image)             # create a skeleton of the bacteria

plt.imshow(fluo)
plt.colorbar()

np.mean(fluo[skel == True])
np.std(fluo[skel == True])
# %%
