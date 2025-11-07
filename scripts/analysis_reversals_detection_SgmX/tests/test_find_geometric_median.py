#%%

#-----------------------------------
# CHOSE BACT INDEX AND TIME
#----------------------------------
bact_index = 1302
t = 0


import matplotlib.pyplot as plt
import pandas as pd
from skimage.morphology import skeletonize
import numpy as np
import pandas as pd
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import import_image

# initialize path of segmented phase contrast image and fluoresent image
path_seg_dir = "C:/Joni/Uni/Vorlesungen/AMU/Internship_Mignot_Lab/Fluorescence_Detection/images/image_sequence/seg/"
path_fluo_dir = "C:/Joni/Uni/Vorlesungen/AMU/Internship_Mignot_Lab/Fluorescence_Detection/images/image_sequence/fluo/"

path_seg, path_fluo, label_img_seg, label_img_fluo, mean_noise, sd_noise = import_image.import_denoise_normalize(time = t, path_seg_dir = path_seg_dir, path_fluo_dir = path_fluo_dir)

# initialize regions. Regions measure all kind of different properties of labeled image regions = bacteria
regions = regionprops(label_img_seg) #only the segmented image has information about the regions
regions_fluo = regionprops(label_image=label_img_seg, intensity_image=label_img_fluo) #the fluorescenece is selected on the regions, but as an extra intensity map

# extract image information about the current bacteria
bact_fluo = regions_fluo[bact_index]            # create the region object of a bacteria which can extract a lot of measures about it
segment = regions_fluo[bact_index].image        # extract the picture of the bacteria containing the segmentation information (ID)
fluo = regions_fluo[bact_index].image_intensity # extract the fluorescence picture of the bacteria
skel = skeletonize(bact_fluo.image) 

#%% extract image information about the current bacteria
bact_fluo = regions_fluo[bact_index]            # create the region object of a bacteria which can extract a lot of measures about it
segment = regions_fluo[bact_index].image        # extract the picture of the bacteria containing the segmentation information (ID)
fluo = regions_fluo[bact_index].image_intensity # extract the fluorescence picture of the bacteria
skel = skeletonize(bact_fluo.image)   

a = np.where(skel == 1)
#%% method 1
y = np.median(a[0])
x = np.median(a[1])
plthelp = np.zeros(np.shape(skel))
plthelp[np.round(y).astype(int),np.round(x).astype(int)] = 1
plt.figure()
plt.imshow(skel.astype(int) + 3* plthelp)
plt.show()

#%% method 2
from scipy.optimize import minimize
x = [point for point in a[0]]
y = [point for point in a[1]]

x0 = np.array([sum(x)/len(x),sum(y)/len(y)])
def dist_func(x0):
    return sum(((np.full(len(x),x0[0])-x)**2+(np.full(len(x),x0[1])-y)**2)**(1/2))
res = minimize(dist_func, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
res.x
# array([12.58942487,  3.51573846,  7.28710679])
resy = res.x[0]
resx = res.x[1]

plt.figure()
plthelp = np.zeros(np.shape(skel))
plthelp[np.round(resy).astype(int),np.round(resx).astype(int)] = 1
plt.imshow(skel.astype(int) + 3* plthelp)
plt.show()

