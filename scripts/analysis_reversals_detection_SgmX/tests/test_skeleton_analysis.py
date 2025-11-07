#%%
from external_skan_csr import Skeleton


#-----------------------------------
# CHOSE BACT INDEX AND TIME
#----------------------------------
bact_index = 800
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

Skeleton(skel)


start = Skeleton(skel).path_coordinates(0)[0]
end = Skeleton(skel).path_coordinates(0)[-1]
n_paths = Skeleton(skel).n_paths

plt.imshow(skel)
plt.scatter(start[1],start[0])
plt.scatter(end[1],end[0])

coords_path = Skeleton(skel).path_coordinates(0)
path_x = coords_path[:,0]
path_y = coords_path[:,1]
dist = np.sqrt( (path_x[1:] - path_x[:-1])**2 + (path_y[1:] - path_y[:-1])**2)

total_dist = np.zeros(len(coords_path))
for path_segment_index in range(1,len(coords_path)):
    total_dist[path_segment_index] = total_dist[path_segment_index-1] + dist[path_segment_index-1]

length = total_dist[-1] # can also be obtained with: length = Skeleton(skel).path_lengths() 

middle_coords_index = np.argmin(np.abs(total_dist - length/2))
middle_coordinates = coords_path[middle_coords_index]

plt.scatter(middle_coordinates[1],middle_coordinates[0])
plt.show()
# %%
