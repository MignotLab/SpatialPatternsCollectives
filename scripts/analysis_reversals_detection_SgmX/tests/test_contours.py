#%%
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops, find_contours
from skimage.segmentation import find_boundaries
from skimage.io import imread
import cv2

# initialize path of segmented phase contrast image and fluoresent image
path_seg = "C:/Joni/Uni/Vorlesungen/AMU/Internship Mignot Lab/python_codes/fluorescent_detection/movie/seg.tif"
path_fluo = "C:/Joni/Uni/Vorlesungen/AMU/Internship Mignot Lab/python_codes/fluorescent_detection/movie/fluo.tif"

# load the data that is usually provided from the other functions 
label_img_seg = imread(path_seg)
label_img_fluo = imread(path_fluo)
regions_fluo = regionprops(label_image=label_img_seg, intensity_image=label_img_fluo)

segment = regions_fluo[100].image.astype(int)
# closing = cv2.morphologyEx(segment, cv2.MORPH_CLOSE, kernel)
# erosion = cv2.erode(closing, kernel, iterations = 1)
# boundary_img = closing - erosion

contours = find_boundaries(segment, mode = "inner")

plt.imshow(contours)

contours = find_boundaries(label_img_seg, mode = "inner")

plt.imshow(contours)

#plt.imshow(label_img_seg)
#contours = find_contours(segment,0)

#plt.imshow(contours[:,1],contours[:,0],linewidth = 2)
# %%
