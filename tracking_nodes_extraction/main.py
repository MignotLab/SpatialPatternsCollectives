# %%
import numpy as np
import pandas as pd

from extract_skeleton_nodes import ExtractSkeletonNodes
from parameters import Parameters
from tools import Tools

par = Parameters()
tool = Tools()
# Put the number of image of the movie
n_im = 50
indices_frame_df = np.arange(0, n_im, 1)
n_jobs = 4
esc = ExtractSkeletonNodes(indices_frame_df=indices_frame_df, n_im=n_im, n_jobs=n_jobs)

# EXTRACT SKELETON NODES
df = esc.start()
file_name_save = par.name_folder_csv+str(par.n_nodes)+"_nodes_"+par.name_file_csv
esc.save(df=df, path=file_name_save)

# APPLY CORRECTIONS IF NEEDED (MAINLY DUE TO THE KALMAN FILTER TRACKER)
file_name_save = par.name_folder_csv+str(par.n_nodes)+"_nodes_"+par.name_file_csv
df = pd.read_csv(file_name_save)
df_cor = esc.correction_detection(df=df)
file_name_save_cor = par.name_folder_csv+"correction_"+str(par.n_nodes)+"_nodes_"+par.name_file_csv
esc.save(df=df_cor, path=file_name_save_cor)


















# %% TEST SKAN
# import pandas as pd 
# from skimage.measure import regionprops
# from skimage.morphology import skeletonize
# from skan.csr import Skeleton as skan_skel
# import cv2
# import matplotlib.pyplot as plt


# label_image = cv2.imread("W:/jb/movie_analyse/100X/2023_06_01_SgmX-YFP-OD-01_low_density/seg_images/0.tif", cv2.IMREAD_UNCHANGED)
# regionmask = regionprops(label_image=label_image)
# plt.figure()
# plt.imshow(regionmask[0].image)
# skel = skeletonize(regionmask[0].image, method='lee')
# plt.figure()
# plt.imshow(skel)
# coords_skel = skan_skel(skel).path_coordinates(0)