#%%
from external_skan_csr import Skeleton
import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
import error_message


# ------------------------------------------------------------------
# DETECTS POLES USING THE LIBRARY SKAN = SKELETON ANALYSIS
# ------------------------------------------------------------------

def skan_pole_detection_fct(skel, segment_contour, bact_index, df):

    #-----------------EXTRACT SKELETON PATH COORDINATES--------------
    try:
        # extract the coordinates of each path
        coords_path = Skeleton(skel).path_coordinates(0)
    except:
        # print("Error: Problem with skan. Probably because only 0 or 1 poles were detected.")
        error_message.create_error_msg_fct(df, bact_index, "skan", affected_poles=[1,2])
        n_paths = 0
        length = 0
        return np.array([np.nan, np.nan]), np.array([np.nan, np.nan]), np.array([np.nan, np.nan]), n_paths, length, np.array([np.nan, np.nan]), np.array([np.nan, np.nan])
    
    # count the number of paths. If there are multiple paths = multiple poles --> segmentation error
    n_paths = Skeleton(skel).n_paths

    # Number of point between the extremities of the skeleton and previous_point 
    # to compute the slope of the skeleton at each extremities.
    previous_point = 4 
    if len(coords_path) <= previous_point:
        error_message.create_error_msg_fct(df, bact_index, "skan", affected_poles=[1,2])
        n_paths = 0
        length = 0
        return np.array([np.nan, np.nan]), np.array([np.nan, np.nan]), np.array([np.nan, np.nan]), n_paths, length, np.array([np.nan, np.nan]), np.array([np.nan, np.nan])
    
    # Search for the pole on the contour starting from the extremities of the skeleton
    # First pole
    dy_0 = coords_path[0, 0] - coords_path[previous_point, 0]
    dx_0 = coords_path[0, 1] - coords_path[previous_point, 1] # It is the inverse in the y-axis (first element is on the top of the image)
    dn_0 = np.sqrt(dy_0**2 + dx_0**2)
    dy_0 /= dn_0
    dx_0 /= dn_0
    saved_point_0 = coords_path[0, :].astype(float)

    while ~segment_contour[int(saved_point_0[0]), int(saved_point_0[1])]:
        saved_point_0[0] += dy_0
        if segment_contour[int(saved_point_0[0]), int(saved_point_0[1])]:
            pass
        else:
            saved_point_0[1] += dx_0

    # Second pole
    dy_1 = coords_path[-1, 0] - coords_path[-1-previous_point, 0]
    dx_1 = coords_path[-1, 1] - coords_path[-1-previous_point, 1] # It is the inverse in the y-axis (first element is on the top of the image)
    dn_1 = np.sqrt(dy_1**2 + dx_1**2)
    dy_1 /= dn_1
    dx_1 /= dn_1
    saved_point_1 = coords_path[-1, :].astype(float)

    while ~segment_contour[int(saved_point_1[0]), int(saved_point_1[1])]:
        saved_point_1[0] += dy_1
        if segment_contour[int(saved_point_1[0]), int(saved_point_1[1])]:
            pass
        else:
            saved_point_1[1] += dx_1

    # Points used for removing the pollution are taking a bit before the edge
    saved_point_0_pollution = saved_point_0.copy()
    saved_point_0_pollution[0] -= 2*dy_0
    saved_point_0_pollution[1] -= 2*dx_0
    saved_point_1_pollution = saved_point_1.copy()
    saved_point_1_pollution[0] -= 2*dy_1
    saved_point_1_pollution[1] -= 2*dx_1

    #-----------------SELECT POLES AS BEGINNING/ END OF PATH--------------
    # extract the beginning and end of each path = poles
    # start_skel_coords = coords_path[0]
    # end_skel_coords = coords_path[-1]
    start_pole_coords = saved_point_0.astype(int)
    end_pole_coords = saved_point_1.astype(int)
    start_pole_coords_pollution = saved_point_0_pollution.astype(int)
    end_pole_coords_pollution = saved_point_1_pollution.astype(int)

    #-----------------FIND MIDDLE NODES-------------------------

    # select x and y coordinates of the path
    path_x = coords_path[:, 0]
    path_y = coords_path[:, 1]

    # compute the distance between all the elements
    dist = np.sqrt((path_x[1:] - path_x[:-1])**2 + (path_y[1:] - path_y[:-1])**2)

    # compute the summed distance along a path
    total_dist = np.zeros(len(coords_path))
    for path_segment_index in range(1, len(coords_path)):
        total_dist[path_segment_index] = total_dist[path_segment_index-1] + dist[path_segment_index-1]
    length = total_dist[-1] # can also be obtained with: length = Skeleton(skel).path_lengths() 

    # the middle node is the node that lies at half of the paths length
    middle_coords_index = np.argmin(np.abs(total_dist - length/2))
    middle_coordinates = coords_path[middle_coords_index]

    # plot
    # plt.imshow(skel)
    # plt.scatter(middle_coordinates[1],middle_coordinates[0])
    # plt.scatter(start[1],start[0])
    # plt.scatter(end[1],end[0])

    # bind output together
    pole_1_coords = start_pole_coords
    pole_2_coords = end_pole_coords
    pole_1_coords_pollution = start_pole_coords_pollution
    pole_2_coords_pollution = end_pole_coords_pollution
    center_coords = middle_coordinates
    return pole_1_coords, pole_2_coords, center_coords, n_paths, length, pole_1_coords_pollution, pole_2_coords_pollution
    
