import skimage as ski
import numpy as np
from scipy.signal import convolve2d

def index_max_neighbor_sum_convolution(im_fluo):
    """
    Calculate the index of the maximum sum of neighbors using convolution.

    Parameters:
        im_fluo (ndarray): 2D array representing the input image.

    Returns:
        tuple: The index of the maximum sum of neighbors.
    """

    # Define the convolution kernel to compute the sum of neighbors
    kernel = 1/25 * np.array([[1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1]])
    
    # Apply convolution to compute the sum of neighbors
    neighbor_sum = convolve2d(im_fluo, kernel, mode='same', boundary='fill', fillvalue=0)
    
    # Find the index with the maximum sum
    max_index = np.unravel_index(np.argmax(neighbor_sum), neighbor_sum.shape)
    
    return max_index

def extract_bacteria_regionprops_fct(region, df, bact_index):
    
    # extract segmentation img of 1 bacterium. =  the picture of the bacteria containing the segmentation information (ID)
    segment = region.image    

    # extract fluorescence img of 1 bacterium = the fluorescence picture of the bacteria
    fluo = region.image_intensity 

    # Extract the central position of the fluorescent cluster
    # If there is no cluster, the extracted point can be everywhere in the cell
    fluo_max_coords = index_max_neighbor_sum_convolution(fluo)

    # Extract the contours, so we need to increase the frame of the regionprops by one pixel
    im = np.pad(segment, pad_width=1, mode='constant', constant_values=False)
    segment_erode = ski.morphology.erosion(im)
    # Now we have extract the contour we can remove the extra edge pixels 
    # to keep the good coordinate of the local region
    segment_erode = segment_erode[1:-1, 1:-1]
    segment_contour = segment ^ segment_erode
    
    # create a skeleton of the bacteria
    skel = ski.morphology.skeletonize(segment, method="lee") # lee method works generally better and leads to shorter skeletons  

    # extract the min and max coordinates and save them in df
    segment_coords = region.coords
    x_min = np.min(segment_coords[:, 1]) #extract min and max coordinates for translation from local to global COS
    y_min = np.min(segment_coords[:, 0])
    x_max = np.max(segment_coords[:, 1])
    y_max = np.max(segment_coords[:, 0])
    df.loc[bact_index, "x_min_global"] = x_min.astype(int)
    df.loc[bact_index, "y_min_global"] = y_min.astype(int)
    df.loc[bact_index, "x_max_global"] = x_max.astype(int)
    df.loc[bact_index, "y_max_global"] = y_max.astype(int)


    return(segment, segment_contour, fluo, skel, fluo_max_coords)
