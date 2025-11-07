#%%
import numpy as np
from skimage.restoration import estimate_sigma
import noise_detection
import bacteria_intensity_detection
import PIL

def import_denoise_normalize(t, path_seg_dir, path_fluo_dir):
    
    # extract the path of the current image
    path_seg = path_seg_dir + str(t) + ".tif"
    path_fluo = path_fluo_dir + str(t) + ".tif"
    
    # load image data. For both it is an x,y - intensity table (ID is the intensity)
    label_img_seg = np.array(PIL.Image.open(path_seg))    
    label_img_fluo = np.array(PIL.Image.open(path_fluo))

    # normalize image
    label_img_fluo =  (label_img_fluo - np.mean(label_img_fluo)) / np.std(label_img_fluo)
    
    # denoise (subtract mean noise -> just variance stays)
    mean_noise = noise_detection.detect_noise_fct(label_img_seg,label_img_fluo)
    label_img_fluo = label_img_fluo - mean_noise

    # extract new mean of the denoised picture (should be zero)
    mean_noise = noise_detection.detect_noise_fct(label_img_seg,label_img_fluo)

    # extract the noise standard deviation
    sd_noise = estimate_sigma(label_img_fluo, channel_axis=None, average_sigmas=True)

    # extract the mean bacteria intensity
    mean_bacteria_intensity, sd_bacteria_intensity = bacteria_intensity_detection.detect_bacteria_intensity_fct(label_img_seg,label_img_fluo)

    return(path_seg, path_fluo, label_img_seg, label_img_fluo, mean_noise, sd_noise, mean_bacteria_intensity, sd_bacteria_intensity)