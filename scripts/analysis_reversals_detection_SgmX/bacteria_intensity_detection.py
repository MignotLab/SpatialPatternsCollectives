import numpy as np

# we calculate the noise of the mean only from those positions where there are no bacteria.
# this is a good approximation, but not perfect, since a lot of fluorescence is outside the bacteria and thus not removed.
# still, it tells us sth about the true background noise, without the bacteria fluorescence 

def detect_bacteria_intensity_fct(label_img_seg,label_img_fluo):
    label_img_fluo_wo_bact = np.copy(label_img_fluo) #copy to remove the bacteria fluo just on the copy without bacteria
    bacteria_positions = label_img_seg != 0 #find where the bacteria are
    mean_bacteria_intensity = np.mean(label_img_fluo_wo_bact[bacteria_positions] ) #we calculate the mean of the noise only from those positions where there are bacteria.
    sd_bacteria_intensity = np.std(label_img_fluo_wo_bact[bacteria_positions] ) #the sd might be a bit wring since the poles are also included.

    return(mean_bacteria_intensity, sd_bacteria_intensity)

