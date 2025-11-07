import numpy as np

# we calculate the noise of the mean only from those positions where there are no bacteria.
# this is a good approximation, but not perfect, since a lot of fluorescence is outside the bacteria and thus not removed.
# still, it tells us sth about the true background noise, without the bacteria fluorescence 

def detect_noise_fct(label_img_seg,label_img_fluo):
    label_img_fluo_wo_bact = np.copy(label_img_fluo) #copy to remove the bacteria fluo just on the copy without bacteria
    bacteria_positions = label_img_seg != 0 #find where the bacteria are
    label_img_fluo_wo_bact[bacteria_positions] = 0 #set those values to zero. they should not be considered for noise determination
    mean_noise = np.mean(label_img_fluo_wo_bact[label_img_fluo_wo_bact != 0]) #we calculate the noise of the mean only from those positions where there are no bacteria.

    return(mean_noise)

