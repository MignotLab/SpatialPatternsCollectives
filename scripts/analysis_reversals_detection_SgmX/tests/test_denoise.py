#%%
label_img_fluo_wo_bact = np.copy(label_img_fluo) #copy to remove the bacteria fluo just on the copy without bacteria
bacteria_positions = label_img_seg != 0 #find where the bacteria are
label_img_fluo_wo_bact[bacteria_positions] = 0 #set those values to zero. they should not be considered for noise determination
mean_noise = np.mean(label_img_fluo_wo_bact[label_img_fluo_wo_bact != 0]) #we calculate the noise of the mean only from those positions where there are no bacteria.
label_img_fluo_new = label_img_fluo - mean_noise

plt.figure()
plt.imshow(label_img_fluo)
plt.colorbar()

plt.figure()
plt.imshow(label_img_fluo_new)
plt.colorbar()
# %%
