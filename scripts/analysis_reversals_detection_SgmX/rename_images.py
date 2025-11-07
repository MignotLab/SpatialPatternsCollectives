#%%
import os
import numpy as np

def remove_zeros_one_iteration_fct(path_dir):

    # list all files
    old_name = os.listdir(path_dir)

    # remove the .tif
    new_name = ' '.join(old_name).replace('.tif','').split()

    # convert to numpy array and convert to int
    new_name = np.array(new_name).astype(int)

    # convert back to string
    new_name = new_name.astype(str)

    # adding back the +tif
    new_name = [s + ".tif" for s in new_name]

    # perform the renaming
    for i in range(len(new_name)):
        # for the renaming, we have to give the entire path
        # thus recombine folder and file name
        source = path_dir + old_name[i]
        destination = path_dir + new_name[i]
        os.rename(source, destination)

def remove_zeros_fct(path_dir_array):
    for path_dir in path_dir_array:
        remove_zeros_one_iteration_fct(path_dir)


def renumber_images_fct(path_dir, step_frames):
    # import all file names
    old_name = os.listdir(path_dir)

    # sort them correctly
    old_name.sort(key= lambda x: float(x.strip('.tif')))

    # rename them in steps of step_frames
    intermediate_name = ["intermediate" + str(step_frames*int(s.strip(".tif"))) + ".tif" for s in old_name]

    # perform the renaming (intermediate step)
    for i in range(len(old_name)):
        # for the renaming, we have to give the entire path
        # thus recombine folder and file name
        source = path_dir + old_name[i]
        destination = path_dir + intermediate_name[i]
        os.rename(source, destination)

    new_name = [s.strip("intermediate") for s in intermediate_name]

    # perform the renaming
    for i in range(len(old_name)):
        # for the renaming, we have to give the entire path
        # thus recombine folder and file name
        source = path_dir + intermediate_name[i]
        destination = path_dir + new_name[i]
        os.rename(source, destination)

# %%
