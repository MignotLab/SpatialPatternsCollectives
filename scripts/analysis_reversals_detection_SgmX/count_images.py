import os
import rename_images
import time
import sys
import numpy as np
import settings

def count_and_prepare_images_fct(step_frames):
    
    #------------------IMPORT GENERAL SETTINGS---------------------
    settings_dict_general = vars(settings.settings_general_fct())
    path_seg_dir = settings_dict_general["path_seg_dir"]
    path_fluo_dir = settings_dict_general["path_fluo_dir"]
    print("The segmentation images are imported from:\n" + str(path_seg_dir))
    print("The fluorescence images are imported from:\n" + str(path_fluo_dir) + "\n")

    #------------------COUNT IMAGES---------------------
    # count images
    n_seg = len([entry for entry in os.listdir(path_seg_dir) if os.path.isfile(os.path.join(path_seg_dir, entry))])
    n_fluo = len([entry for entry in os.listdir(path_fluo_dir) if os.path.isfile(os.path.join(path_fluo_dir, entry))])
    
    #------------------EQUAL AMOUNT?---------------------
    # if there are as many fluo as seg images, we just return the
    # amount of images and leave
    if n_seg != n_fluo:
        print("Warning: Nr. of seg. images is not equal to nr. of fluo- images!", flush=True)
        print("The program examines if this causes troubles...", flush=True)

        # if this is not the case, we have to make a number of tests

        #------------------ENOUGH IMAGES?---------------------
        # find out if there are enough fluo images for the segmented images
        
        try_again_choice = "yes"
        while try_again_choice != "no":
            if n_seg / n_fluo != step_frames:
                print("\nThe amount of seg and fluo images do not correspond, \nsince n_seg / n_fluo is not divisible into steps of " + str(step_frames) + "!", flush=True)
                print("There are ", str(n_seg), " segmented images and ", n_fluo, " fluo images. \nPlease remove ", n_seg % n_fluo, " image(s) from the right stack." , flush=True)
                print("The easiest way is to remove the last image(s).", flush=True) 
                print("\n\nHave you removed the image and want to try again?\nPlease enter your choice.", flush = True)             
                try_again_choice = input("Have you removed the image? Try again? (Yes/no)")
                n_seg = len([entry for entry in os.listdir(path_seg_dir) if os.path.isfile(os.path.join(path_seg_dir, entry))]) #read new amounts
                n_fluo = len([entry for entry in os.listdir(path_fluo_dir) if os.path.isfile(os.path.join(path_fluo_dir, entry))])

                if (n_seg % n_fluo != 0) and  (try_again_choice == "no"):
                    sys.exit("Error: Please fix the amount of images.")

            else:
                try_again_choice = "no"

                
            

        # if this test succeds, go on with the next test

        #------------------ZEROS IN FRONT?---------------------
        # do a test if images have wrong names:

        # find all files in this directory 
        list_seg = os.listdir(path_seg_dir)
        list_fluo = os.listdir(path_fluo_dir)

        # sort by number
        list_seg.sort(key= lambda x: float(x.strip('.tif')))
        list_fluo.sort(key= lambda x: float(x.strip('.tif')))

        # print first and last item
        print("\n\nSegmentation images go from: " + str(list_seg[0]) +  " to: " + str(list_seg[-1]) + " in steps of " + str(int(list_seg[1].strip(".tif")) - int(list_seg[0].strip(".tif"))), flush=True)
        print("Fluorescence images go from: " + str(list_fluo[0]) +  " to: " + str(list_fluo[-1]) + " in steps of " + str(int(list_fluo[1].strip(".tif")) - int(list_fluo[0].strip(".tif"))) + "\n", flush=True)

        # detect and remove zeros from the beginning of the file names
        if (list_seg[0] != "0.tif") or (list_fluo[0] != "0.tif"):
                print("\nThere are unnecessary zero digits in front. The program must rename it.", flush=True)
                print("Rename? Please enter your choice.", flush=True)
                rename_choice = input("Rename? (yes/no)")
                if rename_choice == "yes":
                    # rename images in the directory in the case that they have unnecessary zeros in front
                    rename_images.remove_zeros_fct( [path_seg_dir, path_fluo_dir] )
                    
                    # display changes
                    print("\nChanges applied!")
                    list_seg = os.listdir(path_seg_dir)
                    list_fluo = os.listdir(path_fluo_dir)
                    list_seg.sort(key= lambda x: float(x.strip('.tif')))
                    list_fluo.sort(key= lambda x: float(x.strip('.tif')))
                    print("Segmentation images now go from: " + str(list_seg[0]) +  " to: " + str(list_seg[-1]) + " in steps of " + str(int(list_seg[1].strip(".tif")) - int(list_seg[0].strip(".tif"))))
                    print("Fluorescence images now go from: " + str(list_fluo[0]) +  " to: " + str(list_fluo[-1]) + " in steps of " + str(int(list_fluo[1].strip(".tif")) - int(list_fluo[0].strip(".tif"))))

        #------------------CORRECT NUMBERING?---------------------
        # do a test if the number of segmented and fluorescent images are numbered correctly.
        # We want that if seg img are 0,1,2,3..10, the fluo images are 0,5,10 instead of 0,1,2
        # we just look if their last numbers are the same
        # if not, start renaming procedure

        # recognize numbering step distance
        list_seg = os.listdir(path_seg_dir)
        list_fluo = os.listdir(path_fluo_dir)
        list_seg.sort(key= lambda x: float(x.strip('.tif')))
        list_fluo.sort(key= lambda x: float(x.strip('.tif')))
        step_size_seg = int(list_seg[1].strip(".tif")) - int(list_seg[0].strip(".tif"))
        step_size_fluo = int(list_fluo[1].strip(".tif")) - int(list_fluo[0].strip(".tif"))


        if step_size_fluo != step_frames:
            print("\nThe fluorescence images have to be renumbered, in order to go in steps of ", str(step_frames) + ".", flush=True)
            print("Renumber the images? Please enter your choice.", flush=True)
            renumber_choice = input("Renumber? (yes/no)")
            if renumber_choice == "yes":
                # renumber images to have the same time numbers
                rename_images.renumber_images_fct( path_fluo_dir, step_frames )
                # display changes
                print("\nChanges applied!")
                list_seg = os.listdir(path_seg_dir)
                list_fluo = os.listdir(path_fluo_dir)
                list_seg.sort(key= lambda x: float(x.strip('.tif')))
                list_fluo.sort(key= lambda x: float(x.strip('.tif')))
                print("Segmentation images now go from: " + str(list_seg[0]) +  " to: " + str(list_seg[-1]) + " in steps of " + str(int(list_seg[1].strip(".tif")) - int(list_seg[0].strip(".tif"))))
                print("Fluorescence images now go from: " + str(list_fluo[0]) +  " to: " + str(list_fluo[-1]) + " in steps of " + str(int(list_fluo[1].strip(".tif")) - int(list_fluo[0].strip(".tif"))))

                print("\nThe analysis can be started, but will just go untill frame",str(np.minimum(int(list_seg[-1].strip(".tif")),int(list_fluo[-1].strip(".tif"))  )))
    
        print("\n\nNo further problems. Ready to start with the analysis...")
        time.sleep(5)



    return(n_seg)


