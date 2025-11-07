import os
import warnings
from tqdm import tqdm
import tifffile
from nd2reader import ND2Reader
from cellpose_omni import models
from cellpose_omni.models import MODEL_NAMES

# Ignore low contrast warnings
warnings.filterwarnings("ignore", message=".*is a low contrast image")

def seg(images, model_name, gpu, output_folder, frame_start=0):
    """
    Perform segmentation on all frames starting from `frame_start` using Cellpose Omni.

    Parameters
    ----------
    images : ND2Reader
        ND2Reader object containing the image sequence.
    model_name : str
        Name of the Cellpose Omni model to use.
    gpu : bool
        Whether to use GPU.
    output_folder : str
        Folder where masks will be saved.
    frame_start : int
        Frame index to start segmentation from.
    """
    print("MODEL_NAMES are ", MODEL_NAMES)
    os.makedirs(output_folder, exist_ok=True)

    # Load the model once
    model = models.CellposeModel(gpu=gpu, model_type=model_name)

    # Parameters for evaluation
    params = {
        'channels': [0, 0],
        'rescale': None,
        'mask_threshold': -2,
        'flow_threshold': 0,
        'transparency': True,
        'omni': True,
        'cluster': True,
        'resample': True,
        'verbose': False,
        'tile': False,
        'niter': None,
        'augment': False,
        'affinity_seg': False,
    }

    # Iterate through frames
    for frame in tqdm(range(frame_start, len(images))):
        img = images[frame]
        mask, _, _ = model.eval(img, **params)
        filename = os.path.join(output_folder, f"{frame}.tif")
        tifffile.imwrite(filename, mask.astype("uint16"))

# Run the segmentation if this script is executed directly
if __name__ == "__main__":
    # `path_im` is the path folder where the ND2 file is located
    path_im = '/Volumes/dock_ssd/DATA/a_postdoc_2024/publication/article_myxo_2024/data_for_submitted_paper/non_formatted_data/fig1/'
    
    # `filename` is the name of the ND2 file without the extension
    filename = 'swarming_movie_4'
    nd2_path = os.path.join(path_im, filename + '.nd2')
    images = ND2Reader(nd2_path)
    
    # output_folder is the folder where the masks will be saved
    output_folder = f'/Volumes/dock_ssd/DATA/a_thesis_2019/codes/python/my_libraries_mx/scripts/output/segmentation_omnipose/new_seg_paper_2024/{filename}/'
    
    # Fill `frame_start` with the index of the first frame to process
    # If you want to start from the first frame, set `frame_start` = 0
    # If you want to skip the first few frames, set `frame_start` to the desired index
    frame_start = 0

    # IMPORTANT: If you are running this script on a Mac, it is recommended to use the `caffeinate` command
    # to prevent the Mac from going to sleep during the execution of the script.
    # This is especially important for long-running scripts to ensure they complete without interruption.
    # You can run the following command in the terminal to execute the script:    
    # caffeinate -i python segmentation_omnipose.py
    seg(images, model_name='bact_phase_omni', gpu=True, output_folder=output_folder, frame_start=frame_start)
