# %%
import omnipose
import cellpose
import numpy as np
from cellpose import models, core
from pathlib import Path
import os
from cellpose import io
import skimage.io as skio
from cellpose import io, transforms
from omnipose.utils import normalize99
from pathlib import Path
from cellpose import models
from cellpose.models import MODEL_NAMES
from nd2reader import ND2Reader
from tqdm import tqdm
MODEL_NAMES

# Folders names
input_folder = ""
filename = "rippling_movie_1.nd2"
# filename = "swarming_movie_1.nd2"
output_folder = ""

# Load images
im = ND2Reader(input_folder+filename)

# Choose omnipose parameters
model_name = 'CP'
model = models.CellposeModel(gpu=True, model_type=model_name, diam_mean=30.)
nimg = len(im)
print('number of images:', nimg)

mask_threshold = -1
rescale=None # give this a number if you need to upscale or downscale your images
flow_threshold = 0 # default is .4, but only needed if there are spurious masks to clean up; slows down output
resample = False #whether or not to run dynamics on rescaled grid or original grid

start = 0

for i in tqdm(range(start, nimg)):
    mask, flow, style = model.eval(im[i],rescale=rescale, flow_threshold=flow_threshold,resample=resample)
    imname = str(i)+".tif"
    skio.imsave(output_folder+imname, mask)

print("end mask")