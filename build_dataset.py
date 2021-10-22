# import necessary packages
import config
from imutils import paths
import shutil
import random
import os

# grab the paths to all input images in the original input directory zipped with their respective label and shuffle them
imagePaths = list(paths.list_images(config.ORIG_INPUT_DATASET))
training_labels = []
with open()
random.seed(42)
random.shuffle(imagePaths)


# loop over the data splits
for split in (config.TRAIN, config.VAL):
	# grab all image paths in the current split
	print("[INFO] processing '{} split'...".format(split))
	p = os.path.sep.join([config.ORIG_INPUT_DATASET, split])
	imagePaths = list(paths.list_images(p))
