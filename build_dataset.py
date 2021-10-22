# import necessary packages
import config
from imutils import paths
import shutil
import os

# loop over the data splits
for split in (config.TRAIN, config.VAL):
	# grab all image paths in the current split
	print("[INFO] processing '{} split'...".format(split))
	p = os.path.sep.join([config.ORIG_INPUT_DATASET, split])
	imagePaths = list(paths.list_images(p))
