# import necessary packages
import config
from imutils import paths
import shutil
import random
import os

# grab the paths to all input images in the original input directory zipped with their respective label and shuffle them
imagePaths = list(paths.list_images(config.ORIG_INPUT_DATASET))
training_labels = []
with open(config.TRAIN_LABEL_PATH) as f:
	for line in f:
		training_labels.append(line)
zippedPaths = list(zip(imagePaths, training_labels))
random.seed(42)
random.shuffle(zippedPaths)

# we'll be using part of the training data for validation
i = int(len(zippedPaths) * config.VAL_SPLIT)
val_zippedPaths = zippedPaths[:i]
train_zippedPaths = zippedPaths[i:]


# define the datasets that we'll be building
datasets = [
	("training", train_zippedPaths, config.TRAIN_PATH),
	("validation", val_zippedPaths, config.VAL_PATH),
]

# loop over the datasets
for (dType, zippedPaths, baseOutput) in datasets:
	# show which data split we are creating
	print("[INFO] 'building {}' split".format(dType))
	
	# if the output base output directory does not exist, create it
	if not os.path.exists(baseOutput):
		print("[INFO] 'creating {}' directory".format(baseOutput))
		os.makedirs(baseOutput)

	# loop over the input image paths
	for zippedPath in zippedPaths:
		# unzip the 
		imagePath, label_name = zip(*zippedPath)
		
		# extract the filename of the input image
		filename = image.split(os.path.sep)[-1]
		
		# build the path to the label directory
		label_directory = os.path.sep.join([baseOutput, label_name])
		
		# if the label output directory does not exist, create it
		if not os.path.exists(label_directory):
			print("[INFO] 'creating {}' directory".format(label_directory))
			os.makedirs(label_directory)
		
		# construct the path to the destination image and then copy
		# the image itself
		p = os.path.sep.join([label_directory, filename])
		shutil.copy2(imagePath, p)



