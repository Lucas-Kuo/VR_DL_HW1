# import necessary packages
import config
from imutils import paths
import shutil
import random
import os

random.seed(42)

# grab the paths to all input images
# in the original input directory and shuffle them
imagePaths = list(paths.list_images(config.ORIG_INPUT_DATASET))


# get the labels of training images
training_labels = []
with open(config.TRAIN_LABEL_PATH) as f:
    for line in f:
        line = line[:-1]  # take away the trailing \n
        training_labels.append(line)

# key: label/class name
# value: a list of the image file names belonging to this class
class_file_pairs = {}

# initialize the dictionary
for class_name in config.CLASSES:
    class_file_pairs[class_name] = []

# extract the filenames and class names
# and put them in the right list
for training_label in training_labels:
    filename, class_name = training_label.split()
    class_file_pairs[class_name].append(filename)

# shuffle each list
for class_name in class_file_pairs:
    random.shuffle(class_file_pairs[class_name])

# saving images to the 'dataset' directory
print("[INFO] building dataset...")
for class_name in class_file_pairs:
    # on average, there should be 15 images for each class
    # we split 2 of them(~15%) for validation and the rest for training
    for imageName in class_file_pairs[class_name][:2]:
        # directories for the image source and destination
        save_dir = os.sep.join([config.VAL_PATH, class_name])
        source_dir = os.sep.join([config.ORIG_INPUT_DATASET, imageName])

        # in case the config.py didn't build the directory, check again
        if not os.path.exists(save_dir):
            print("[INFO] creating '{}' directory".format(save_dir))
            os.makedirs(save_dir)

        # copy the image
        p = os.sep.join([save_dir, imageName])
        shutil.copy2(source_dir, p)

    # do the same thing above except it's for training dataset this time
    for imageName in class_file_pairs[class_name][2:]:
        save_dir = os.sep.join([config.TRAIN_PATH, class_name])
        source_dir = os.sep.join([config.ORIG_INPUT_DATASET, imageName])

        if not os.path.exists(save_dir):
            print("[INFO] creating '{}' directory".format(save_dir))
            os.makedirs(save_dir)

        p = os.sep.join([save_dir, imageName])
        shutil.copy2(source_dir, p)
