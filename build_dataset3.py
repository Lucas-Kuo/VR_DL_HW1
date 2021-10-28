# import necessary packages
import config
from imutils import paths
import shutil
import os


# grab the paths to all input images in the original input directory and shuffle them
imagePaths = list(paths.list_images(config.ORIG_INPUT_DATASET))


# get the labels of training images
training_labels = []
with open(config.TRAIN_LABEL_PATH) as f:
    for line in f:
        line = line[:-1] # take away the trailing \n
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

# saving images to the 'dataset' directory
print("[INFO] building dataset...")
for class_name in class_file_pairs:
    # put all the images into the training directory since we already have
    # some clue as to when the model is going to overfit the training dataset
    for imageName in class_file_pairs[class_name]:
        save_dir = os.sep.join([config.TRAIN_PATH, class_name])
        source_dir = os.sep.join([config.ORIG_INPUT_DATASET, imageName])

        if not os.path.exists(save_dir):
            print("[INFO] creating '{}' directory".format(save_dir))
            os.makedirs(save_dir)

        p = os.sep.join([save_dir, imageName])
        shutil.copy2(source_dir, p)
