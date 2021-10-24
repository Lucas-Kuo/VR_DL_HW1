# import necessary packages
import config
from imutils import paths
import shutil
import random
import os

random.seed(42)

# grab the paths to all input images in the original input directory and shuffle them
imagePaths = list(paths.list_images(config.ORIG_INPUT_DATASET))


# get the labels of training images
training_labels = []
with open(config.TRAIN_LABEL_PATH) as f:
    for line in f:
        line = line[:-1]
        training_labels.append(line)

class_file_pairs = {}

for class_name in config.CLASSES:
    class_file_pairs[class_name] = []

for training_label in training_labels:
    filename, class_name = training_label.split()
    class_file_pairs[class_name].append(filename)


for class_name in class_file_pairs:
    random.shuffle(class_file_pairs[class_name])


for class_name in class_file_pairs:
    # on average, there should be 15 images for each class
    # therefore, we split 2 of them for validation and the rest for training
    for imageName in class_file_pairs[class_name][:2]:
        save_dir = os.sep.join([config.VAL_PATH, class_name])
        source_dir = os.sep.join([config.ORIG_INPUT_DATASET, imageName])

        if not os.path.exists(save_dir):
            print("[INFO] creating '{}' directory".format(save_dir))
            os.makedirs(save_dir)
        
        p = os.sep.join([save_dir, imageName])
        shutil.copy2(source_dir, p)

    for imageName in class_file_pairs[class_name][2:]:
        save_dir = os.sep.join([config.TRAIN_PATH, class_name])
        source_dir = os.sep.join([config.ORIG_INPUT_DATASET, imageName])

        if not os.path.exists(save_dir):
            print("[INFO] creating '{}' directory".format(save_dir))
            os.makedirs(save_dir)

        p = os.sep.join([save_dir, imageName])
        shutil.copy2(source_dir, p)
