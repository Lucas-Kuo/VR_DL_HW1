# inport necessary packages
import os

# initialize the path to the *original* input directory of images


# initialize the base path to the *new* directory that will contain
# our images after computing the training and testing split
BASE_PATH = "dataset"

# define the names of the training, testing, and validation
# directories
TRAIN = "training"
TEST = "evaluation"
VAL = "validation"

# initialize the list of class label names
CLASSES = []
CLASS_NAMES_FILE = "classes.txt"
with open(CLASS_NAMES_FILE, "r") as f:
	for line in f:
		CLASSES.append(line)

# set the batch size
BATCH_SIZE = 32
