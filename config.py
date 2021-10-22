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

# the amount of validation data will be a percentage of the training data
VAL_SPLIT = 0.1

# initialize the label encoder file path and the output directory to
# where the extracted features (in CSV file format) will be stored
LE_PATH = os.path.sep.join(["output", "le.cpickle"])
BASE_CSV_PATH = "output"
# set the path to the serialized model after training
MODEL_PATH = os.path.sep.join(["output", "model.cpickle"])
