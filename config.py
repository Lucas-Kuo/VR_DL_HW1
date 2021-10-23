# inport necessary packages
import os

# initialize the path to the *original* input directory of images
ORIG_INPUT_DATASET = "training_images"

TRAIN_LABEL_PATH = "training_labels.txt"

# initialize the base path to the *new* directory that will contain
# our images after computing the training and testing split
BASE_PATH = "dataset"

# derive the training, validation, and testing directories
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])

# initialize the list of class label names
CLASSES = []
CLASS_NAMES_FILE = "classes.txt"
with open(CLASS_NAMES_FILE, "r") as f:
    for line in f:
        CLASSES.append(line)

# build class directories for training and validation datasets
for split in (TRAIN_PATH, VAL_PATH):
    for labels in CLASSES:
        label_directory = os.path.sep.join([split, labels])
        if not os.path.exists(label_directory):
            print("[INFO] creating '{}' directory".format(label_directory))
            os.makedirs(label_directory)
 
        
# set the batch size
BATCH_SIZE = 32

# initialize our number of epochs, early stopping patience, initial learning rate
NUM_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 5
INIT_LR = 1e-2

# the amount of validation data will be a percentage of the training data
VAL_SPLIT = 0.1

# initialize the label encoder file path and the output directory to
# where the extracted features (in CSV file format) will be stored
LE_PATH = os.path.sep.join(["output", "le.cpickle"])
BASE_CSV_PATH = "output"
# set the path to the serialized model after training
MODEL_PATH = os.path.sep.join(["output", "model.cpickle"])
