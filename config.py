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
TEST_PATH = os.path.sep.join([BASE_PATH, "evaluation"])

# for getting the order of images to output answer
SAMPLE_ANSWER_PATH = os.path.sep.join(["self_utils", "sample_answer.txt"])

# initialize the list of class label names
CLASSES = []
CLASS_NAMES_FILE = "classes.txt"
with open(CLASS_NAMES_FILE, "r") as f:
    for line in f:
        line = line[:-1]  # remove the trailing \n
        CLASSES.append(line)

# build class directories for training and validation datasets
for split in (TRAIN_PATH, VAL_PATH):
    for labels in CLASSES:
        label_directory = os.path.sep.join([split, labels])
        if not os.path.exists(label_directory):
            print("[INFO] creating '{}' directory".format(label_directory))
            os.makedirs(label_directory)
 

# set the image size and shape
IMG_SIZE = (480,480)
IMG_SHAPE = IMG_SIZE + (3,)

# set the batch size
BATCH_SIZE = 32

# initialize our number of epochs, early stopping patience, initial learning rate
NUM_EPOCHS = 30
INIT_LR = 1e-3

# set the directory to save our trained model's weight
WEIGHT_PATH = "weightings"
if not os.path.exists(WEIGHT_PATH):
    print("[INFO] creating '{}' directory".format(WEIGHT_PATH))
    os.makedirs(WEIGHT_PATH)

# the link and name of the final model .h5 file
MODEL_URL = "https://drive.google.com/u/0/uc?id=1-wnA207-0fuqKZMPL1r5lT_w9KqK2oKl&export=download"
MODEL_NAME = "Enet_v2_finetuned_final.h5"

