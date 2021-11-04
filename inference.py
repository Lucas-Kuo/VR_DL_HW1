import numpy as np
import os
import tensorflow as tf
import gdown
from tensorflow.keras.preprocessing import image_dataset_from_directory

import config


def load_images(imagePath):
    # pass in the path of testing dataset
    # since we're making inferences, there will be no labels
    return image_dataset_from_directory(
        imagePath, labels=None, shuffle=False, label_mode=None,
        batch_size=config.BATCH_SIZE, image_size=config.IMG_SIZE)


# load testing dataset
test_dataset = load_images(config.TEST_PATH)

# download the trained model from google drive
gdown.download(config.MODEL_URL, config.MODEL_NAME, quiet=False)
model = tf.keras.models.load_model(config.MODEL_NAME)

# make our inference
print("[INFO] making inferences...")
predictions = model.predict(test_dataset)

# predictions are 200-dimentional vectors
# we only keep the largest ones' indices
output = list(np.argmax(predictions, axis=1))

# map the image file names to their respective result
result = {}
N = len(output)
for i in range(N):
    name = test_dataset.file_paths[i][-8:]
    label = config.CLASSES[output[i]]
    result[name] = label

# the retrieve the order of sample answer
evaluation_filenames = []
with open(config.SAMPLE_ANSWER_PATH, "r") as f:
    for line in f:
        # the file name and class name is split
        # by a space
        filename = line.split()[0]
        evaluation_filenames.append(filename)

# output the result
print("[INFO] outputing results...")
with open("answer.txt", "w") as f:
    for filename in evaluation_filenames:
        s = filename + ' ' + result[filename] + '\n'
        f.write(s)
print("[INFO] inference completed")
