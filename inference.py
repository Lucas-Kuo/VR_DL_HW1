import numpy as np
import os
import tensorflow as tf
from tensorflow.data import AUTOTUNE
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation
from tensorflow.keras.applications.efficientnet import preprocess_input
from imutils import paths

import config
from train_model import build_model, weight_path


# for unlabeled images, i.e. testing/evaluation dataset
def load_images(imagePath):
	# read the image from disk, decode it, convert the data type to
	# floating point, and resize it
	image = tf.io.read_file(imagePath)
	image = tf.image.decode_png(image, channels=3)
	image = tf.image.convert_image_dtype(image, dtype=tf.float32)
	image = tf.image.resize(image, config.IMAGE_SIZE)
    
	# no label is available for testing dataset
	label = None
	
	# return the image and the label
	return (image, label)

# grab all testing dataset image paths
testPaths = list(paths.list_images(config.TEST_PATH))

# build the testing dataset and data input pipeline
test_dataset = tf.data.Dataset.from_tensor_slices(testPaths)
test_dataset = (test_dataset
	.map(load_images, num_parallel_calls=AUTOTUNE)
	.cache()
	.batch(config.BATCH_SIZE)
	.prefetch(AUTOTUNE)
)

# build our model and retrieve the weights we saved
model = build_model()
model.load_weights(weight_path)

# make our inference
predictions = model.predict(test_dataset)

# to be modified
largest_confidence = list(np.argmax(predictions, axis = 1))
result = {}
N = len(largest_confidence)
for i in range(N):
  name = test_dataset.file_paths[i][-8:] # the file path has the format: .../.../xxxx.jpg
  label = CLASSES[output[i]]
  result[name] = label
