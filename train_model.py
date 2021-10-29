# inport necessary packages
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.data import AUTOTUNE
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation, RandomZoom, RandomFlip
from tensorflow.keras.applications.efficientnet import preprocess_input
from imutils import paths

import config

# for labeled images
def load_images(imagePath):
	# read the image from disk, decode it, convert the data type to
	# floating point, and resize it
	image = tf.io.read_file(imagePath)
	image = tf.image.decode_png(image, channels=3)
	image = tf.image.convert_image_dtype(image, dtype=tf.float32)
	image = tf.image.resize(image, config.IMG_SIZE)
    
	# parse the class label from the file path
	label = (tf.strings.split(imagePath, os.path.sep).numpy())[-2].decode('UTF-8')
	label = tf.strings.to_number(label, tf.int32)
	label = label.numpy()
	label_vector = np.zeros((1,200))
	label_vector[0][label-1] = 1
	
	# return the image and the label
	return (image, label_vector)

# build the model we'll be using
def build_model():
    # the base model I use this time is EfficientNet B7
    base_model = tf.keras.applications.efficientnet.EfficientNetB7(input_shape=config.IMG_SHAPE,
                                                                   include_top=False,
                                                                   weights='imagenet')
    # to perform transfer learning, we disable the
    # trainability of the base model layers
    base_model.trainable = False
    
    # the data augmentation I adopt includes horizontal flipping(Line 29),
    # rotation and contrast adjustment
#     data_augmentation = tf.keras.Sequential([
#         RandomRotation(0.2),
#         tf.keras.layers.RandomContrast(0.5, seed=None)
#     ])
    
    # constructing the model
    inputs = tf.keras.Input(shape=config.IMG_SHAPE)
#     x = data_augmentation(inputs)
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.AveragePooling2D(pool_size=(19, 19))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(len(config.CLASSES), activation="softmax", activity_regularizer=tf.keras.regularizers.L2(0.1))(x)
    model = tf.keras.Model(inputs, outputs)
    
    return model

# plot out the training result, including loss(cross entropy) and accuracy
def plot_graph(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,6.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

# grab all the training and validation dataset image paths
trainPaths = list(paths.list_images(config.TRAIN_PATH))
valPaths = list(paths.list_images(config.VAL_PATH))

# the data augmentation I adopt includes horizontal flipping,
# rotation , zomming in and contrast adjustment
trainAugmentation = tf.keras.Sequential([
	RandomRotation(0.3),
	tf.keras.layers.RandomContrast(0.5, seed=None),
	RandomZoom(
		height_factor=(-0.05, -0.15),
		width_factor=(-0.05, -0.15)),
	RandomFlip("horizontal")
])

# build the training dataset and data input pipeline
train_dataset = tf.data.Dataset.from_tensor_slices(trainPaths)
train_dataset = (train_dataset
	.shuffle(len(trainPaths))
	.map(load_images, num_parallel_calls=AUTOTUNE)
	.map(lambda x, y: (trainAugmentation(x), y), num_parallel_calls=AUTOTUNE)
	.cache()
	.batch(config.BATCH_SIZE)
	.prefetch(AUTOTUNE)
)

# build the validation dataset and data input pipeline
val_dataset = tf.data.Dataset.from_tensor_slices(valPaths)
val_dataset = (val_dataset
	.map(load_images, num_parallel_calls=AUTOTUNE)
	.cache()
	.batch(config.BATCH_SIZE)
	.prefetch(AUTOTUNE)
)

# configuring the model
model = build_model()
opt = tf.keras.optimizers.Adam(learning_rate=config.INIT_LR)
model.compile(optimizer=opt,
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

# initialize an early stopping callback to prevent the model from
# overfitting
es = EarlyStopping(
	monitor="val_loss",
	patience=config.EARLY_STOPPING_PATIENCE,
	restore_best_weights=True)

# fit the model
history = model.fit(
	x=train_dataset,
	validation_data=val_dataset,
	epochs=config.NUM_EPOCHS,
	callbacks=[es],
	verbose=1)

# show the result
plot_graph(history)

# save the weight
weight_path = config.WEIGHT_PATH + 'E_netB7_weight'
model.save_weights(weight_path)
