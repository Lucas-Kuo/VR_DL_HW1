# inport necessary packages
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import tensorflow as tf
from tensorflow.data import AUTOTUNE
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation, RandomZoom, RandomFlip
from tensorflow.keras.applications.efficientnet import preprocess_input
from imutils import paths

import config

# for labeled images
def load_images(imagePath):
    # pass in the image directory and set the class names
	return image_dataset_from_directory(imagePath, shuffle=True, class_names=config.CLASSES, label_mode="categorical",
                                        batch_size=config.BATCH_SIZE, image_size=config.IMG_SIZE)

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
    plt.ylim([0,3.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


# build the training dataset
train_dataset = load_images(config.TRAIN_PATH)

# build the validation dataset
val_dataset = load_images(config.VAL_PATH)

# configuring the model
print("[INFO] building model...")

# the base model I use this time is EfficientNet v2 M
base_model = tf.keras.models.load_model(config.BASE_MODEL_NAME)

# to perform transfer learning, we disable the
# trainability of the base model layers first
base_model.trainable = False

# the data augmentation I adopt includes horizontal flipping,
# rotation and contrast adjustment
 data_augmentation = tf.keras.Sequential([
     tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
     RandomRotation(0.2),
     tf.keras.layers.RandomContrast(0.5, seed=None)
 ])

# constructing the model
inputs = tf.keras.Input(shape=config.IMG_SHAPE)
x = data_augmentation(inputs)
x = base_model(x, training=False) # set training to False to avoid BN training
x = tf.keras.layers.AveragePooling2D(pool_size=(15, 15))(x)
x = tf.keras.layers.Flatten()(x)
outputs = tf.keras.layers.Dense(len(config.CLASSES), activation="softmax", activity_regularizer=tf.keras.regularizers.L2(0.1))(x)
model = tf.keras.Model(inputs, outputs)

opt = tf.keras.optimizers.Adam(learning_rate=config.INIT_LR)
model.compile(optimizer=opt,
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

# train the model
print("[INFO] training model...")
history = model.fit(
	x=train_dataset,
	validation_data=val_dataset,
	epochs=config.NUM_EPOCHS,,
	verbose=1)

# show the result
print("[INFO] first part of training result:")
plot_graph(history)

# dump the result
print("[INFO] saving training history1...")
json.dump(history.history, open(config.HISTORY1, 'w'))

# save the model
print("[INFO] saving model1...")
model.save(config.SAVE_MODEL1)

# -------------------- Fine Tuning -------------------- #

# Here's a detail: a layer is trainable
# if and only if the whole model's trainable attribute is True
# and the layer's trainable attribute is True
# So Line 135 is essential. Doing it the opposite way won't work
base_model.trainable = True
fine_tune_at = 755
for layer in model.layers[:fine_tune_at]:
    layer.trainable = False
print(f"[INFO] {base_model.trainable_variables} layers in the base model is now trainable")

# configuring fine tuning setup
opt = tf.keras.optimizers.RMSprop(learning_rate=config.INIT_LR/10) # set a smaller LR when fine tuning
model.compile(optimizer=opt,
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

# fine tuning
print("[INFO] fine tuning...")
total_epochs = config.NUM_EPOCHS * 2
history2 = model1.fit(train_dataset,
                     epochs=total_epochs,
                     initial_epoch=history.epoch[-1],
                     validation_data=validation_dataset)

# show the result
print("[INFO] second part of training result:")
plot_graph(history2)

# dump the result
print("[INFO] saving training history2...")
json.dump(history2.history, open(config.HISTORY2, 'w'))

# save the model
print("[INFO] saving model1...")
model.save(config.SAVE_MODEL2)

print("[INFO] training completed")
