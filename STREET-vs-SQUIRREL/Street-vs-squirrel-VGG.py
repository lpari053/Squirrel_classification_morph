"""
Binary Classification using VGG

This script illustrates the process of training a binary classification model using the VGG architecture.
The objective is to classify images as either 'SQUIRREL' or 'STREET' using a dataset containing images of north-east
American squirrels and images from Google Street View.

The dataset is divided into three subsets: train, validation, and test, and consists of 6000 images in the training set.

The 'SQUIRREL' class comprises images of north-east American squirrels in various colors and appearances, while the 'STREET'
class contains images sourced from Google Street View.

VGG is utilized as a feature extractor, and a custom dense layer is added for binary classification.
The model is compiled using the Adam optimizer and categorical cross-entropy loss.

Data preprocessing includes image normalization and resizing. Early stopping is employed during training to prevent overfitting
and to achieve optimal generalization.

Author: Laura PARISOT
Date: August 2023
"""

# Path to the dataset directory
image_path = 'database/squirell_vs_street'

#Path where thhe best model will be save
nom_MODEL='STREET-VS-SQUIRREL/street_vs_squirell.keras'

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model

# Image dimensions
image_width, image_height = 224, 224
input_shape = (224, 224, 3)

# Batch size and number of epochs for training
batch_size = 54
epochs = 30

def dataset(train_dir, validation_dir, test_dir):
    # Data preprocessing using ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Load training data using flow_from_directory
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_width, image_height),
        batch_size=batch_size,
        class_mode='binary'
    )

    # Load validation data using flow_from_directory
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_width, image_height),
        batch_size=batch_size,
        class_mode='binary'
    )

    # Load test data using flow_from_directory
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(image_width, image_height),
        batch_size=batch_size,
        class_mode='binary'
    )

    return train_generator, validation_generator, test_generator

# Load VGG16 base model with pre-trained weights
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Data augmentation using Sequential API
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.2)
])

# Build the complete model using Functionnal API
x=base_model.output
x=data_augmentation(x)
x=keras.layers.Flatten()(x)
x=keras.layers.Dense(128, activation='relu')(x)
x=keras.layers.Dense(64, activation='relu')(x)
outputs=keras.layers.Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=outputs)


# Print the model summary
model.summary()

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# Load the datasets
train_generator, validation_generator, test_generator = dataset(
    os.path.join(image_path, 'train'),
    os.path.join(image_path, 'validation'),
    os.path.join(image_path, 'test')
)

# Set up early stopping
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5)

# Set up model checkpoint to save the best model
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=nom_MODEL,
        save_best_only=True,
        monitor="val_accuracy"
    ),
    early_stopping
]

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.n // batch_size,
    callbacks=callbacks
)


