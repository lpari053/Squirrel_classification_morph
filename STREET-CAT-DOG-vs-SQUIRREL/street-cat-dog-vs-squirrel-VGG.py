

"""
Training a Binary Classification Model using VGG

This script demonstrates how to train a binary classification model using the VGG architecture.
The model is designed to classify images as either 'OTHER' or 'SQUIRREL' using a dataset that encompasses
images from various sources, including north-east American squirrels, Google Street View, and images of cats and dogs.

The dataset is divided into three subsets: train, validation, and test, and consists of a total of 5000 images in the training set.

The 'SQUIRREL' class comprises images of north-east American squirrels in different color variations, such as gray, black, and others.
The 'OTHER' class includes images from diverse sources, including Google Street View, as well as images of cats and dogs.

The VGG architecture is employed as a feature extractor, and a custom dense layer is added for binary classification.
The script compiles the model using the Adam optimizer and categorical cross-entropy loss.

Data preprocessing involves image resizing and normalization. Early stopping is employed during training to prevent overfitting
and ensure the model's generalization capability.

Author: Laura PARISOT
Date: August 2023
"""


import tensorflow as tf
import os
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Path to the image dataset
chemin_image = 'database\catdog_squirell'

# Path to save the best model during training

os.makedirs('model/squirell_street_cat_dog',exist_ok=True)
nom_MODEL = 'model/squirell_street_cat_dog/squirell_street_cat_dog.keras'



## Hyperparameter 

# Image size for preprocessing
image_size = (224, 224)

# Batch size for data generator
batch_size = 52

#Number of epoch 
epoch=50

#Learning rate of the Adam optimzer
learning_rate = 0.01


# Data generator for training images
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

val_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

# Training dataset
train_dataset = train_generator.flow_from_directory(
    os.path.join(chemin_image, 'train'),
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

# Validation dataset
validation_dataset = val_generator.flow_from_directory(
    os.path.join(chemin_image, 'validation'),
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

# Test dataset
test_dataset = test_generator.flow_from_directory(
    os.path.join(chemin_image, 'test'),
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

# Load the VGG16 pre-trained model and freeze its layers
base_model = VGG16(weights='imagenet', 
                   include_top=False, 
                   input_shape=(image_size[0], image_size[1], 3))

for layer in base_model.layers:
    layer.trainable = False

# Add custom dense layers for binary classification
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu', name='dense')(x)
x = Dense(128, activation='relu', name='d1')(x)
x = Dense(64, activation='relu', name='d2')(x)
x = Dense(32, activation='relu', name='d3')(x)
predictions = Dense(1, activation='sigmoid', name='fin')(x)

# Create the final model with custom dense layers
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model with binary cross-entropy loss and Adam optimizer

model.compile(loss="binary_crossentropy",      
              optimizer=Adam(learning_rate),      
              metrics=["accuracy"])



# Define early stopping callback to prevent overfitting
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5)   

# Train the model and save the best model based on validation loss
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=nom_MODEL,        
        save_best_only=True,
        monitor="val_loss"),
    early_stopping                
]

# Train the model with the training dataset
model.fit(
    train_dataset,
    epochs=epoch,
    validation_data=validation_dataset,
    callbacks=callbacks
)
