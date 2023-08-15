"""
Training a Binary Classification Model using VGG

This script demonstrates how to train a binary classification model using the VGG architecture,
aiming to classify images as either 'OTHER' or 'SQUIRREL' from a diverse dataset. The dataset includes
three subsets for training, validation, and testing, with a total of 5000 images in the training set.

The 'SQUIRREL' class encompasses images of north-east American squirrels with variations in color and appearance,
including gray, black, and other types. On the other hand, the 'OTHER' class includes images of various animals
from the comprehensive 'animals10' dataset available at https://www.kaggle.com/datasets/alessiocorrado99/animals10.

The VGG architecture is employed as a pre-trained feature extractor. The script fine-tunes the model on
the custom binary classification task by freezing certain layers and allowing fine-tuning of later blocks. The
model is compiled with the Adam optimizer and categorical cross-entropy loss.

The training process involves data preprocessing, augmentation, and early stopping to ensure optimal model performance.


Author: Laura PARISOT
Date: August 2023
"""


import tensorflow as tf
import os

# Dataset path and model path
chemin_image = r'database\animaux-vs_squirell'

os.makedirs('model/animaux_vs_squirell',exist_ok=True)

nom_MODEL = 'model/animaux_vs_squirell/animaux_vs_squirell.keras'

# Image size for preprocessing
image_size = (224, 224)

# Batch size for data generator
batch_size = 16

#Number of epochs
epoch=50

# Data generator for images
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

# Load the VGG19 pre-trained model and freeze its layers
base_model = tf.keras.applications.VGG19(weights='imagenet', 
                                         include_top=False, 
                                         input_shape=(image_size[0], image_size[1], 3))
base_model.trainable = False

# Make some layers trainable for fine-tuning
for layer in base_model.layers[:-5]:
    layer.trainable = True

# Add custom dense layers for binary classification
x = base_model.output
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation='relu', name='d1')(x)
x = tf.keras.layers.Dense(96, activation='relu', name='d2')(x)
x = tf.keras.layers.Dense(64, activation='relu', name='d3')(x)
x = tf.keras.layers.Dense(32, activation='relu', name='d4')(x)

predictions = tf.keras.layers.Dense(1, activation='sigmoid', name='fin')(x)

# Create the final model with custom dense layers
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

# Compile the model with binary cross-entropy loss and SGD optimizer
learning_rate = 0.01
model.compile(loss="binary_crossentropy",
              optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
              metrics=["accuracy"])

# Define early stopping callback to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=10)   

# Train the model and save the best model based on validation accuracy
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=nom_MODEL,
        save_best_only=True,
        monitor="val_accuracy"),
    early_stopping                
]

# Train the model with the training dataset
model.fit(
    train_dataset,
    epochs=epoch,
    validation_data=validation_dataset,
    callbacks=callbacks
)


