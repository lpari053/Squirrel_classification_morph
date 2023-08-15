"""
Binary Classification using MobileNetV2

This script illustrates the process of training a binary classification model using the MobileNetV2 architecture.
The objective is to classify images as either 'SQUIRREL' or 'STREET' using a dataset containing images of north-east
American squirrels and images from Google Street View.

The dataset is divided into three subsets: train, validation, and test, and consists of 6000 images in the training set.

The 'SQUIRREL' class comprises images of north-east American squirrels in various colors and appearances, while the 'STREET'
class contains images sourced from Google Street View.

MobileNetV2 is utilized as a feature extractor, and a custom dense layer is added for binary classification.
The model is compiled using the Adam optimizer and categorical cross-entropy loss.

Data preprocessing includes image normalization and resizing. Early stopping is employed during training to prevent overfitting
and to achieve optimal generalization.

Author: Laura PARISOT
Date: August 2023
"""


# Set paths and model name
chemin_image = r'database/squirell_vs_street'
nom_MODEL = 'STREET-VS-SQUIRREL/street_squirell_mobilenet.keras'


# Import necessary libraries
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model

# Define image dimensions
image_width, image_height = 224, 224

# Define hyperparameters
batch_size = 512
epochs = 100

# Function to preprocess and create data generators
def dataset(train_dir, validation_dir, test_dir):
    
    
    # Data preprocessing
    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create data generators for training, validation, and test sets
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_width, image_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)
    
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_width, image_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(image_width, image_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)
    
    return train_generator, validation_generator, test_generator

# Function to define and compile the model
def modelie(new_learning_rate=0.01):
    # Load the pre-trained MobileNetV2 model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(image_width, image_height, 3))

    # Create a custom model using the Model API
    x = base_model.output
    x = Flatten()(x)
    x = Dense(252, activation='relu')(x)
    outputs = Dense(2, activation='softmax')(x)

    # Combine the base model and custom layers into the final model
    model = Model(inputs=base_model.input, outputs=outputs)

    # Freeze the layers of the base model to retain pre-trained weights
    for layer in base_model.layers:
        layer.trainable = False
    
    # Compile the model with Adam optimizer and categorical cross-entropy loss
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    
    return model

# Load the dataset using the defined function
train_generator, validation_generator, test_dataset = dataset(
    os.path.join(chemin_image, 'train'),
    os.path.join(chemin_image, 'validation'),
    os.path.join(chemin_image, 'test')
)


# Function for training the model
def training():
    model = modelie()
    # Early stopping callback to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
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
    
    

# Call the training function
training()
