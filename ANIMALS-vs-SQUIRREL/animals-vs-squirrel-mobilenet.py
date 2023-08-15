"""
Training a Binary Classification Model using MobileNetV2

This script demonstrates how to train a binary classification model using the MobileNetV2 architecture,
aiming to classify images as either 'OTHER' or 'SQUIRREL' from a diverse dataset. The dataset includes
three subsets for training, validation, and testing, with a total of 5000 images in the training set.

The 'SQUIRREL' class encompasses images of north-east American squirrels with variations in color and appearance,
including gray, black, and other types. On the other hand, the 'OTHER' class includes images of various animals
from the comprehensive 'animals10' dataset available at https://www.kaggle.com/datasets/alessiocorrado99/animals10.

The MobileNetV2 architecture is employed as a pre-trained feature extractor. The script fine-tunes the model on
the custom binary classification task by freezing certain layers and allowing fine-tuning of later blocks. The
model is compiled with the Adam optimizer and categorical cross-entropy loss.

The training process involves data preprocessing, augmentation, and early stopping to ensure optimal model performance.


Author: Laura PARISOT
Date: August 2023
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.models import Model

# Set image dimensions
image_width, image_height = 224, 224

# Define the path to the dataset
chemin_image = r'database\animaux-vs_squirell'

# Create directories to save the model
os.makedirs('model/animals_vs_squirell_mobilenet', exist_ok=True)
nom_MODEL = 'model/animals_vs_squirell_mobilenet/animaux_vs_squirell.keras'

# Set batch size and number of epochs
batch_size = 16
epochs = 100

def dataset(train_dir, validation_dir, test_dir):
    # Data preprocessing
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

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

def modelie(new_learning_rate=0.01):
    # Load the pre-trained MobileNetV2 model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze layers up to certain blocks, allowing fine-tuning on later blocks
    for layer in base_model.layers:
        if layer in base_model.layers:
            layer.trainable = False
            if 'block_15' in layer.name or 'block_14' in layer.name:
                layer.trainable = True
            else:
                layer.trainable = False
        
    # Create a custom model using the functional API
    x = base_model.output
    x = tf.keras.layers.Flatten()(x)

    # Add dense layers with regularization
    x = tf.keras.layers.Dense(1000 , activation='relu')(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Load and preprocess the dataset
train_generator, validation_generator, test_generator = dataset(
    os.path.join(chemin_image, 'train'),
    os.path.join(chemin_image, 'validation'),
    os.path.join(chemin_image, 'test')
)

def test_model(test_generator=test_generator, nom_MODEL=nom_MODEL):
    best_model = keras.models.load_model(nom_MODEL)
    best_model.evaluate(test_generator)

def training():
    model = modelie()

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=5)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=nom_MODEL,
            save_best_only=True,
            monitor="val_accuracy"),
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

    # Evaluate the model on the test set
    model.evaluate(test_generator)
    
    # Test the saved model on the test set
    test_model(test_generator, nom_MODEL=nom_MODEL)

# Start training
training()
