import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  
    
chemin_image=r'database/world-vs-squirell'
image_width, image_height=224,224
batch_size=512
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
        shuffle=True
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_width, image_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(image_width, image_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    return train_generator, validation_generator, test_generator
    
    
train_generator, validation_generator, test_dataset = dataset(
    os.path.join(chemin_image, 'train'),
    os.path.join(chemin_image, 'validation'),
    os.path.join(chemin_image, 'test')
)

nom_MODEL = 'model/world_vs_squirell_mobilenet/world_vs_squirell_mobilenet.keras'

# Load the best model from the checkpoint for evaluation
model = keras.models.load_model(nom_MODEL)

# Evaluate the model on the test data
train_acc = model.evaluate(train_generator)[1]
print(f"Train accuracy: {train_acc:.5f} \n")

val_acc = model.evaluate(validation_generator)[1]
print(f"Val accuracy: {val_acc:.5f} \n")

test_acc = model.evaluate(test_dataset)[1]
print(f"Test accuracy: {test_acc:.5f} \n")