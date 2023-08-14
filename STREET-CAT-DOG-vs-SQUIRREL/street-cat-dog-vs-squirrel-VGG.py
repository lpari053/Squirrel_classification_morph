
#This code permit to train a model using the pre-trained model VGG. 
#This is a binary classification OTHER vs SQUIRREL with 3 dataset train validation test.
#5000 images from the train dataset work well 
#In the database the squirrel folder contains north-east american squirrel gray-black-other
#In the other folder , you can  put images from Google Street View and images of Cat and Dog




import tensorflow as tf
import os
import pathlib
import matplotlib.pyplot as plt
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
