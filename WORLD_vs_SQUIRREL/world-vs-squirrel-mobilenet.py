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

image_width, image_height = 224, 224
chemin_image = r'database/world-vs-squirell'

nom_MODEL = 'WORLD_vs_SQUIRREL/world_vs_squirell_mobilenet.keras'
batch_size = 512
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



new_learning_rate=0.001
# Load pre-trained MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model's weights
base_model.trainable = False

# Create a custom model by adding additional layers on top of the base model

x = base_model.output
x = Flatten()(x)
x = Dense(500, activation='relu')(x)
outputs = Dense(2, activation='softmax')(x)  # Replace 2 with the number of classes in your task

model = Model(inputs=base_model.input, outputs=outputs)

# Compile the model
model.compile(optimizer=SGD(learning_rate=new_learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

train_generator, validation_generator, test_dataset = dataset(
    os.path.join(chemin_image, 'train'),
    os.path.join(chemin_image, 'validation'),
    os.path.join(chemin_image, 'test')
)


early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=5)
    
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=nom_MODEL,    # Save the best model based on the monitor
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


