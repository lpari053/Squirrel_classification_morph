import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

image_width, image_height = 224, 224

chemin_image = r'database\animaux-vs_squirell'


os.makedirs('model/animals_vs_squirell_mobilenet',exist_ok=True)
nom_MODEL = 'model/animals_vs_squirell_mobilenet/animaux_vs_squirell.keras'
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
    
    for layer in base_model.layers:
        if layer in base_model.layers:
            layer.trainable = False
            if 'block_15' in layer.name or 'block_14' in layer.name:
                layer.trainable = True
            else:
                layer.trainable = False
        

    # Create a custom model using the functional API
    x = base_model.output
    x = Flatten()(x)

    # Add dense layers with regularization
    x = Dense(1000 , activation='relu')(x)

    outputs = Dense(2, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    return model


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

    model.evaluate(test_generator)
    test_model(test_generator, nom_MODEL=nom_MODEL)


training()
