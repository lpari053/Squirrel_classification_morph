import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model


# Image dimensions for preprocessing
image_width, image_height = 224, 224

# Directory paths
chemin_image = r'database\catdog_squirell'
os.makedirs('model/squirell_street_cat_dog_mobilenet',exist_ok=True)
nom_MODEL = 'model/squirell_street_cat_dog_mobilenet/catdog-mobilenet.keras'

# Hyperparameters
batch_size = 16
epochs = 100


def dataset(train_dir, validation_dir, test_dir):
    # Data preprocessing
    
    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    
    # Data generators
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


def modelie():
    # Load the pre-trained MobileNetV2 model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Create a new model using the functional API
    x = base_model.output
    x = Flatten()(x)
    x = Dense(252, activation='relu')(x)
    outputs = Dense(2, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    
    # Freeze the base model's weights
    for layer in base_model.layers:
        layer.trainable = False
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.summary()
    
    return model


train_generator, validation_generator, test_generator = dataset( 
    os.path.join(chemin_image, 'train'),
    os.path.join(chemin_image, 'validation'),
    os.path.join(chemin_image, 'test')
)


def training():
    # Create and compile the model
    model = modelie()
    
    # Define early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
        
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=nom_MODEL,           # Save the best model according to the monitor
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
        callbacks=callbacks)
    
    


training()
