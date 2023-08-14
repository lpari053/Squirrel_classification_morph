#This code permit to train a model without using a pre-trained model. 
#This is a binary classification OTHER vs SQUIRREL with 3 dataset train validation test. 
#This code work well with the repartition 25000 pictures to train an 10000 pictures for the validation and test dataset
#In the database the other folder contains images comming from https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset
#In the database the squirrel folder contains north-east american squirrel gray-black-other



import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16,VGG19
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.callbacks import EarlyStopping
image_width, image_height = 224, 224


chemin_image=r'database/world-vs-squirell'
os.makedirs('model/world-vs-squirell',exist_ok=True)
nom_MODEL='WORLD_vs_SQUIRREL/world_vs_squirell.keras'
# Define hyperparameter
batch_size = 512
epochs = 50



def dataset(train_dir,validation_dir,test_dir):
    
    # Data processing
    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create data generators for training, validation, and test sets
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_width, image_height),
        batch_size=batch_size,
        class_mode='binary')
    
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_width, image_height),
        batch_size=batch_size,
        class_mode='binary')
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(image_width, image_height),
        batch_size=batch_size,
        class_mode='binary')
    
    return(train_generator,validation_generator,test_generator
           )


new_learning_rate=0.01

data_augmentation = keras.Sequential(
    [
    keras.layers.Input((image_width, image_height,3)),
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.2),
    ]
)
    
inputs=keras.layers.Input((image_width, image_height,3)),

x=data_augmentation(inputs)

x= keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same',input_shape=(image_width, image_height, 3))(x)
x= keras.layers.MaxPooling2D((2, 2))(x)

x= keras.layers.Conv2D(16, (3, 3), padding='same',activation='relu')(x)
x= keras.layers.MaxPooling2D((2, 2))(x)
x= keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
x= keras.layers.MaxPooling2D((2, 2))(x)
x= keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x= keras.layers.MaxPooling2D((2, 2))(x)

x= keras.layers.Conv2D(128, (3, 3), padding='same',activation='relu')(x)
x= keras.layers.MaxPooling2D((2, 2))(x)
x= keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
x= keras.layers.MaxPooling2D((2, 2))(x)
x= keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
x= keras.layers.MaxPooling2D((2, 2))(x)


x= keras.layers.Flatten()(x)

x= keras.layers.Dense(256, activation='relu')(x)

x= keras.layers.Dense(128, activation='relu')(x)

x= keras.layers.Dense(64, activation='relu')(x)

x= keras.layers.Dense(32, activation='relu')(x)

x= keras.layers.Dense(16, activation='relu')(x)

x= keras.layers.Dense(8, activation='relu')(x)

outputs= keras.layers.Dense(1, activation='sigmoid')(x)


model = keras.Model(inputs=inputs, outputs=outputs)
    

# Compile the modele
model.compile(optimizer="adam",
                loss='binary_crossentropy',
                metrics=['accuracy'])



# Load and preprocess datasets
train_generator,validation_generator,test_dataset=dataset( 
    os.path.join(chemin_image,'train'),
    os.path.join(chemin_image,'validation'),
    os.path.join(chemin_image,'test')
    )

# Define early stopping callback to prevent overfitting
early_stopping = EarlyStopping(monitor='accuracy', patience=5)   
    
    
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=nom_MODEL,           #enregistrement du meilleure modele selon le monitor
        save_best_only=True,
        monitor="val_accuracy"),early_stopping               
]

    # Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.n // batch_size,
    callbacks=callbacks)
    


# Load the best model from the checkpoint for evaluation
best_model = keras.models.load_model(nom_MODEL)

# Load and preprocess datasets
train_generator,validation_generator,test_dataset=dataset( 
    os.path.join(chemin_image,'train'),
    os.path.join(chemin_image,'validation'),
    os.path.join(chemin_image,'test')
    )

# Evaluate the model on the test data
train_acc = best_model.evaluate(train_generator)[1]
print(f"Train accuracy: {train_acc:.5f} \n")

val_acc = best_model.evaluate(validation_generator)[1]
print(f"Val accuracy: {val_acc:.5f} \n")

test_acc = best_model.evaluate(test_dataset)[1]
print(f"Test accuracy: {test_acc:.5f} \n")
