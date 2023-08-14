import tensorflow as tf
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import KFold
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import preprocess_input

# Define parameters
num_classes = 3
input_shape = (224, 224, 3)
num_folds = 3
batch_size = 16
epochs = 10

# Prepare the data directory
image_path = r'database/morph'      #Path to dataset
model_name = 'model/model_classification_morph/model_train_model/model'

# Load the pre-trained MobileNet model (weights on ImageNet without the classification layer)
base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=input_shape)

# Set layers to be trainable based on their names
for layer in base_model.layers:
    if layer in base_model.layers:
        layer.trainable = False
        if 'conv5' in layer.name or 'conv4' in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False


# Build the model architecture on top of the pre-trained base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1000, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])

# Define the data generator with data augmentation
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

# Create a validation data generator with data preprocessing
validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Create a test data generator with data preprocessing
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Create a KFold object for cross-validation
kfold = KFold(n_splits=num_folds, shuffle=True)

# Iterate over the cross-validation folds
fold = 1
for train_indices, val_indices in kfold.split(range(num_classes)):

    print(f"Training Fold {fold}...")
    print("")

    # Create directory flows for training and validation data
    train_data = data_generator.flow_from_directory(
        os.path.join(image_path, 'train'),
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

    val_data = validation_datagen.flow_from_directory(
        os.path.join(image_path, 'validation'),
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f'{model_name}_{fold}.keras',   # Save the best model based on validation accuracy
            save_best_only=True,
            monitor="val_accuracy")
    ]

    # Train the model on the training data
    model.fit(
        train_data,
        epochs=epochs,
        validation_data=val_data,
        callbacks=callbacks
    )

    # Evaluate the model on the test data
    test_data = test_datagen.flow_from_directory(
        os.path.join(image_path, 'test'),
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

    # Load the best model from the checkpoint and evaluate on the test data
    best_model = tf.keras.models.load_model(f'{model_name}_{fold}.keras')
    best_model.evaluate(test_data)

    fold += 1
