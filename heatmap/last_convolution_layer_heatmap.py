"""
Grad-CAM Image Visualization

This script demonstrates how to use the Grad-CAM (Gradient-weighted Class Activation Mapping)
technique to generate visualizations that highlight the regions of an input image that contribute
the most to the predictions made by convolutional neural network.

The script loads a neural network model already train, processes an input image, and generates a Grad-CAM
heatmap to visualize the regions of interest. The heatmap is then colorized and superimposed on the
original image for intuitive visualization. The resulting visualization can be saved and displayed.

Grad-CAM is a powerful tool for understanding the decision-making process of a neural network and
provides insights into which parts of an image are most influential in determining its predictions.

Note: This script assumes the availability of a pre-trained neural network model and input images.

Author: Laura PARISOT
Date: August 2023
"""

#Importation packages needed
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D
from tensorflow.keras.preprocessing import image
# Display
from keras.applications.densenet import preprocess_input
import matplotlib

# Define the path to the saved model
nom_MODEL = 'path/to/model'

# Load the pre-trained model
model = tf.keras.models.load_model(nom_MODEL)

# Set the image size for processing
img_size = (224, 224)

# Find the last Conv2D layer in the model
last_conv_layer = None
for layer in reversed(model.layers):
    if isinstance(layer, Conv2D):
        last_conv_layer = layer
        break

# Store the name of the last Conv2D layer for later use
if last_conv_layer is not None:
    last_conv_layer_name = last_conv_layer.name

# Function to convert an image file into a processed array
def get_img_array(img_path, size):
    # Load the image and convert it to an array
    img = keras.utils.load_img(img_path, target_size=size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

# Function to generate a Grad-CAM heatmap
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Create a gradient model to get the output of the last Conv2D layer and the model predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Use GradientTape to record operations for gradient computation
    with tf.GradientTape() as tape:
        # Get the output of the last Conv2D layer and the model predictions
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Calculate the gradients of the chosen class with respect to the output of the last Conv2D layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Calculate the mean gradient over all spatial positions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Extract the output of the last Conv2D layer for the given input
    last_conv_layer_output = last_conv_layer_output[0]

    # Generate the heatmap by performing element-wise multiplication
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap to ensure values are within [0, 1]
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Function to save and display the Grad-CAM visualization on the original image
def save_and_display_gradcam(img_path, heatmap, cam_path, alpha=0.4):
    # Load the original image
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

    # Normalize the heatmap values and convert to uint8 for visualization
    heatmap = np.uint8(255 * heatmap)

    # Use a color map (jet) to enhance the heatmap visualization
    jet = matplotlib.colormaps.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Resize the colorized heatmap to match the original image dimensions
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on the original image using an alpha factor
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image if a destination path is provided
    if cam_path is not None:
        superimposed_img.save(cam_path)
    
    return superimposed_img

# Function to generate and visualize the Grad-CAM heatmap for an image
def get_heatmap(img_path, cam_path):
    # Convert the image to an array and preprocess it for the model
    img_array = get_img_array(img_path, size=img_size)
    images = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(images)
    x = np.expand_dims(x, axis=0)
    img_array = preprocess_input(x)

    # Deactivate the final activation to get the intermediate layer output
    model.layers[-1].activation = None

    # Generate the Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    # Save and display the Grad-CAM visualization
    if cam_path is not None:
        save_and_display_gradcam(img_path, heatmap, cam_path)
    else:
        superpose_img = save_and_display_gradcam(img_path, heatmap, cam_path)
        return superpose_img

# Example usage
get_heatmap('image_input.jpg', 'heatmap.jpg')
