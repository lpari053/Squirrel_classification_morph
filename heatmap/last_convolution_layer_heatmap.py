import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Conv2D
from tensorflow.keras.preprocessing import image
# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from keras.applications.densenet import preprocess_input
import os
import matplotlib

nom_MODEL='path/to/model'


# Load the model
model = tf.keras.models.load_model(nom_MODEL)

# Set the image size
img_size = (224, 224)

# Find the last Conv2D layer in the model
last_conv_layer = None
for layer in reversed(model.layers):
    if isinstance(layer, Conv2D):
        last_conv_layer = layer
        break

if last_conv_layer is not None:
    last_conv_layer_name = last_conv_layer.name

def get_img_array(img_path, size):
    # Load the image and convert it to an array
    img = keras.utils.load_img(img_path, target_size=size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Create a gradient model to get the output of the last Conv2D layer and the model predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        # Get the output of the last Conv2D layer and the model predictions
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Calculate the gradients
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Calculate the mean gradient
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]

    # Calculate the heatmap
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path, alpha=0.4):
    # Load the original image
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = matplotlib.colormaps.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on the original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    if cam_path is not None:
        superimposed_img.save(cam_path)
    
    return superimposed_img

def get_heatmap(img_path,cam_path):
    img_array = get_img_array(img_path, size=img_size)

    # Preprocess the image for the model
    images = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(images)
    x = np.expand_dims(x, axis=0)
    img_array = preprocess_input(x)

    # Deactivate the final activation to get the intermediate layer output
    model.layers[-1].activation = None

    # Generate the heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)


    
    if cam_path is not None:
        save_and_display_gradcam(img_path, heatmap, cam_path)
        
    else:
        
        superpose_img=save_and_display_gradcam(img_path, heatmap, cam_path)
        
        return superpose_img
    

    
get_heatmap('heatmap/MORPH/image_input.jpg','heatmap/MORPH/heatmap.jpg')