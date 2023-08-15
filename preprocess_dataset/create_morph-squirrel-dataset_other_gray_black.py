# Image Dataset Organization Script
# Author: [Your Name]
# Date: [Current Date]
# Description: This script is designed to iteratively organize and distribute images from a source directory containing 
#              mixed-color squirrel images ('black', 'gray', 'other') into separate 'test', 'validation', and 'train' 
#              sets. The images are randomly selected based on the desired proportion for each set and their respective 
#              color categories. This script helps create a structured dataset for machine learning tasks, ensuring a 
#              balanced distribution of classes and colors across different sets.
# Instructions: Before running the script, ensure that the source directory ('squirrel_dir') contains the mixed-color 
#               squirrel image files to be organized. Adjust the 'iterative_dir' variable to set the root directory for 
#               the iterative organization. The script will create subdirectories for 'test', 'validation', and 'train' 
#               sets within the 'iterative_dir' and distribute images based on their color categories.
# Important Note: This script assumes that the source directory contains sufficient images for the specified 
#                  proportions, and that the desired destination folders do not already exist. Make sure to adjust the 
#                  'num_images' variable and other settings according to your dataset size and requirements.



import os
import random
import shutil

# Set the root directory for the organized dataset
iterative_dir = r"path/to/destination_folder"
if not os.path.exists(iterative_dir):
    os.makedirs(iterative_dir)

# Directory containing mixed-color squirrel images
squirrel_dir = r"path/to/squirrel_images_mix_color"

# List of subdirectories for 'test', 'validation', and 'train' sets
dossier = ['test', 'validation', 'train']

# Iterate through the sets
for jeu in dossier:
    
    # Create directories for each set
    jeu_dir = os.path.join(iterative_dir, f"{jeu}")
    os.makedirs(jeu_dir)
    
    black_iteration_dir = os.path.join(jeu_dir, "black")
    os.makedirs(black_iteration_dir)
    
    gray_iteration_dir = os.path.join(jeu_dir, "gray")
    os.makedirs(gray_iteration_dir)

    other_iteration_dir = os.path.join(jeu_dir, "other")
    os.makedirs(other_iteration_dir)
    
    # Count the total number of image files in the source directory
    num_image = len([f for f in os.listdir(squirrel_dir) if os.path.isfile(os.path.join(squirrel_dir, f))])
    
    num_image = int(num_image)
    
    # Calculate the number of images for the current set based on proportions
    if jeu == 'test' or jeu == 'validation':
        num_images = int(num_image * 0.20)
    else:
        num_images = int(num_image * 0.50)
    
    # Randomly select images for the current set
    squi_images = random.sample(os.listdir(squirrel_dir), num_images)
    for image in squi_images:
        src = os.path.join(squirrel_dir, image)
        image_name = os.path.basename(src).split('_')
        
        # Determine the color category of the image
        color_category = image_name[2]
        
        # Choose the appropriate destination directory based on color category
        if color_category == 'gray':
            dst = os.path.join(gray_iteration_dir, image)
        elif color_category == 'black':
            dst = os.path.join(black_iteration_dir, image)
        elif color_category == 'other':
            dst = os.path.join(other_iteration_dir, image)
        
        # Copy the image to the destination directory
        shutil.copy(src, dst)

            








