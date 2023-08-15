# Data Iteration and Organizing Script

# Description: This script is designed to iteratively organize and distribute images from the 'SQUIRELL' and 'OTHER' 
#              directories into separate 'test', 'validation', and 'train' sets. The images are randomly selected based 
#              on the desired proportion for each set. This script helps create a structured dataset for machine learning 
#              tasks, ensuring a balanced distribution of classes across different sets.
# Instructions: Before running the script, ensure that the source directories ('SQUIRELL' and 'OTHER') contain the 
#               image files to be organized. Adjust the 'iterative_dir' variable to set the root directory for the 
#               iterative organization. The script will create subdirectories for 'test', 'validation', and 'train' sets 
#               within the 'iterative_dir' and distribute images accordingly.
# Important Note: This script assumes that the source directories contain sufficient images for the specified 
#                  proportions, and that the desired destination folders do not already exist. Make sure to adjust the 
#                  'num_images' variable and other settings according to your dataset size and requirements.
# Author: Laura PARISOT
# Date: August 2023



import os
import random
import shutil

# Create the iterative folder if it doesn't exist
iterative_dir = r"path/to/destination_folder"
if not os.path.exists(iterative_dir):
    os.makedirs(iterative_dir)

squirrel_dir = r"path/to/squirrel_images"
other_dir = r'path/to/other_images'

folders = ['test', 'validation', 'train']

for folder in folders:
    
    folder_dir = os.path.join(iterative_dir, folder)
    os.makedirs(folder_dir)
    
    squirrel_iteration_dir = os.path.join(folder_dir, "SQUIRELL")
    os.makedirs(squirrel_iteration_dir)

    other_iteration_dir = os.path.join(folder_dir, "OTHER")
    os.makedirs(other_iteration_dir)
    
    num_images = 5000

    if folder == 'test' or folder == 'validation':
        num_images = int(num_images * 0.20)
    else:
        num_images = int(num_images * 0.50)
    
    # Randomly select squirrel images
    squirrel_images = random.sample(os.listdir(squirrel_dir), num_images)
    for image in squirrel_images:
        src = os.path.join(squirrel_dir, image)
        dst = os.path.join(squirrel_iteration_dir, image)
        shutil.move(src, dst)
            
    other_images = random.sample(os.listdir(other_dir), num_images)
    
    for image in other_images:
        src = os.path.join(other_dir, image)
        dst = os.path.join(other_iteration_dir, image)
        shutil.move(src, dst)
