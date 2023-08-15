import pandas as pd
import requests
import os

### Retrieving and Downloading Images from URLs in a CSV

## CSV: The CSV file should contain URLs that lead to images

# Additionally, you can retrieve other information contained in the CSV. In this example, we use:
#   inat_id: the unique identifier to trace the image's origin on INaturaliste
#   morph_class: the color identified by squirrel experts
#   latitude and longitude: the WGS84 coordinates of where the photo was taken

# Before running the script, create a folder to store the downloaded images

# Path to the CSV file and the destination folder for images
csv_file_path = r"path/to/csv"
destination_folder_path = r'path/to/destination-folder-image'

# Create a pandas dataframe containing the complete CSV
dataframe = pd.read_csv(csv_file_path, delimiter=';')

def create_image_folder(csv_file_path, destination_folder_path, dataframe=dataframe):
    
    url_list = dataframe['url'].tolist()                   # Retrieve all URLs as a list
    inat_id_list = dataframe['inat_id'].tolist()           # Retrieve unique IDs per image
    morph_class_list = dataframe['morph_class'].tolist()   # Retrieve squirrel color
    color_black_list = dataframe['color_black'].tolist()
    
    # Iterate through the list containing information for each squirrel, where an index represents an image
    for index in range(len(url_list)):   
        
        url = url_list[index]                   # Image URL
        
        inat_id = inat_id_list[index]           # INaturalist image ID
        
        morph_class = morph_class_list[index]   # Squirrel color on the image
        
        color_black = color_black_list[index]
        
        print(url)
        # Request to retrieve the image
        response = requests.get(url)
        
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            
            # Generate the image name using the retrieved information
            image_name = f"image_{inat_id}_{morph_class}_color_black_{color_black}.jpg"
            
            # Complete destination path with the folder path and the final image name
            destination_path = os.path.join(destination_folder_path, image_name)
            
            # Save the image to the desired folder with the new name
            with open(destination_path, 'wb') as f:
                f.write(response.content)
            print("Image downloaded and saved successfully.  ", index)
        else:
            print("Failed to download the image.  ", index)

# Call the function to create the image folder
create_image_folder(csv_file_path, destination_folder_path, dataframe=dataframe)
