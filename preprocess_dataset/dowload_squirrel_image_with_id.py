import requests
import shutil

def get_large_image_url(observation_id):
    url = f'https://api.inaturalist.org/v1/observations/{observation_id}'
    response = requests.get(url)
    data = response.json()

    # Get the image URL from the response
    image_url = data['results'][0]['photos'][0]['url']

    # Replace the size identifier that is default 'square' 75x75 to large image
    large_image_url = image_url.replace('square','large') # or other appropriate replacement

    return large_image_url

def download_image(image_url, file_name):
    response = requests.get(image_url, stream=True)
    with open(file_name, 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    print(f"Downloaded {file_name}")

# Change for observation id in the inat.id field in the csv file
path_save="path/to/save/images"


csv_file_path = r"path/to/csv"

# Create a pandas dataframe containing the complete CSV
dataframe = pd.read_csv(csv_file_path, delimiter=';')

inat_id_list = dataframe['inat_id'].tolist()         

for observation_id in inat_id_list:

    image_url = get_large_image_url(observation_id)
    download_image(image_url,"{}{}{}".format( path_save,observation_id,'.jpg'))
