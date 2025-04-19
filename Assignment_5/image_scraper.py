import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from PIL import Image
from io import BytesIO

# Function to download images from a given URL
def download_images(url, folder_path, max_images):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    image_tags = soup.find_all('img')
    image_count = 0
    
    for image_tag in image_tags:
        image_url = image_tag.get('src')
        
        if image_url:
            image_url = urljoin(url, image_url)
            image_name = f"image{image_count}.jpg"  # Changed to a generic name
            
            try:
                image_data = requests.get(image_url).content
                image = Image.open(BytesIO(image_data))
                image_path = os.path.join(folder_path, image_name)
                image.save(image_path, "JPEG")  # Save as JPEG format
                image_count += 1
                if image_count >= max_images:
                    break
            except Exception as e:
                print(f"Failed to download {image_url}: {e}")

# Define URLs for chicken and duck images
chicken_url = "https://unsplash.com/images/animals/chicken"
duck_url = "https://unsplash.com/s/photos/duck"

# Create directories to save images
os.makedirs('chickens', exist_ok=True)
os.makedirs('ducks', exist_ok=True)

# Download images
download_images(chicken_url, 'chickens', max_images=100)  # Change max_images as needed
download_images(duck_url, 'ducks', max_images=100)  # Change max_images as needed
