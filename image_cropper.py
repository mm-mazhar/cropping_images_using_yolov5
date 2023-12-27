# Import necessary libraries for image cropping
from io import BytesIO

import requests
from PIL import Image


def crop_object(image_obj, coordinates):
    # Open the image
    image = Image.open(image_obj)

    # Crop the image based on the provided coordinates
    cropped_image = image.crop(coordinates)

    return cropped_image


def crop_object_url(url, coordinates):
    # Open the image from a file or URL
    if url.startswith("http"):
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(url)

    # Crop the image using the provided coordinates
    x1, y1, x2, y2 = coordinates
    cropped_image = image.crop((x1, y1, x2, y2))

    return cropped_image
