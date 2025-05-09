import asyncio
import os
from OPCUA import collect_data
import cv2
from pypylon import pylon as py
from ImageCap import PylonCameras
from PIL import Image
import ns_hymap_inference
import json

json_file_path = 'output_data.json'


# Define constants
SIZE = (1080, 720)
BASE_IMG_LOC = 'Dataset'

# Ensure the base image directory exists
os.makedirs(BASE_IMG_LOC, exist_ok=True)

#Save the Image from the Cameras
def save_image(image, file_path):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    pil_image.save(file_path)


#Grab Image from Cameras, create the file path
async def capture_and_save_images(cap,iteration):
    count = 0
    batch = f'BATCH{count // 2 + 1}'
    os.makedirs(os.path.join(BASE_IMG_LOC, batch), exist_ok=True)

    # Capture images from both cameras
    for i in range(2):
        res = cap.cameras[i].RetrieveResult(5000, py.TimeoutHandling_ThrowException)
        if res.GrabSucceeded():
            # idx, device = cap.get_image_device(res)
            img = cap.converters[i].Convert(res)
            image = img.GetArray()
            image = cap.set_img_size(image, SIZE)
            image = cap.adjust_white_balance(image)
            # Save the raw image
            raw_path = os.path.join(BASE_IMG_LOC, batch, f'{iteration:06d}_camera{i}.png')
            save_image(image, raw_path)
    print("Image capture complete.")

#Append new output to JSON file for logging
def append_to_json_file(data, file_path):
    # Read existing data from the file if it exists
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            try:
                existing_data = json.load(file)
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []
    # Append new data
    existing_data.append(data)
    # Write the updated data back to the file
    with open(file_path, 'w') as file:
        json.dump(existing_data, file, indent=4)

def get_image_path(iteration, camera_index):
    return os.path.join(BASE_IMG_LOC, 'BATCH1', f'{iteration:06d}_camera{camera_index}.png')

#Capture Image, get sensor data from OPC UA, merge and run through the different forecasting models


