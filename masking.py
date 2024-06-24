from models import refine_mask, remove_background, get_embedding
import json
import os
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import cv2

import tqdm

#read json file : image_dict.json
with open('data/image_dict.json') as f:
    image_dict = json.load(f)


#if data folder does not exist, create it
if not os.path.exists('data/masked_images'):
    os.makedirs('data/masked_images')

#iterate over the images in data/images_sample
list_of_images = os.listdir('data/images_sample')

embeddings_dict = {}
for image_file in tqdm.tqdm(list_of_images):
    image_id = image_file.split('.')[0]
    image_path = 'data/images_sample/' + image_file
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_without_bg = remove_background(image_rgb)

    # Save the result to check the masking
    q=cv2.imwrite("data/masked_images/" + image_id + "_mask.png", cv2.cvtColor(image_without_bg, cv2.COLOR_RGBA2BGRA))

