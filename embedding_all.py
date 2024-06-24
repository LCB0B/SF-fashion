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


#create a new dictionary to store the embeddings and mask images

#if data folder does not exist, create it
if not os.path.exists('embeddings'):
    os.makedirs('embeddings')

#iterate over the images in data/images_sample
list_of_images = os.listdir('data/masked_images')

embeddings_dict = {}
for image_file in tqdm.tqdm(list_of_images):
    image_id = image_file.split('.')[0][:-5]
    image_path = 'data/masked_images/' + image_file
    image_without_bg = cv2.imread(image_path)
    # Convert back to RGB for embedding extraction (ignore alpha channel)
    image_rgb_no_alpha = cv2.cvtColor(image_without_bg, cv2.COLOR_BGRA2BGR)
    # Generate embeddings
    embedding = get_embedding(image_rgb_no_alpha)
    embeddings_dict[image_id] = embedding
    #cv2.imwrite('data/embeddings/' + image_id + '.png', image_rgb_no_alpha)


#save the embeddings dictionary to a json file
with open('embeddings_dict.json', 'w') as f:
    json.dump(embeddings_dict, f)
#save the embeddings values to a npy file
np.save('embeddings.npy', embeddings_dict.values())