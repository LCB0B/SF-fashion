import numpy as np
import pacmap
import matplotlib.pyplot as plt
import json

#load the description dictionary
with open('data/image_dict.json') as f:
    description_dict = json.load(f)
#replace the key by integer
description_dict = {int(k): v for k, v in description_dict.items()}

# Load the embeddings dictionary from the JSON file
with open('embeddings_dict.json') as f:
    embeddings_dict = json.load(f)

# Convert the keys to integers
embeddings_dict = {int(k): v for k, v in embeddings_dict.items()}
# match on key, the description "collection: and the embeddings in a vector
collection = [description_dict[f]['collection'] for f in embeddings_dict.keys()]

# Convert the dictionary values to a numpy array
embeddings = np.array(list(embeddings_dict.values()))

# Initialize the PaCMAP model
pacmap_model = pacmap.PaCMAP(n_components=2)

# Fit and transform the embeddings into 2D space
embeddings_2d = pacmap_model.fit_transform(embeddings)

name = 'collection'
# collection color dict based on tab20
unique_collection = list(set(collection))
collection_color = dict(zip(unique_collection,range(len(unique_collection))))

#create a color list
color_list = [collection_color[c] for c in collection]

# # Plot the projected embeddings
plt.figure(figsize=(10, 7))
a=plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=1, alpha=0.8,c=collection,cmap='tab20')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
#remove the axis
plt.axis('off')
plt.tight_layout()
plt.savefig(f'embeddings_{name}.pdf')
plt.savefig(f'embeddings_{name}.png',dpi=300)
