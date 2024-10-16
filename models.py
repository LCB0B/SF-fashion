import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image

# Check if MPS is available
device = torch.device("mps" if torch.has_mps else "cpu")
print(f"Using device: {device}")

# Load the DeepLabV3 model for background removal
deeplab_model = models.segmentation.deeplabv3_resnet50(pretrained=True).to(device)
deeplab_model.eval()

# Load the MobileNetV2 model for embeddings
mobilenet_model = models.mobilenet_v2(pretrained=True)
mobilenet_model.classifier = torch.nn.Identity()  # Remove the classification layer
mobilenet_model = mobilenet_model.to(device)
mobilenet_model.eval()

# Define the transformation for input images
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def refine_mask(mask):
    # Convert mask to uint8 format
    mask = (mask * 255).astype(np.uint8)

    # Apply morphological operations to remove noise and smooth the edges
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    return mask

def keep_largest_connected_component(mask):
    num_labels, labels_im = cv2.connectedComponents(mask)
    if num_labels > 2:  # More than one person (background counts as one label)
        print(f"Found {num_labels - 1} people in the image. Keeping only the largest connected component.")
        largest_component = 0
        largest_size = 0
        for label in range(1, num_labels):
            size = np.sum(labels_im == label)
            if size > largest_size:
                largest_size = size
                largest_component = label
        mask = labels_im == largest_component
    return mask.astype(np.uint8)

def remove_background(image):
    image_pil = Image.fromarray(image)
    input_tensor = preprocess(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        output = deeplab_model(input_tensor)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()
    mask = output_predictions == 15  # Class 15 is the person class
    mask = cv2.resize(mask.astype(np.uint8), (image_pil.width, image_pil.height))

    # Keep only the largest connected component
    mask = keep_largest_connected_component(mask)

    # Refine the mask
    refined_mask = refine_mask(mask)

    # Create a 4-channel image (RGBA)
    image_rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    image_rgba[:, :, 3] = refined_mask  # Apply the refined mask to the alpha channel

    return image_rgba


def get_embedding(image):
    image = Image.fromarray(image)
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = mobilenet_model(input_tensor)
    return embedding.squeeze().cpu().numpy()


if __name__ == "__main__":
    # Load an image
    image_path = 'data/images_sample/4608.jpg'
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Remove the background
    image_without_bg = remove_background(image_rgb)

    # Save the result to check the masking
    cv2.imwrite("image_with_mask.png", cv2.cvtColor(image_without_bg, cv2.COLOR_RGBA2BGRA))

    # Convert back to RGB for embedding extraction (ignore alpha channel)
    image_rgb_no_alpha = cv2.cvtColor(image_without_bg, cv2.COLOR_BGRA2BGR)

    # Generate embeddings
    embedding = get_embedding(image_rgb_no_alpha)

    print("Embedding shape:", embedding.shape)
