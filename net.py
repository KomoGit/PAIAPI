import torch
from transformers import EfficientNetForImageClassification
from PIL import Image
from torchvision import transforms
import os

# Define the folder where uploaded files are stored
UPLOAD_FOLDER = 'uploads/'

# Function to load an image file, preprocess it, and run inference
def run_inference(image_filename):
    # Full path to the image
    image_path = os.path.join(UPLOAD_FOLDER, image_filename)
    
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')

    # Define the transformations
    resize_transform = transforms.Compose([
        transforms.Resize((600, 600)),   # Resize image to 600x600
        transforms.ToTensor()            # Convert to Tensor
    ])

    image = resize_transform(image)

    # Add a batch dimension
    image = image.unsqueeze(0)  # This makes it [1, 3, 600, 600]

    # Load the model
    model = EfficientNetForImageClassification.from_pretrained("google/efficientnet-b7")

    # Perform inference
    with torch.no_grad():
        logits = model(image).logits

    # Get the predicted label
    predicted_label = logits.argmax(-1).item()
    
    return model.config.id2label[predicted_label]