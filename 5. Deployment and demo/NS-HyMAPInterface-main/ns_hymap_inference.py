import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet
import numpy as np
import random

time_series_input_dim = 3
image_output_dim = 128  # Adjusted for EfficientNet output
hidden_dim = 64
output_dim = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensure reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

# Call the seed function
set_seed(0)

# Define the TimeSeriesAutoencoder and FusionModel classes
class TimeSeriesAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(TimeSeriesAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

class FusionModel(nn.Module):
    def __init__(self, time_series_input_dim, image_output_dim, hidden_dim, output_dim):
        super(FusionModel, self).__init__()
        self.time_series_autoencoder = TimeSeriesAutoencoder(time_series_input_dim, hidden_dim, image_output_dim)
        self.time_series_autoencoder.freeze_encoder()  # Freeze the encoder part
        self.image_encoder = EfficientNet.from_pretrained('efficientnet-b0')
        self.fc = nn.Sequential(
            nn.Linear(image_output_dim + 1280, hidden_dim),  # EfficientNet-b0 output features are 1280
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Adding Dropout for regularization
            nn.Linear(hidden_dim, output_dim)
        )
        self.image_encoder._fc = nn.Identity()  # Remove the final classification layer

    def forward(self, time_series, image=None):
        reconstructed_time_series, latent_time_series = self.time_series_autoencoder(time_series)
        if image is not None:
            latent_image = self.image_encoder(image)
            latent_image = torch.flatten(latent_image, 1)  # Flatten the image features
            latent_time_series = torch.flatten(latent_time_series, 1)  # Flatten time series features
            fused = torch.cat((latent_time_series, latent_image), dim=1)
            output = self.fc(fused)
            return output, reconstructed_time_series, latent_image
        else:
            return latent_time_series, reconstructed_time_series

# Load the model

model = FusionModel(time_series_input_dim, image_output_dim, hidden_dim, output_dim).to(device)

    # Load the state dictionary, mapping to CPU if necessary
model_path = 'OldModels/KI_new_final.pth'  # Replace with actual path
state_dict = torch.load(model_path, map_location=device)

    # Filter out the keys in the state_dict that are not present in the model's state_dict
model_state_dict = model.state_dict()
filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}

    # Update the model's state_dict with the filtered state_dict
model_state_dict.update(filtered_state_dict)

    # Load the updated state_dict into the model
model.load_state_dict(model_state_dict)

    # Set the model to evaluation mode
model.eval()

# Dummy data for testing
def prepare_image(image_path):
    # Define transformations for the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
    return image

def prepare_time_series(time_series_data):
    # Convert time series data to a tensor and move to device
    return torch.tensor(time_series_data, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension

# # Paths and data
# image_path = '000000_camera0.png'
# time_series_data = [[658, 468, 700]]  # Example time series data

# Prepare inputs
# image = prepare_image(image_path)
# time_series = prepare_time_series(time_series_data)

# Perform inference with image
def make_inference_wImage(image, time_series):
    with torch.no_grad():
        output_with_image, reconstructed_time_series_with_image, latent_image = model(time_series, image)
        # print("Output with image:", output_with_image)
        print("Reconstructed Time Series with image:", reconstructed_time_series_with_image)
        # print("Latent Image Features:", latent_image)
        return reconstructed_time_series_with_image
# Perform inference without image
def make_inference_woutImage(time_series):
    with torch.no_grad():
        latent_time_series_only, reconstructed_time_series_only = model(time_series)
        # print("Latent Time Series only:", latent_time_series_only)
        print("Reconstructed Time Series only:", reconstructed_time_series_only)
        return reconstructed_time_series_only
# Print the outputs

#
# make_inference_wImage(image,time_series)
# make_inference_woutImage(time_series)
