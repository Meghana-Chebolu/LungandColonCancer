import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from efficientnet_pytorch import EfficientNet
from PIL import Image
import numpy as np
# Define the custom model
class EfficientNetB3Custom(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB3Custom, self).__init__()
        self.base_model = EfficientNet.from_pretrained('efficientnet-b3')
        
        # Freeze the parameters of the base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Modify the classifier part of the base model
        num_ftrs = self.base_model._fc.in_features
        self.base_model._fc = nn.Identity()  # Removing the classification head
        
        # Add custom dense layers
        self.custom_classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.custom_classifier(x)
        return x

def test_image(image):
    transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
    model = EfficientNetB3Custom(5)
    state_dict_path = 'lungcolon.pt' 
    state_dict = torch.load(state_dict_path,map_location=torch.device('cpu'))

    # Load state dict into model
    model.load_state_dict(state_dict)

    # Set the model to evaluation mode
    model.eval()
    transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    if image.mode != 'RGB':
        image = image.convert('RGB')
    device = torch.device("cpu")  # Set device to CPU
    model = model.to(device)
    transformed_image = transform(image)
    numpy_array = transformed_image.permute(1, 2, 0).numpy()
    pil_image = Image.fromarray((numpy_array * 255).astype(np.uint8))
    pil_image.save('static/output_image.jpg')
    transformed_image = transformed_image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(transformed_image)
    index = torch.argmax(output)
    d = {}
    class_names = ['Colon Adenocarcinoma', 'Colon Benign', 'Lung Adenocarcinoma', 'Lung Benign', 'Lung Squamous Cell Carcinoma']
    d["Prediction"] =  class_names[index]
    print(d)
    return d
