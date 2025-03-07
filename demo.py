import torch
import joblib
from PIL import Image
import torchvision.transforms as transforms
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Define the EmotionResNet18 architecture
import torch.nn as nn
from torchvision import models

class EmotionClassifier(nn.Module):
    def __init__(self, num_classes):
        super(EmotionClassifier, self).__init__()
        self.resnet = models.resnet18(weights=None)

        # Modify the final layers
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        # Add dropout to some intermediate layers
        self.resnet.layer3.add_module('dropout', nn.Dropout(0.2))
        self.resnet.layer4.add_module('dropout', nn.Dropout(0.2))

    def forward(self, x):
        return self.resnet(x)

def predict_emotion(image_path):
    # Load class names and transform config
    class_names = joblib.load("class_names.joblib")
    transform_config = joblib.load("transform_config.joblib")

    # Create transform
    inference_transform = transforms.Compose([
        transforms.Resize(transform_config["resize_size"]),
        transforms.ToTensor(),
        transforms.Normalize(
            transform_config["normalization_mean"],
            transform_config["normalization_std"]
        )
    ])

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Try loading the full model first
    try:
        model = torch.load("full_model.pth", map_location=device)
    except:
        # If that fails, load the architecture and state dict separately
        num_classes = len(class_names)
        model = EmotionClassifier(num_classes)
        model.load_state_dict(torch.load("model_state_dict.pth", map_location=device))

    model.to(device)
    model.eval()

    # Load and prepare image
    image = Image.open(image_path).convert('RGB')
    tensor = inference_transform(image).unsqueeze(0).to(device)

    # Get prediction
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, predicted = torch.max(probabilities, 0)
        predicted_class = class_names[predicted.item()]

    # Display the image and prediction
    plt.figure(figsize=(8, 6))
    plt.imshow(np.array(image))
    plt.title(f"Predicted emotion: {predicted_class} (Confidence: {confidence.item():.2f})")
    plt.axis('off')
    plt.show()

    # Print all probabilities
    print("Emotion probabilities:")
    for i, emotion in enumerate(class_names):
        print(f"{emotion}: {probabilities[i].item():.4f}")

    return predicted_class, confidence.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion Recognition Demo")
    parser.add_argument("image_path", help="Path to the image file")
    args = parser.parse_args()

    predict_emotion(args.image_path)
