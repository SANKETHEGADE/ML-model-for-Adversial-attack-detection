import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import os

# Model
class DetectorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,16,3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32*54*54,128),
            nn.ReLU(),
            nn.Linear(128,2)
        )

    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Load model
model = DetectorCNN()
model.load_state_dict(torch.load("detector_model.pth"))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# Folder with test images
folder = "test_images"

# Loop through images
for file in os.listdir(folder):
    if file.endswith(".png") or file.endswith(".jpg"):
        path = os.path.join(folder, file)

        img = Image.open(path).convert("RGB")
        img = transform(img).unsqueeze(0)

        output = model(img)
        _, predicted = torch.max(output, 1)

        # ✅ Correct label mapping
        if predicted.item() == 0:
            print(f"{file} → Adversarial Image 🚨")
        else:
            print(f"{file} → Normal Image ✅")