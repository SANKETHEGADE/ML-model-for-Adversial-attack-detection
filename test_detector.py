import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ================= UPDATED CNN MODEL =================
class DetectorCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(

            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(128, 2)
        )

    def forward(self, x):

        x = self.conv(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x

# ================= LOAD MODEL =================
model = DetectorCNN().to(device)

model.load_state_dict(
    torch.load("detector_model.pth", map_location=device)
)

model.eval()

# ================= CLASS NAMES =================
classes = ["adversarial", "normal"]

# ================= TEST FOLDER =================
test_folder = "test_images"

# ================= PREDICTION =================
for filename in os.listdir(test_folder):

    path = os.path.join(test_folder, filename)

    try:
        img = Image.open(path).convert("RGB")

        img = transform(img)

        img = img.unsqueeze(0).to(device)

        with torch.no_grad():

            output = model(img)

            probs = torch.softmax(output, dim=1)

            confidence, predicted = torch.max(probs, 1)

        label = classes[predicted.item()]
        conf = confidence.item() * 100

        # Confidence threshold
        if label == "adversarial" and conf < 75:
            label = "normal"

        if label == "adversarial":
            print(f"{filename} → Adversarial 🚨 ({conf:.2f}%)")
        else:
            print(f"{filename} → Normal ✅ ({conf:.2f}%)")

    except Exception as e:
        print(f"Error processing {filename}: {e}")