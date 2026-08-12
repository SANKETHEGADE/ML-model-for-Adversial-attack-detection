"""
verify_model.py

Diagnostic tool. Loads detector_model.pth exactly the way app.py does
(same architecture, same transform, same .eval()), then classifies
every single file currently in detector_dataset/adversarial and
detector_dataset/normal.

Why this matters: detector.py reported 93.8% adversarial recall on its
own internal test split, but live testing through app.py shows almost
all adversarial images called "normal". This script tells us WHERE
the disagreement is:

- If this script ALSO shows near-100% "normal" on the adversarial
  folder -> the currently-saved detector_model.pth genuinely doesn't
  match the images currently on disk (most likely cause: main.py was
  re-run after detector.py trained, regenerating a fresh random batch
  of images that overwrote the ones the model was actually trained on).

- If this script shows GOOD accuracy (matching detector.py's
  confusion matrix) but app.py still gets it wrong on the exact same
  files -> the bug is specific to app.py's request-handling code, not
  the model or dataset.

Run this from the same folder as app.py / detector.py.
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

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
        return self.fc(x)

model = DetectorCNN().to(device)
model.load_state_dict(torch.load("detector_model.pth", map_location=device))
model.eval()

classes = ["adversarial", "normal"]  # same order as app.py

print(f"Loaded detector_model.pth "
      f"(size: {os.path.getsize('detector_model.pth')} bytes, "
      f"modified: {os.path.getmtime('detector_model.pth')})")
print()

for folder, true_label in [("detector_dataset/adversarial", "adversarial"),
                            ("detector_dataset/normal", "normal")]:

    files = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    correct = 0
    normal_calls = 0
    adversarial_calls = 0

    print(f"--- {folder} ({len(files)} files) ---")

    for fname in files:
        path = os.path.join(folder, fname)
        img = Image.open(path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img)
            probs = torch.softmax(output, dim=1)
            conf, predicted = torch.max(probs, 1)

        label = classes[predicted.item()]
        if label == "normal":
            normal_calls += 1
        else:
            adversarial_calls += 1
        if label == true_label:
            correct += 1

    accuracy = 100 * correct / len(files) if files else 0
    print(f"Accuracy on this folder: {accuracy:.1f}% "
          f"({correct}/{len(files)})")
    print(f"Predicted 'adversarial': {adversarial_calls}   "
          f"Predicted 'normal': {normal_calls}")
    print()

print("If accuracy here is high but app.py still gets it wrong on the "
      "same files -> bug is in app.py's request handling.")
print("If accuracy here is ALSO low (near-100% 'normal' on the "
      "adversarial folder) -> the model and the current dataset don't "
      "match. Most likely: main.py was re-run after detector.py "
      "trained, silently regenerating a new random batch of images.")