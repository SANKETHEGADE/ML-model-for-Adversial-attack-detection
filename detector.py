import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ================= LOAD DATA =================
dataset = torchvision.datasets.ImageFolder(
    "detector_dataset",
    transform=transform
)

print("Classes:", dataset.classes)

# ================= BALANCED TRAIN/TEST SPLIT =================
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(
    dataset,
    [train_size, test_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)

# ================= CNN MODEL =================
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

# ================= MODEL =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DetectorCNN().to(device)

# ================= LOSS + OPTIMIZER =================
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0005
)

# ================= TRAINING =================
epochs = 10

print("\nTraining Detector...\n")

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}  Loss: {running_loss:.4f}")

print("\n✅ Detector trained successfully")

# ================= SAVE MODEL =================
torch.save(model.state_dict(), "detector_model.pth")

print("✅ Model saved successfully")

# ================= TEST ACCURACY =================
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        probs = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print(f"\n✅ REAL Test Accuracy: {accuracy:.2f}%")

# ================= SAMPLE PREDICTIONS =================
print("\nSample Predictions:\n")

classes = dataset.classes

count = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        probs = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probs, 1)

        for i in range(len(images)):

            pred_class = classes[predicted[i].item()]
            real_class = classes[labels[i].item()]
            conf = confidence[i].item() * 100

            print(
                f"Predicted: {pred_class} | "
                f"Actual: {real_class} | "
                f"Confidence: {conf:.2f}%"
            )

            count += 1

            if count >= 10:
                break

        if count >= 10:
            break