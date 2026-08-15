"""
detector.py

Trains the adversarial-vs-normal detector CNN.

Fixes vs. the original version:
1. Class-weighted CrossEntropyLoss (previously unweighted -- if
   detector_dataset is imbalanced, the model could just learn to
   predict the majority class and still look "accurate").
2. Full confusion matrix + per-class precision/recall/F1 printed at
   the end, instead of a single overall accuracy number, which can
   hide exactly the "always says normal" failure mode you're seeing.
3. Prints the raw class counts up front so imbalance is visible
   immediately.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


dataset = torchvision.datasets.ImageFolder("detector_dataset", transform=transform)
print("Classes:", dataset.classes)

label_counts = Counter(label for _, label in dataset.samples)
print("Class counts:", {dataset.classes[k]: v for k, v in label_counts.items()})
if len(set(label_counts.values())) > 1:
    ratio = max(label_counts.values()) / min(label_counts.values())
    print(f"⚠ Class imbalance ratio: {ratio:.2f}x "
          f"(consider running prepare_normal_dataset.py to balance counts)")


train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DetectorCNN().to(device)


num_classes = len(dataset.classes)
total = sum(label_counts.values())
class_weights = torch.tensor([
    total / (num_classes * label_counts.get(i, 1)) for i in range(num_classes)
]).to(device)
print("Loss class weights:", {dataset.classes[i]: round(w.item(), 3)
                               for i, w in enumerate(class_weights)})


criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)


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


torch.save(model.state_dict(), "detector_model.pth")
print("✅ Model saved successfully")


model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(probs, 1)

        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

accuracy = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
print(f"\n✅ Overall Test Accuracy: {accuracy:.2f}%")

print("\nConfusion matrix (rows=actual, cols=predicted):")
print("           ", "  ".join(f"{c:>10}" for c in dataset.classes))
cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
for i, row in enumerate(cm):
    print(f"{dataset.classes[i]:>10} ", "  ".join(f"{v:>10}" for v in row))

print("\nPer-class report:")
print(classification_report(all_labels, all_preds, target_names=dataset.classes, digits=3))

print("👉 If 'adversarial' recall is low here, the model genuinely isn't "
      "learning to catch adversarial images -- that's a training/data "
      "problem, separate from the confidence-threshold bug in app.py.")


print("\nFitting temperature scaling on the held-out test split...")

logits_list, labels_list = [], []
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        logits_list.append(model(images).cpu())
        labels_list.append(labels)
all_logits = torch.cat(logits_list)
all_labels_t = torch.cat(labels_list)

temperature = torch.nn.Parameter(torch.ones(1) * 1.5)
temp_optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
nll_criterion = nn.CrossEntropyLoss()

def _temp_closure():
    temp_optimizer.zero_grad()
    loss = nll_criterion(all_logits / temperature, all_labels_t)
    loss.backward()
    return loss

temp_optimizer.step(_temp_closure)
fitted_T = max(temperature.item(), 1.0)  
print(f"✅ Fitted temperature: {fitted_T:.3f}")

with open("temperature.json", "w") as f:
    import json
    json.dump({"temperature": fitted_T}, f)
print("✅ Saved temperature.json (app.py / test_detector.py load this to "
      "calibrate confidence at inference time)")
