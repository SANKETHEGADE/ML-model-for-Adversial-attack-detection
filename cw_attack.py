import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from PIL import Image

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= TRANSFORMS =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# ================= LOAD DATA =================
dataset = torchvision.datasets.ImageFolder("dataset/train", transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print("Classes:", dataset.classes)

# ================= MODEL (UPDATED) =================
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# ================= LOSS =================
class_counts = [0, 0]
for _, label in dataset:
    class_counts[label] += 1

weights = torch.tensor([
    1.0 / class_counts[0],
    1.0 / class_counts[1]
]).to(device)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# ================= TRAIN =================
epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

print("✅ Model trained successfully")

# ================= SAVE =================
torch.save(model.state_dict(), "model.pth")

# ================= C&W ATTACK =================
def cw_attack(model, images, labels, steps=50, lr=0.01):
    images = images.clone().detach().to(device)

    w = torch.atanh((images * 2 - 1).clamp(-0.999, 0.999)).clone().detach().requires_grad_(True)
    optimizer_cw = torch.optim.Adam([w], lr=lr)

    for _ in range(steps):
        adv = 0.5 * torch.tanh(w) + 0.5
        outputs = model(adv)

        loss = -criterion(outputs, labels)

        optimizer_cw.zero_grad()
        loss.backward()
        optimizer_cw.step()

    adv_images = 0.5 * torch.tanh(w) + 0.5
    return torch.clamp(adv_images, 0, 1)

# ================= DEEPFOOL ATTACK =================
def deepfool_attack(model, image, num_classes=2, overshoot=0.02, max_iter=20):
    image = image.clone().detach().to(device)
    image.requires_grad = True

    output = model(image)
    _, label = torch.max(output, 1)

    perturbed = image.clone().detach()
    r_tot = torch.zeros_like(image)

    for _ in range(max_iter):
        perturbed.requires_grad = True
        outputs = model(perturbed)

        if torch.argmax(outputs) != label:
            break

        grads = []
        for i in range(num_classes):
            model.zero_grad()
            outputs[0, i].backward(retain_graph=True)
            grads.append(perturbed.grad.clone())

        grad_orig = grads[label.item()]
        min_pert = float('inf')

        for k in range(num_classes):
            if k == label.item():
                continue

            w_k = grads[k] - grad_orig
            f_k = (outputs[0, k] - outputs[0, label]).detach()

            pert_k = torch.abs(f_k) / torch.norm(w_k.flatten())

            if pert_k < min_pert:
                min_pert = pert_k
                w = w_k

        r_i = (min_pert + 1e-4) * w / torch.norm(w)
        r_tot = r_tot + r_i

        perturbed = torch.clamp(image + (1 + overshoot) * r_tot, 0, 1).detach()

    return perturbed

# ================= GENERATE ATTACK IMAGES =================
save_path = "detector_dataset/adversarial/"
os.makedirs(save_path, exist_ok=True)

count = 0
print("Generating C&W + DeepFool images...")

for data, target in train_loader:
    data, target = data.to(device), target.to(device)

    # C&W
    cw_data = cw_attack(model, data.clone(), target)

    # DeepFool (first image only)
    deepfool_data = deepfool_attack(model, data[:1].clone())

    # SAVE C&W
    for img_tensor in cw_data:
        img = img_tensor.detach().cpu().numpy().transpose(1,2,0)
        img = img.clip(0,1)
        plt.imsave(save_path + f"cw_{count}.png", img)
        count += 1

    # SAVE DEEPFOOL
    img = deepfool_data[0].detach().cpu().numpy().transpose(1,2,0)
    img = img.clip(0,1)
    plt.imsave(save_path + f"deepfool_{count}.png", img)
    count += 1

    if count >= 150:
        break

print(f"✅ Images saved: {count}")

# ================= SAFE TEST (NO ERROR) =================
def predict_image(image_path):
    model.eval()

    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        probs = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)

    class_names = dataset.classes
    return class_names[predicted.item()], confidence.item()

# Safe check
test_img = "test_image.png"

if os.path.exists(test_img):
    label, conf = predict_image(test_img)
    print(f"Prediction: {label}")
    print(f"Confidence: {conf*100:.2f}%")
else:
    print("⚠ test_image.png not found, skipping prediction")