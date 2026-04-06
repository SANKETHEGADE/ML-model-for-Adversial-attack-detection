import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# ================= SETTINGS =================

attack_type = "bim"   # 🔥 CHANGE HERE → "fgsm" or "bim"

# ================= TRANSFORM =================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ================= LOAD DATA =================

dataset = torchvision.datasets.ImageFolder("dataset/train", transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# ================= MODEL =================

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(32 * 54 * 54, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = CNN()

# ================= LOSS + OPTIMIZER =================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ================= TRAIN =================

for epoch in range(10):
    for data, target in loader:
        output = model(data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch done")

print("Model trained successfully")

# ================= FGSM ATTACK =================

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return torch.clamp(perturbed_image, 0, 1)

# ================= BIM ATTACK =================

def bim_attack(image, epsilon, alpha, iterations):
    perturbed_image = image.clone().detach()

    for i in range(iterations):
        perturbed_image.requires_grad = True

        output = model(perturbed_image)
        loss = criterion(output, target)

        model.zero_grad()
        loss.backward()

        grad = perturbed_image.grad.data

        perturbed_image = perturbed_image + alpha * grad.sign()

        eta = torch.clamp(perturbed_image - image, -epsilon, epsilon)
        perturbed_image = torch.clamp(image + eta, 0, 1).detach()

    return perturbed_image

# ================= GENERATE ADVERSARIAL IMAGES =================

save_path = "C:/Adversarial_Project/detector_dataset/adversarial/"
os.makedirs(save_path, exist_ok=True)

count = 0

for data, target in loader:

    # ===== GET GRADIENTS =====
    data.requires_grad = True

    output = model(data)
    loss = criterion(output, target)

    model.zero_grad()
    loss.backward()

    data_grad = data.grad.data

    # ===== SELECT ATTACK =====
    if attack_type == "fgsm":
        perturbed_data = fgsm_attack(data, 0.1, data_grad)

    elif attack_type == "bim":
        perturbed_data = bim_attack(data, epsilon=0.1, alpha=0.01, iterations=5)

    # ===== SAVE IMAGES =====
    for i in range(len(perturbed_data)):
        img = perturbed_data[i].detach().numpy().transpose(1, 2, 0)

        plt.figure()
        plt.imshow(img)
        plt.axis('off')
        plt.savefig(save_path + f"adv_{count}.png")
        plt.close()

        count += 1

        if count >= 100:
            break

    if count >= 100:
        break

print(f"{count} adversarial images generated using {attack_type.upper()}")

# ================= SAVE EXAMPLE =================

img = perturbed_data[0].detach().numpy().transpose(1, 2, 0)

plt.figure()
plt.imshow(img)
plt.title(f"{attack_type.upper()} Example")
plt.axis('off')
plt.savefig("C:/Adversarial_Project/adversarial_example.png")
plt.close()

print("Example adversarial image saved")