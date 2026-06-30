import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ================= LOAD DATA =================
dataset = torchvision.datasets.ImageFolder("dataset/train", transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

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

model = CNN().to(device)

# ================= LOSS =================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ================= TRAIN =================
for epoch in range(10):
    for data, target in loader:
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch done")

print("Model trained successfully")

# ================= FGSM =================
def fgsm_attack(image, epsilon, data_grad):
    return torch.clamp(image + epsilon * data_grad.sign(), 0, 1)

# ================= BIM =================
def bim_attack(model, images, labels, epsilon=0.1, alpha=0.01, iters=10):
    original_images = images.clone().detach()
    perturbed_images = images.clone().detach()

    for _ in range(iters):
        perturbed_images.requires_grad = True

        outputs = model(perturbed_images)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        model.zero_grad()
        loss.backward()

        grad = perturbed_images.grad.data
        perturbed_images = perturbed_images + alpha * grad.sign()

        eta = torch.clamp(perturbed_images - original_images, -epsilon, epsilon)
        perturbed_images = torch.clamp(original_images + eta, 0, 1).detach()

    return perturbed_images

# ================= PGD =================
def pgd_attack(model, images, labels, epsilon=0.1, alpha=0.01, iters=10):
    original_images = images.clone().detach()
    perturbed_images = images.clone().detach()

    for _ in range(iters):
        perturbed_images.requires_grad = True

        outputs = model(perturbed_images)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        model.zero_grad()
        loss.backward()

        grad = perturbed_images.grad.data
        perturbed_images = perturbed_images + alpha * grad.sign()

        eta = torch.clamp(perturbed_images - original_images, -epsilon, epsilon)
        perturbed_images = torch.clamp(original_images + eta, 0, 1).detach()

    return perturbed_images

# ================= C&W ATTACK =================
def cw_l2_attack(model, images, labels, c=1.0, kappa=0, max_iter=50, lr=0.01):
    model.eval()
    images = images.to(device)
    labels = labels.to(device)

    x_tanh = torch.atanh((images * 2 - 1).clamp(-0.9999, 0.9999))
    w = x_tanh.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([w], lr=lr)

    for step in range(max_iter):
        adv_images = 0.5 * torch.tanh(w) + 0.5

        l2_loss = ((adv_images - images) ** 2).sum()

        outputs = model(adv_images)

        one_hot = torch.zeros_like(outputs).to(device)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)

        real = (outputs * one_hot).sum(dim=1)
        other = (outputs * (1 - one_hot) - one_hot * 1e4).max(dim=1)[0]

        f_loss = torch.clamp(real - other + kappa, min=0)

        loss = l2_loss + c * f_loss.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    adv_images = 0.5 * torch.tanh(w) + 0.5
    return adv_images.detach()

# ================= GENERATE ATTACKS =================
save_path = "C:/Adversarial_Project/detector_dataset/adversarial/"
os.makedirs(save_path, exist_ok=True)

count = 0

print("Generating FGSM, BIM, PGD, and C&W attacks...")

for data, target in loader:
    data, target = data.to(device), target.to(device)

    data.requires_grad = True

    output = model(data)
    loss = criterion(output, target)

    model.zero_grad()
    loss.backward()

    data_grad = data.grad.data

    # Generate attacks
    fgsm_data = fgsm_attack(data, 0.1, data_grad)
    bim_data = bim_attack(model, data.clone(), target)
    pgd_data = pgd_attack(model, data.clone(), target)
    cw_data = cw_l2_attack(model, data.clone(), target)

    # Save images
    for attack_name, attack_data in zip(
        ["fgsm", "bim", "pgd", "cw"],
        [fgsm_data, bim_data, pgd_data, cw_data]
    ):
        for img_tensor in attack_data:
            img = img_tensor.cpu().detach().numpy().transpose(1, 2, 0)
            plt.imsave(save_path + f"{attack_name}_{count}.png", img)
            count += 1

    if count >= 150:
        break

print("✅ FGSM + BIM + PGD + C&W images generated successfully 🚀")