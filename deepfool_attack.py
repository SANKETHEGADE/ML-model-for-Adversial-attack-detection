import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ================= LOAD DATA =================
dataset = torchvision.datasets.ImageFolder("dataset/train", transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

# ================= MODEL (MUST MATCH main.py) =================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
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

# ================= LOAD TRAINED MODEL =================
model = CNN().to(device)

# ?? IMPORTANT: use base model, NOT detector_model
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

print("Model loaded successfully ?")

# ================= DEEPFOOL ATTACK =================
def deepfool(image, model, num_classes=2, overshoot=0.02, max_iter=50):
    image = image.clone().detach().to(device)
    image.requires_grad = True

    output = model(image)
    _, label = torch.max(output, 1)

    perturbed = image.clone().detach()
    r_tot = torch.zeros_like(image)

    for _ in range(max_iter):
        perturbed.requires_grad = True
        outputs = model(perturbed)

        current_label = torch.argmax(outputs, dim=1)
        if current_label != label:
            break

        gradients = []

        for i in range(num_classes):
            model.zero_grad()
            outputs[0, i].backward(retain_graph=True)
            gradients.append(perturbed.grad.clone())

        grad_orig = gradients[label.item()]
        min_pert = float('inf')

        for k in range(num_classes):
            if k == label.item():
                continue

            w_k = gradients[k] - grad_orig
            f_k = (outputs[0, k] - outputs[0, label]).detach()

            pert_k = torch.abs(f_k) / torch.norm(w_k.flatten())

            if pert_k < min_pert:
                min_pert = pert_k
                w = w_k

        r_i = (min_pert + 1e-4) * w / torch.norm(w)
        r_tot = r_tot + r_i

        perturbed = image + (1 + overshoot) * r_tot
        perturbed = torch.clamp(perturbed, 0, 1).detach()

    return perturbed

# ================= SAVE PATH =================
save_path = "detector_dataset/adversarial/"
os.makedirs(save_path, exist_ok=True)

# ================= GENERATE IMAGES =================
count = 0

print("Generating DeepFool adversarial images...")

for data, target in loader:
    data, target = data.to(device), target.to(device)

    adv = deepfool(data, model)

    img = adv[0].cpu().detach().numpy().transpose(1,2,0)
    img = img.clip(0,1)

    plt.imsave(save_path + f"deepfool_{count}.png", img)
    count += 1

    if count >= 100:
        break

print(f"DeepFool images saved: {count} ?")

# ================= VISUAL =================
sample_data, _ = next(iter(loader))
sample_data = sample_data.to(device)

adv_sample = deepfool(sample_data, model)

orig = sample_data[0].cpu().detach().numpy().transpose(1,2,0)
adv  = adv_sample[0].cpu().detach().numpy().transpose(1,2,0)

fig, ax = plt.subplots(1,2, figsize=(8,4))
ax[0].imshow(orig)
ax[0].set_title("Original")
ax[0].axis("off")

ax[1].imshow(adv)
ax[1].set_title("DeepFool Attack")
ax[1].axis("off")

plt.savefig("deepfool_comparison.png")
plt.show()