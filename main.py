import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


dataset = torchvision.datasets.ImageFolder("dataset/train", transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

num_classes = len(dataset.classes)
print("Classes:", dataset.classes, f"({num_classes} classes)")


class CNN(nn.Module):
    def __init__(self, num_classes):
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
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


model = CNN(num_classes).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


print("Training model...")

for epoch in range(10):
    model.train()

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/10 done")

print("Model trained successfully")


model.eval()
correct, total = 0, 0
with torch.no_grad():
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        preds = torch.argmax(model(data), dim=1)
        correct += (preds == target).sum().item()
        total += target.size(0)
print(f"Surrogate train accuracy: {100*correct/total:.2f}%")


torch.save(model.state_dict(), "model.pth")
print("Base model saved as model.pth")


def fgsm_attack(image, epsilon, data_grad):
    perturbed = image + epsilon * data_grad.sign()
    return torch.clamp(perturbed, 0, 1)


def bim_attack(model, images, labels, epsilon=0.1, alpha=0.01, iters=10):
    original = images.clone().detach()
    perturbed = images.clone().detach()

    for _ in range(iters):
        perturbed.requires_grad = True

        outputs = model(perturbed)
        loss = criterion(outputs, labels)

        model.zero_grad()
        loss.backward()

        grad = perturbed.grad.data

        perturbed = perturbed + alpha * grad.sign()
        eta = torch.clamp(perturbed - original, -epsilon, epsilon)
        perturbed = torch.clamp(original + eta, 0, 1).detach()

    return perturbed


def pgd_attack(model, images, labels, epsilon=0.1, alpha=0.01, iters=10):
    original = images.clone().detach()

    perturbed = original + torch.empty_like(original).uniform_(-epsilon, epsilon)
    perturbed = torch.clamp(perturbed, 0, 1)

    for _ in range(iters):
        perturbed.requires_grad = True

        outputs = model(perturbed)
        loss = criterion(outputs, labels)

        model.zero_grad()
        loss.backward()

        grad = perturbed.grad.data

        perturbed = perturbed + alpha * grad.sign()
        eta = torch.clamp(perturbed - original, -epsilon, epsilon)
        perturbed = torch.clamp(original + eta, 0, 1).detach()

    return perturbed


def cw_attack(model, images, labels, steps=50, lr=0.01, c=1.0):
    images = images.clone().detach()
    w = torch.atanh((images * 2 - 1).clamp(-0.999, 0.999)).clone().detach().requires_grad_(True)
    optimizer_cw = torch.optim.Adam([w], lr=lr)

    for _ in range(steps):
        adv = 0.5 * torch.tanh(w) + 0.5 
        outputs = model(adv)

        cls_loss = -criterion(outputs, labels)
        l2_loss = torch.norm((adv - images).view(images.size(0), -1), dim=1).mean()
        loss = cls_loss + c * l2_loss

        optimizer_cw.zero_grad()
        loss.backward()
        optimizer_cw.step()

    adv_images = 0.5 * torch.tanh(w) + 0.5
    return torch.clamp(adv_images, 0, 1).detach()


def _deepfool_single(model, image, num_classes, overshoot=0.02, max_iter=50):
    
    image = image.clone().detach()
    output = model(image)
    label = torch.argmax(output, dim=1)

    perturbed = image.clone().detach()
    r_tot = torch.zeros_like(image)

    for _ in range(max_iter):
        perturbed.requires_grad = True
        outputs = model(perturbed)

        if torch.argmax(outputs, dim=1) != label:
            break

        gradients = []
        for i in range(num_classes):
            model.zero_grad()
            outputs[0, i].backward(retain_graph=True)
            gradients.append(perturbed.grad.clone())

        grad_orig = gradients[label.item()]
        min_pert = float('inf')
        w = None

        for k in range(num_classes):
            if k == label.item():
                continue
            w_k = gradients[k] - grad_orig
            f_k = (outputs[0, k] - outputs[0, label]).detach()
            pert_k = torch.abs(f_k) / (torch.norm(w_k.flatten()) + 1e-8)
            if pert_k < min_pert:
                min_pert = pert_k
                w = w_k

        r_i = (min_pert + 1e-4) * w / (torch.norm(w) + 1e-8)
        r_tot = r_tot + r_i
        perturbed = torch.clamp(image + (1 + overshoot) * r_tot, 0, 1).detach()

    return perturbed.squeeze(0)


def deepfool_attack(model, images, num_classes):
    """
    DeepFool works per-image (it needs a fresh gradient pass per sample
    to find the nearest decision boundary), so this loops over the
    batch. It's noticeably slower than FGSM/BIM/PGD/C&W -- that's
    expected, not a bug.
    """
    was_training = model.training
    model.eval()
    results = []
    for i in range(images.size(0)):
        single = images[i:i+1].clone().detach()
        results.append(_deepfool_single(model, single, num_classes))
    if was_training:
        model.train()
    return torch.stack(results, dim=0)


def attack_success_mask(model, perturbed_images, true_labels):
    """
    Returns a boolean tensor: True where the perturbed image actually
    fools the model (predicted label != true label).

    This is the core fix. The original script saved every perturbed
    image into detector_dataset/adversarial regardless of whether the
    attack worked. Images that failed to flip the model's prediction
    are statistically almost indistinguishable from normal images --
    training the detector on a folder full of those mislabeled as
    "adversarial" directly teaches it that "adversarial" often looks
    just like "normal", which pushes it toward predicting "normal"
    across the board.
    """
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(perturbed_images), dim=1)
    model.train()
    return preds != true_labels


normal_path = "detector_dataset/normal"
adv_path = "detector_dataset/adversarial"

os.makedirs(normal_path, exist_ok=True)
os.makedirs(adv_path, exist_ok=True)

TARGET_PER_CLASS = 150 
ATTACK_NAMES = ["fgsm", "bim", "pgd", "cw", "deepfool"]
TARGET_PER_ATTACK = TARGET_PER_CLASS // len(ATTACK_NAMES) 

normal_count = 0
adv_count = 0
attack_attempts = 0
attack_successes = 0
per_attack_stats = {name: {"attempts": 0, "successes": 0, "saved": 0}
                     for name in ATTACK_NAMES}

print("Generating NORMAL + FGSM + BIM + PGD + C&W + DeepFool images...")

model.eval()

def all_attack_quotas_met():
    return all(per_attack_stats[a]["saved"] >= TARGET_PER_ATTACK for a in ATTACK_NAMES)

for data, target in loader:
    if normal_count >= TARGET_PER_CLASS and all_attack_quotas_met():
        break

    data, target = data.to(device), target.to(device)


    if normal_count < TARGET_PER_CLASS:
        for img_tensor in data:
            if normal_count >= TARGET_PER_CLASS:
                break
            img = img_tensor.cpu().detach().numpy().transpose(1, 2, 0).clip(0, 1)
            plt.imsave(os.path.join(normal_path, f"normal_{normal_count}.png"), img)
            normal_count += 1

    if all_attack_quotas_met():
        continue

  
    data_for_grad = data.clone().detach().requires_grad_(True)
    output = model(data_for_grad)
    loss = criterion(output, target)
    model.zero_grad()
    loss.backward()
    data_grad = data_for_grad.grad.data

   
    fgsm_data = fgsm_attack(data, 0.1, data_grad)
    bim_data = bim_attack(model, data.clone(), target)
    pgd_data = pgd_attack(model, data.clone(), target)
    cw_data = cw_attack(model, data.clone(), target)
   
    deepfool_slice = data.size(0)
    deepfool_data = deepfool_attack(model, data[:deepfool_slice].clone(), num_classes)
    deepfool_targets = target[:deepfool_slice]

    attack_dict = {
        "fgsm": (fgsm_data, target),
        "bim": (bim_data, target),
        "pgd": (pgd_data, target),
        "cw": (cw_data, target),
        "deepfool": (deepfool_data, deepfool_targets),
    }

    # ================= SAVE ONLY *SUCCESSFUL* ATTACK IMAGES =================
    for attack_name, (attack_images, attack_labels) in attack_dict.items():
        if per_attack_stats[attack_name]["saved"] >= TARGET_PER_ATTACK:
            continue  # this attack already hit its own quota -- don't let
                      # earlier attacks in the dict order eat deepfool's slots

        success_mask = attack_success_mask(model, attack_images, attack_labels)
        attack_attempts += len(success_mask)
        attack_successes += success_mask.sum().item()
        per_attack_stats[attack_name]["attempts"] += len(success_mask)
        per_attack_stats[attack_name]["successes"] += success_mask.sum().item()

        for img_tensor, succeeded in zip(attack_images, success_mask):
            if per_attack_stats[attack_name]["saved"] >= TARGET_PER_ATTACK:
                break
            if not succeeded:
                continue  # skip -- this perturbation didn't actually fool the model

            img = img_tensor.cpu().detach().numpy().transpose(1, 2, 0).clip(0, 1)
            plt.imsave(os.path.join(adv_path, f"{attack_name}_{adv_count}.png"), img)
            adv_count += 1
            per_attack_stats[attack_name]["saved"] += 1

print("\n==============================")
print(f"Normal images      : {normal_count}")
print(f"Adversarial images : {adv_count}")
if attack_attempts:
    print(f"Attack success rate: {100*attack_successes/attack_attempts:.1f}% "
          f"({attack_successes}/{attack_attempts} perturbations actually fooled the model)")
print("------------------------------")
print("Per-attack breakdown:")
for name, stats in per_attack_stats.items():
    if stats["attempts"] == 0:
        continue
    rate = 100 * stats["successes"] / stats["attempts"]
    print(f"  {name:9s} success rate: {rate:5.1f}%  ({stats['successes']}/{stats['attempts']})"
          f"  | saved: {stats['saved']}")
print("==============================")

for name in ATTACK_NAMES:
    saved = per_attack_stats[name]["saved"]
    if saved < TARGET_PER_ATTACK:
        print(f"⚠ {name} only reached {saved}/{TARGET_PER_ATTACK} -- ran out of "
              f"source data before enough attacks succeeded. Consider raising "
              f"epsilon (currently 0.1), more iters, or looping the dataset "
              f"more than once.")

print("Dataset generated successfully")
