from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os

app = Flask(__name__)

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# ================= LOAD MODEL =================
model = DetectorCNN().to(device)
model.load_state_dict(torch.load("detector_model.pth", map_location=device))
model.eval()

# ================= CONFIDENCE CALIBRATION =================
# temperature.json is produced by detector.py's temperature-scaling step.
# Dividing logits by T before softmax is what turns "always 100%" into a
# realistic, evidence-based confidence. Falls back to T=1 (no change) if
# the file is missing, e.g. detector.py hasn't been re-run yet.
import json
try:
    with open("temperature.json") as f:
        TEMPERATURE = json.load(f)["temperature"]
    print(f"Loaded calibration temperature: {TEMPERATURE:.3f}")
except FileNotFoundError:
    TEMPERATURE = 1.0
    print("⚠ temperature.json not found -- confidence will be uncalibrated. "
          "Re-run detector.py to generate it.")

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ================= CLASS NAMES =================
classes = ["adversarial", "normal"]

# ================= HOME PAGE =================
@app.route("/", methods=["GET", "POST"])
def index():

    prediction = ""
    confidence = 0
    filename = ""

    if request.method == "POST":

        file = request.files.get("file")

        if file and file.filename:

            os.makedirs("static", exist_ok=True)
            filepath = os.path.join("static", file.filename)
            file.save(filepath)
            filename = file.filename

            try:
                img = Image.open(filepath).convert("RGB")
                img = transform(img)
                img = img.unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(img)
                    probs = torch.softmax(output / TEMPERATURE, dim=1)
                    conf, predicted = torch.max(probs, 1)

                confidence = round(conf.item() * 100, 2)
                prediction = classes[predicted.item()]

                # NOTE: the old hardcoded rule
                #   if prediction == "adversarial" and confidence < 75:
                #       prediction = "normal"
                # has been removed. It was silently relabeling most
                # true adversarial predictions as "normal" whenever the
                # model wasn't more than 75% confident -- which is the
                # main reason adversarial images were getting called
                # "normal". Trust the model's own argmax now. If you
                # still see false positives on clean images after
                # retraining with detector.py's class weighting, revisit
                # calibration (e.g. temperature scaling) rather than a
                # one-sided hard cutoff.

            except Exception as e:
                prediction = "error"
                confidence = 0
                print(f"Error processing {filename}: {e}")

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        filename=filename
    )

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)