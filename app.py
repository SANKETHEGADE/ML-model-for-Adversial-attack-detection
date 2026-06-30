
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

model.load_state_dict(
    torch.load("detector_model.pth", map_location=device)
)

model.eval()

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

        file = request.files["file"]

        if file:

            os.makedirs("static", exist_ok=True)

            filepath = os.path.join("static", file.filename)

            file.save(filepath)

            filename = file.filename

            # LOAD IMAGE
            img = Image.open(filepath).convert("RGB")

            img = transform(img)

            img = img.unsqueeze(0).to(device)

            # PREDICT
            with torch.no_grad():

                output = model(img)

                probs = torch.softmax(output, dim=1)

                conf, predicted = torch.max(probs, 1)

            confidence = round(conf.item() * 100, 2)

            prediction = classes[predicted.item()]

            # Confidence threshold
            if prediction == "adversarial" and confidence < 75:
                prediction = "normal"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        filename=filename
    )

# ================= RUN =================
if __name__ == "__main__":

    app.run(debug=True)