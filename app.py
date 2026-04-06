from flask import Flask, render_template, request
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn

app = Flask(__name__)

# ================= MODEL =================

class DetectorCNN(nn.Module):
    def __init__(self):
        super().__init__()
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

# Load trained model
model = DetectorCNN()
model.load_state_dict(torch.load("detector_model.pth"))
model.eval()

# ================= IMAGE TRANSFORM =================

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ================= ROUTE =================

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    confidence = ""
    path = ""

    if request.method == "POST":
        file = request.files["file"]

        # Save image
        path = "static/" + file.filename
        file.save(path)

        # Process image
        img = Image.open(path).convert("RGB")
        img = transform(img).unsqueeze(0)

        # Prediction
        output = model(img)
        probs = torch.softmax(output, dim=1)
        conf, predicted = torch.max(probs, 1)

        confidence = round(conf.item() * 100, 2)

        # Result
        if predicted.item() == 0:
            result = "Adversarial Image 🚨"
        else:
            result = "Normal Image ✅"

    return render_template(
        "index.html",
        result=result,
        filename=path,
        confidence=confidence
    )

# ================= RUN =================

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)