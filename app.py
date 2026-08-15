from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import io
import base64
import json
import qrcode

from blockchain import Blockchain, sha256_of_bytes

app = Flask(__name__)

# ================= BLOCKCHAIN =================
# Certifies ORIGINAL X-rays. Separate, deterministic check from the CNN
# below -- see blockchain.py's module docstring for why both exist.
blockchain = Blockchain()

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


# ================= SINGLE-PAGE APP =================
# Everything -- detection, registration, and the chain audit log -- lives
# on one page (templates/index.html) as tab sections, all served from this
# one route. `action` (a hidden field on each form) tells us which form was
# submitted; `active_tab` is echoed back so the page reloads showing the
# section the user was just working in, instead of snapping back to tab 1.
@app.route("/", methods=["GET", "POST"])
def index():

    active_tab = "detect"

    # ---- detect tab state ----
    prediction = ""
    confidence = 0
    filename = ""
    blockchain_status = ""
    blockchain_class = ""

    # ---- register tab state ----
    register_error = ""
    block_info = None
    qr_data_uri = None

    if request.method == "POST":
        action = request.form.get("action")

        # ================= DETECT + VERIFY =================
        if action == "detect":
            active_tab = "detect"
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

                    # Deterministic check: does this exact file match a
                    # previously-certified original? Independent of what the
                    # CNN above thinks -- a single changed pixel here would
                    # produce a completely different hash (avalanche effect),
                    # so this catches tampering the CNN might miss.
                    with open(filepath, "rb") as f:
                        image_hash = sha256_of_bytes(f.read())

                    match = blockchain.find_by_image_hash(image_hash)
                    if match:
                        blockchain_status = (
                            f"Verified — matches certified original. "
                            f"Block #{match.index} | Patient: {match.patient_id} | "
                            f"Disease: {match.disease}"
                        )
                        blockchain_class = "verified"
                    else:
                        blockchain_status = (
                            "Not found on-chain — this image was never registered "
                            "as an original, or it has been modified since "
                            "registration. Authenticity cannot be confirmed."
                        )
                        blockchain_class = "unverified"

                    # NOTE: the old hardcoded rule
                    #   if prediction == "adversarial" and confidence < 75:
                    #       prediction = "normal"
                    # has been removed. It was silently relabeling most
                    # true adversarial predictions as "normal" whenever the
                    # model wasn't more than 75% confident. Trust the
                    # model's own argmax now.

                except Exception as e:
                    prediction = "error"
                    confidence = 0
                    print(f"Error processing {filename}: {e}")

        # ================= REGISTER ORIGINAL X-RAY =================
        elif action == "register":
            active_tab = "register"
            patient_id = request.form.get("patient_id", "").strip()
            disease = request.form.get("disease", "").strip()
            file = request.files.get("file")

            if not patient_id or not disease or not file or not file.filename:
                register_error = "Patient ID, disease, and an X-ray image are all required."
            else:
                image_bytes = file.read()
                image_hash = sha256_of_bytes(image_bytes)

                existing = blockchain.find_by_image_hash(image_hash)
                if existing:
                    register_error = (
                        f"This exact image is already registered on-chain "
                        f"(Block #{existing.index}, Patient {existing.patient_id}). "
                        f"Skipping duplicate registration."
                    )
                else:
                    block = blockchain.add_record(patient_id, disease, image_hash)

                    # keep a copy of the certified original for auditing later
                    os.makedirs("static/originals", exist_ok=True)
                    orig_path = os.path.join(
                        "static/originals", f"block{block.index}_{patient_id}.png"
                    )
                    with open(orig_path, "wb") as f:
                        f.write(image_bytes)

                    qr_payload = json.dumps({
                        "block": block.index,
                        "patient_id": patient_id,
                        "disease": disease,
                        "image_hash": image_hash,
                    })
                    qr_img = qrcode.make(qr_payload)
                    buf = io.BytesIO()
                    qr_img.save(buf, format="PNG")
                    qr_data_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

                    block_info = block.to_dict()

        # ================= SWITCH TO CHAIN TAB (no form submit) =================
        elif action == "view_chain":
            active_tab = "chain"

    # Chain log data is small (local demo chain) so it's cheap to always
    # compute and pass along -- the chain tab can render instantly on
    # every page load without a separate request.
    chain_valid, chain_reason = blockchain.is_chain_valid()
    chain_data = list(reversed(blockchain.chain))

    return render_template(
        "index.html",
        active_tab=active_tab,
        prediction=prediction,
        confidence=confidence,
        filename=filename,
        blockchain_status=blockchain_status,
        blockchain_class=blockchain_class,
        register_error=register_error,
        block_info=block_info,
        qr_data_uri=qr_data_uri,
        chain=chain_data,
        chain_valid=chain_valid,
        chain_reason=chain_reason,
    )

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)