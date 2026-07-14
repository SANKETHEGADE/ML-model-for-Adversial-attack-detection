# Medical AI: Disease Detection + Adversarial Attack Detection

This repo contains two connected projects exploring AI in medical imaging: an AI diagnostic tool that detects disease from medical scans, and a security layer that detects whether those scans have been adversarially manipulated.

## Why both?

CNNs used in medical imaging can be fooled by adversarial examples — inputs with small, often invisible perturbations that cause confident misclassification. Diagnostic accuracy alone isn't enough for real-world deployment; the system also needs to be able to tell when an input has been tampered with. This repo covers both halves: a working diagnostic model, and a defense layer that flags manipulated inputs.

---

## Part 1: MediScan AI — Disease Detection

A multi-scan diagnostic tool that classifies three types of medical images using transfer learning:

- **Chest X-Ray** → Pneumonia detection
- **Brain MRI** → Tumor detection
- **Skin Lesion** → 7-class classification (e.g. melanoma, basal cell carcinoma, benign keratosis) using the HAM10000 dataset

### Tech stack
- TensorFlow / Keras
- MobileNetV2 (transfer learning, frozen base + custom classification head)
- Gradio (web interface)
- scikit-learn (class weighting for imbalanced datasets)

### How it works
Each scan type has its own MobileNetV2-based model, trained separately with class-weighted loss to handle imbalanced medical datasets. All three models are loaded into a single Gradio interface where a user selects a scan type, uploads an image, and receives a diagnosis with a confidence score.

### How to run
```bash
pip install tensorflow gradio pandas scikit-learn pillow 

# Train all three models (chest, brain, skin)
python train_models.py

# Launch the diagnostic interface
python mediscan_app.py
```

---

## Part 2: NeuralShield — Adversarial Attack Detector

A security layer that detects whether a chest X-ray has been adversarially manipulated, with a custom-built web interface.

### Pipeline

1. **Base classifier** — A custom CNN (`main.py`) trained on chest X-ray images.
2. **Attack generation** — Five adversarial attack methods implemented from scratch to stress-test the classifier:
   - **FGSM** (Fast Gradient Sign Method) — single-step gradient attack
   - **BIM** (Basic Iterative Method) — iterative refinement of FGSM
   - **PGD** (Projected Gradient Descent) — iterative attack with perturbation projection
   - **C&W** (Carlini-Wagner L2 attack) — optimization-based attack using a tanh reparameterization
   - **DeepFool** — minimal-perturbation attack that pushes a sample just across the decision boundary
3. **Detector model** — A second CNN (`detector.py`) trained to classify "normal" vs "adversarial," evaluated via an 80/20 train-test split.
4. **Web app** (`app.py`) — A Flask backend with a custom dark-themed UI ("NeuralShield") featuring drag-and-drop upload, live confidence scoring, and attack-type indicators.

### Tech stack
- Python, PyTorch, Torchvision
- Flask
- ResNet18 (transfer learning variant used for attack generation in `cw_attack.py`)
- Custom HTML/CSS frontend

### Project structure

| File | Purpose |
|---|---|
| `main.py` | Trains the base CNN classifier; implements FGSM, BIM, PGD, and C&W attacks; generates adversarial dataset |
| `cw_attack.py` | ResNet18-based training pipeline with C&W and DeepFool attack generation |
| `deepfool_attack.py` | Standalone DeepFool attack implementation and visualization |
| `detector.py` | Trains the adversarial-vs-normal detector CNN on the combined dataset; reports test accuracy |
| `app.py` | Flask backend powering the NeuralShield web interface |
| `templates/index.html` | NeuralShield frontend — drag-and-drop upload, confidence visualization, attack-type chips |
| `adversarial_example.png` | Sample adversarial image output |

### How to run
```bash
pip install torch torchvision flask pillow matplotlib

python main.py        # train the classifier and generate adversarial images
python detector.py    # train the detector
python app.py          # launch NeuralShield
```

Open `http://localhost:5000`, upload a chest X-ray, and NeuralShield will report whether it's normal or adversarially manipulated, with a confidence score.

---

## Why this matters

As AI diagnostic tools like MediScan see wider real-world deployment, robustness against adversarial manipulation becomes critical for patient safety and clinical trust — a system that's accurate but easily fooled isn't safe to deploy. This repo demonstrates both halves of that problem: a working multi-disease diagnostic model, and a dedicated defense layer (five attack strategies, a trained detector, and a deployable interface) built to catch manipulated inputs before they reach a diagnosis.
