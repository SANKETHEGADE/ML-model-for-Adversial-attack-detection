# ML Model for Adversarial Attack Detection

A PyTorch-based system that classifies chest X-ray images, attacks its own classifier using five adversarial methods, then trains a second model to detect those attacks. Deployed as a Flask web app for real-time detection.

## Overview

CNNs used in medical imaging can be fooled by adversarial examples — inputs with small, often invisible perturbations that cause confident misclassification. This project demonstrates that vulnerability directly and builds a practical defense against it.

## Pipeline

1. **Base classifier** — A custom CNN (`main.py`) trained on chest X-ray images to classify between two classes.
2. **Attack generation** — Five adversarial attack methods implemented from scratch to fool the classifier:
   - **FGSM** (Fast Gradient Sign Method) — single-step gradient attack
   - **BIM** (Basic Iterative Method) — iterative refinement of FGSM
   - **PGD** (Projected Gradient Descent) — iterative attack with perturbation projection
   - **C&W** (Carlini-Wagner L2 attack) — optimization-based attack using a tanh reparameterization
   - **DeepFool** — minimal-perturbation attack that pushes a sample just across the decision boundary
3. **Detector model** — A second CNN (`detector.py`) trained on a mixed dataset of clean and adversarial images to classify "normal" vs "adversarial," achieving real measured test accuracy via an 80/20 train-test split.
4. **Web app** (`app.py`) — A Flask interface where a user uploads an image and gets a real-time prediction with a confidence score, including a confidence threshold to reduce false positives.

## Tech stack

- Python, PyTorch, Torchvision
- Flask
- ResNet18 (transfer learning variant used for attack generation in `cw_attack.py`)
- Matplotlib (visualization), PIL

## Project structure

| File | Purpose |
|---|---|
| `main.py` | Trains the base CNN classifier; implements FGSM, BIM, PGD, and C&W attacks; generates adversarial dataset |
| `cw_attack.py` | ResNet18-based training pipeline with C&W and DeepFool attack generation |
| `deepfool_attack.py` | Standalone DeepFool attack implementation and visualization |
| `detector.py` | Trains the adversarial-vs-normal detector CNN on the combined dataset; reports test accuracy |
| `app.py` | Flask web app — upload an image, get a real-time classification with confidence score |
| `adversarial_example.png` | Sample adversarial image output |

## Sample output

The repo includes an example of a BIM-perturbed chest X-ray fed through the pipeline, labeled "BIM Example."

## How to run

```bash
pip install torch torchvision flask pillow matplotlib

# 1. Train the base classifier and generate adversarial images
python main.py

# 2. Train the detector on the resulting dataset
python detector.py

# 3. Launch the web app
python app.py
```

Then open `http://localhost:5000` in your browser, upload a chest X-ray image, and the app will report whether it's normal or adversarial along with a confidence score.

## Why this matters

As medical AI tools see wider deployment, robustness against adversarial manipulation is critical for patient safety and clinical trust. This project explores both the attack surface (five distinct attack strategies) and a practical, deployable defense.
