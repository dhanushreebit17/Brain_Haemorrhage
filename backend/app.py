import base64
import io
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from torchvision import transforms

from gradcam import GradCAM, overlay_heatmap
from model import CLASS_NAMES, build_model, load_trained_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS_PATH = os.path.join("models", "haemorrhage_model.pth")
IMG_SIZE = 224
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)  # allow the frontend (served from file:// or a different port) to call this API

# ---------------------------------------------------------------------------
# Load model (falls back to ImageNet-pretrained, untrained-head weights if no
# fine-tuned checkpoint exists yet, so the app is runnable end-to-end even
# before you've trained on your dataset)
# ---------------------------------------------------------------------------
if os.path.exists(WEIGHTS_PATH):
    model = load_trained_model(WEIGHTS_PATH, device=DEVICE)
    print(f"[app] Loaded fine-tuned weights from {WEIGHTS_PATH}")
else:
    model = build_model(pretrained=True).to(DEVICE)
    model.eval()
    print("[app] WARNING: no fine-tuned weights found at "
          f"{WEIGHTS_PATH}. Serving an UNTRAINED classifier head — "
          "predictions will not be meaningful until you run train.py. "
          "The pipeline (upload -> preprocess -> predict -> Grad-CAM) "
          "still works end-to-end for demo purposes.")

# Target the last conv block of DenseNet's feature extractor for Grad-CAM
gradcam = GradCAM(model, target_layer=model.features)

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])


def read_image_from_upload(file_storage) -> Image.Image:
    img_bytes = file_storage.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return img


def encode_image_to_base64(bgr_img: np.ndarray) -> str:
    success, buffer = cv2.imencode(".png", bgr_img)
    if not success:
        raise RuntimeError("Could not encode image")
    return base64.b64encode(buffer).decode("utf-8")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "device": DEVICE,
        "model_trained": os.path.exists(WEIGHTS_PATH),
        "classes": CLASS_NAMES,
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use form field 'image'."}), 400

    file_storage = request.files["image"]
    if file_storage.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        pil_img = read_image_from_upload(file_storage)
    except Exception as e:
        return jsonify({"error": f"Could not read image: {e}"}), 400

    # Preprocess for the model
    input_tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)  # [1,3,224,224]

    # Prepare a resized BGR version of the original image for overlay display
    display_img = pil_img.resize((IMG_SIZE, IMG_SIZE))
    display_bgr = cv2.cvtColor(np.array(display_img), cv2.COLOR_RGB2BGR)

    # Run Grad-CAM (this also runs the forward+backward pass and gives us probs)
    cam, pred_idx, probs = gradcam.generate(input_tensor, class_idx=None)

    overlay = overlay_heatmap(display_bgr, cam, alpha=0.45)

    response = {
        "predicted_class": CLASS_NAMES[pred_idx],
        "predicted_index": pred_idx,
        "confidence": float(probs[pred_idx]),
        "class_probabilities": {
            CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
        },
        "original_image_base64": encode_image_to_base64(display_bgr),
        "heatmap_overlay_base64": encode_image_to_base64(overlay),
        "model_trained": os.path.exists(WEIGHTS_PATH),
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
