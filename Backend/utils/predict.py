import base64
import logging
from io import BytesIO

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from utils.gradcam import GradCAM, overlay_gradcam

logger = logging.getLogger(__name__)

CLASSES = [
    "Non Demented",
    "Very Mild Dementia",
    "Mild Dementia",
    "Moderate Dementia",
]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def validate_image_file(image_file):
    filename = (image_file.filename or "").lower()
    ext = "." + filename.rsplit(".", 1)[-1] if "." in filename else ""

    if ext not in ALLOWED_EXTENSIONS:
        return False, "Unsupported format. Please use JPG or PNG."

    image_file.seek(0, 2)
    size = image_file.tell()
    image_file.seek(0)

    if size > MAX_FILE_SIZE:
        return False, "File exceeds the maximum size of 10 MB."

    try:
        image = Image.open(image_file).convert("RGB")
        image.verify()
        image_file.seek(0)
        Image.open(image_file).convert("RGB")
        image_file.seek(0)
    except Exception:
        return False, "Invalid or corrupted file."

    return True, None


def _to_base64(img_rgb):
    pil_img = Image.fromarray(img_rgb)
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def predict_image(model, image_file):
    valid, error_message = validate_image_file(image_file)
    if not valid:
        raise ValueError(error_message)

    model.eval()
    image = Image.open(image_file).convert("RGB")
    input_tensor = TRANSFORM(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)

    probs = torch.softmax(output, dim=1)
    confidence, pred = torch.max(probs, 1)
    predicted_class = CLASSES[pred.item()]

    probabilities = {
        CLASSES[i]: round(float(probs[0][i]) * 100, 2)
        for i in range(len(CLASSES))
    }

    gradcam_b64 = None
    try:
        target_layer = model.features.cbam
        gradcam = GradCAM(model, target_layer)
        cam = gradcam.generate(input_tensor, class_idx=pred.item())
        image_np = np.array(image.resize((224, 224)))
        overlay = overlay_gradcam(image_np, cam)
        gradcam_b64 = _to_base64(overlay)
    except Exception as exc:
        logger.warning("GradCAM error: %s", exc)

    final_confidence = round(float(confidence.item()) * 100, 2)

    return {
        "predicted_class": predicted_class,
        "confidence": final_confidence,
        "probabilities": probabilities,
        "gradcam_image": gradcam_b64,
    }
