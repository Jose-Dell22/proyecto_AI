import json
import logging
import os

from flask import Flask, jsonify, request
from flask_cors import CORS

from utils.load_models import load_models
from utils.predict import predict_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

_cors_origins = os.getenv("CORS_ORIGINS", "*")
CORS(app, resources={r"/*": {"origins": _cors_origins.split(",")}})

METRICS_PATH = os.path.join(os.path.dirname(__file__), "metrics.json")

model = None


def get_model():
    global model
    if model is None:
        model = load_models()
    return model


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "DenseNet121 + CBAM"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({
            "error_code": "MISSING_IMAGE",
            "message": "No image was received.",
        }), 400

    image = request.files["image"]

    if not image.filename:
        return jsonify({
            "error_code": "MISSING_IMAGE",
            "message": "No image was received.",
        }), 400

    try:
        result = predict_image(get_model(), image)
        return jsonify(result), 200
    except ValueError as exc:
        return jsonify({
            "error_code": "VALIDATION_ERROR",
            "message": str(exc),
        }), 400
    except Exception as exc:
        logger.exception("Error en inferencia: %s", exc)
        return jsonify({
            "error_code": "INTERNAL_ERROR",
            "message": "Error processing the image. Please contact the administrator.",
        }), 500


@app.route("/metrics", methods=["GET"])
def metrics():
    try:
        with open(METRICS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data), 200
    except Exception as exc:
        logger.exception("Error al cargar métricas: %s", exc)
        return jsonify({
            "error_code": "METRICS_UNAVAILABLE",
            "message": "Metrics are not available at this time.",
        }), 500


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
