from flask import Flask, request, jsonify
from flask_cors import CORS

from utils.load_models import load_models
from utils.predict import predict_image

app = Flask(__name__)
CORS(app)

# cargar modelos al iniciar
models = load_models()

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]
    model_name = request.form.get("model")

    result = predict_image(models, model_name, image)

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)