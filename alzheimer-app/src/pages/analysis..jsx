import React, { useState } from "react";
import ImageUploader from "../components/ImageUploader";
import PredictionCard from "../components/PredictionCard";
import { predictModel } from "../api/modelApi";
import "../styles/main.css";

const Analisis = () => {

  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [gradcam, setGradcam] = useState(null);

  const [result, setResult] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [showOriginal, setShowOriginal] = useState(false);

  const handleImageUpload = (file) => {
    setImage(file);
    setPreview(URL.createObjectURL(file));
  };

  const handlePredict = async () => {

    if (!image) {
      alert("Please upload an MRI image");
      return;
    }

    setResult(null);
    setGradcam(null);
    setMetrics(null);
    setIsLoading(true);

    try {
      const response = await predictModel("DenseNet121", image);

      setResult({
        prediction: response.prediction,
        confidence: response.confidence,
        probabilities: response.probabilities
      });

      setGradcam(response.gradcam);
      setMetrics(response.metrics);

    } catch (error) {
      console.error("Prediction error:", error);
      
      // Specific error handling
      if (error.response) {
        // Server error
        if (error.response.status === 400) {
          alert("Could not process the image. Please verify that the file is a valid MRI image.");
        } else if (error.response.status === 500) {
          alert("Model processing error. Please try again later.");
        } else {
          alert("Server error. Please try again.");
        }
      } else if (error.request) {
        // Connection error
        alert("Connection error with server. Please check your internet connection.");
      } else {
        // Unexpected error
        alert("Unexpected error. Please reload the page and try again.");
      }
    } finally {
      setIsLoading(false);
    }

  };

  return (

    <div className="medical-container">

      <h1 className="title">
        🔬 Diagnosis Analysis
      </h1>

      <div className="diagnostic-layout">

        {/* IMAGE COLUMN */}
        <div className="image-panel">

          <div className="panel">
            <h3>📤 Upload MRI</h3>
            <ImageUploader setImage={handleImageUpload} />
          </div>

          <div className="panel">
            <h3>🧠 Original Image</h3>
            {preview ? (
              <img src={preview} alt="MRI preview" className="preview-img" />
            ) : (
              <p>📷 No image loaded</p>
            )}
          </div>

          <div className="panel">
            <h3>🔬 Grad-CAM</h3>
            {gradcam ? (
              <div>
                <div className="toggle-container">
                  <button onClick={() => setShowOriginal(!showOriginal)}>
                    {showOriginal ? "🔬 View Grad-CAM" : "🧠 View Original"}
                  </button>
                </div>
                <img src={showOriginal ? preview : gradcam} alt={showOriginal ? "original" : "gradcam"} className="preview-img" />
              </div>
            ) : (
              <p>⏳ Run a prediction</p>
            )}
          </div>

        </div>

        {/* RESULTS COLUMN */}
        <div className="result-panel">

          <div className="panel">
            <h3>🤖 Model</h3>
            <p><strong>⚡ DenseNet121</strong></p>
            <p className="model-info">🎯 Model optimized for Alzheimer's diagnosis</p>
          </div>

          <div className="panel">

            <h3>🔍 Prediction</h3>

            <button onClick={handlePredict} disabled={!image || isLoading} className={isLoading ? "loading" : ""}>
              {isLoading ? "⚕️ Processing..." : "🚀 Run Diagnosis"}
            </button>

            {result && (

              <div className="prediction-result">

                <PredictionCard result={result} />

                <p className="detected-class">
                  <strong>🎯 Detected Class:</strong> {result.prediction}
                </p>

                {result.confidence && (
                  <p>
                    <strong>📊 Confidence:</strong>{" "}
                    {(result.confidence * 100).toFixed(2)}%
                  </p>
                )}

              </div>

            )}

          </div>

          {metrics && (

            <div className="panel">

              <h3>📈 Model Metrics</h3>

              <table className="metrics-table">
                <tbody>
                  <tr>
                    <td>✅ Accuracy</td>
                    <td>{metrics.accuracy}</td>
                  </tr>
                  <tr>
                    <td>🎯 Precision</td>
                    <td>{metrics.precision}</td>
                  </tr>
                  <tr>
                    <td>🔄 Recall</td>
                    <td>{metrics.recall}</td>
                  </tr>
                </tbody>
              </table>

            </div>

          )}

        </div>

      </div>

    </div>

  );

};

export default Analisis;
