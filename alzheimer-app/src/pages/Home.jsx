import React, { useState } from "react";
import ImageUploader from "../components/ImageUploader";
import PredictionCard from "../components/PredictionCard";
import { predictModel } from "../api/modelApi";
import "../styles/home.css";

const Home = () => {

  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [gradcam, setGradcam] = useState(null);

  const [result, setResult] = useState(null);
  const [metrics, setMetrics] = useState(null);

  const handleImageUpload = (file) => {
    setImage(file);
    setPreview(URL.createObjectURL(file));
  };

  const handlePredict = async () => {

    if (!image) {
      alert("Por favor suba una imagen MRI");
      return;
    }

    setResult(null);
    setGradcam(null);
    setMetrics(null);

    try {

      // 🔥 Modelo fijo
      const response = await predictModel("DenseNet121", image);

      setResult({
        prediction: response.prediction,
        confidence: response.confidence,
        probabilities: response.probabilities
      });

      setGradcam(response.gradcam);
      setMetrics(response.metrics);

    } catch (error) {
      console.error("Error en predicción:", error);
      alert("Error conectando con el backend");
    }

  };

  return (

    <div className="medical-container">

      <h1 className="title">
        Diagnóstico asistido por IA
      </h1>

      <div className="diagnostic-layout">

        {/* COLUMNA IMÁGENES */}
        <div className="image-panel">

          <div className="panel">
            <h3>Subir MRI</h3>
            <ImageUploader setImage={handleImageUpload} />
          </div>

          <div className="panel">
            <h3>Imagen original</h3>
            {preview ? (
              <img src={preview} alt="MRI preview" className="preview-img" />
            ) : (
              <p>No hay imagen cargada</p>
            )}
          </div>

          <div className="panel">
            <h3>Grad-CAM</h3>
            {gradcam ? (
              <img src={gradcam} alt="gradcam" className="preview-img" />
            ) : (
              <p>Ejecute una predicción</p>
            )}
          </div>

        </div>

        {/* COLUMNA RESULTADOS */}
        <div className="result-panel">

          <div className="panel">
            <h3>Modelo</h3>
            <p><strong>DenseNet121</strong></p>
          </div>

          <div className="panel">

            <h3>Predicción</h3>

            <button onClick={handlePredict}>
              Ejecutar diagnóstico
            </button>

            {result && (

              <div className="prediction-result">

                <PredictionCard result={result} />

                <p className="detected-class">
                  <strong>Clase detectada:</strong> {result.prediction}
                </p>

                {result.confidence && (
                  <p>
                    <strong>Confianza:</strong>{" "}
                    {(result.confidence * 100).toFixed(2)}%
                  </p>
                )}

              </div>

            )}

          </div>

          {metrics && (

            <div className="panel">

              <h3>Métricas del modelo</h3>

              <table className="metrics-table">
                <tbody>
                  <tr>
                    <td>Accuracy</td>
                    <td>{metrics.accuracy}</td>
                  </tr>
                  <tr>
                    <td>Precision</td>
                    <td>{metrics.precision}</td>
                  </tr>
                  <tr>
                    <td>Recall</td>
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

export default Home;