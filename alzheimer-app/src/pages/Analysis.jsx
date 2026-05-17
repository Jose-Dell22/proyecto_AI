import { useState } from "react";
import ImageUploader from "../components/ImageUploader";
import ProgressIndicator from "../components/ProgressIndicator";
import ResultsDisplay from "../components/ResultsDisplay";
import GradCamViewer from "../components/GradCamViewer";
import { useApp } from "../context/AppContext";
import { predictImage, parseApiError } from "../api/modelApi";
import { generatePdfReport } from "../utils/pdfReport";
import { ERROR_MESSAGES, ERROR_SUGGESTIONS } from "../utils/errors";
import "../styles/main.css";

const Analysis = () => {
  const {
    imageFile,
    imagePreview,
    prediction,
    gradcamImage,
    analysisDateTime,
    setValidatedImage,
    setPredictionResult,
    addNotification,
  } = useApp();

  const [isLoading, setIsLoading] = useState(false);

  const handleUploadError = (notification) => {
    addNotification(notification);
  };

  const handleValidImage = (file, previewUrl) => {
    setValidatedImage(file, previewUrl);
  };

  const handlePredict = async () => {
    if (!imageFile || isLoading) return;

    setIsLoading(true);

    try {
      const data = await predictImage(imageFile);

      setPredictionResult({
        prediction: {
          predicted_class: data.predicted_class,
          confidence: data.confidence,
          probabilities: data.probabilities,
        },
        gradcamImage: data.gradcam_image,
        analysisDateTime: new Date().toISOString(),
      });

      if (!data.gradcam_image) {
        addNotification({
          type: "warning",
          message: ERROR_MESSAGES.GRADCAM,
          suggestion: ERROR_SUGGESTIONS.GRADCAM,
        });
      }
    } catch (error) {
      const parsed = parseApiError(error);
      addNotification({
        type: parsed.type,
        message: parsed.message,
        suggestion: parsed.suggestion,
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownloadReport = async () => {
    if (!prediction || !analysisDateTime) return;

    try {
      await generatePdfReport({
        analysisDateTime,
        originalPreview: imagePreview,
        gradcamBase64: gradcamImage,
        prediction,
      });
    } catch {
      addNotification({
        type: "error",
        message: ERROR_MESSAGES.PDF,
        suggestion: ERROR_SUGGESTIONS.PDF,
      });
    }
  };

  return (
    <div className="medical-container">
      <ProgressIndicator visible={isLoading} />

      <header className="page-header">
        <h1 className="title">MRI Analysis</h1>
        <p className="subtitle">
          Upload a T1 axial brain MRI (JPG or PNG) for classification with DenseNet121 + CBAM.
        </p>
      </header>

      <div className="diagnostic-layout">
        <div className="image-panel">
          <div className="panel">
            <h3>Upload MRI Image</h3>
            <ImageUploader onValidImage={handleValidImage} onError={handleUploadError} />
          </div>

          <div className="panel">
            <h3>Preview</h3>
            {imagePreview ? (
              <img src={imagePreview} alt="MRI preview" className="preview-img" />
            ) : (
              <p className="empty-state">No image loaded</p>
            )}
          </div>

          {(gradcamImage || imagePreview) && prediction && (
            <div className="panel">
              <GradCamViewer
                originalPreview={imagePreview}
                gradcamBase64={gradcamImage}
                onError={handleUploadError}
              />
            </div>
          )}
        </div>

        <div className="result-panel">
          <div className="panel">
            <h3>Prediction</h3>
            <p className="model-info">Model: DenseNet121 + CBAM</p>

            <button
              type="button"
              className="predict-btn"
              onClick={handlePredict}
              disabled={!imageFile || isLoading}
            >
              {isLoading ? "Processing..." : "Predict"}
            </button>
          </div>

          {prediction && (
            <>
              <div className="panel">
                <ResultsDisplay prediction={prediction} />
              </div>

              <div className="panel actions-panel">
                <button
                  type="button"
                  className="report-btn"
                  onClick={handleDownloadReport}
                  disabled={!gradcamImage}
                  title={!gradcamImage ? "Grad-CAM map is required" : ""}
                >
                  Download Report
                </button>
                {!gradcamImage && (
                  <p className="hint-text">PDF report requires the Grad-CAM heatmap.</p>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default Analysis;
