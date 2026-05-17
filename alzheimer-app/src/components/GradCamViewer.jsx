import { useState } from "react";
import { ERROR_MESSAGES, ERROR_SUGGESTIONS } from "../utils/errors";

const toDataUrl = (base64) => {
  if (!base64) return null;
  if (base64.startsWith("data:")) return base64;
  return `data:image/png;base64,${base64}`;
};

const GradCamViewer = ({ originalPreview, gradcamBase64, onError }) => {
  const [showOriginal, setShowOriginal] = useState(false);

  if (!gradcamBase64 && !originalPreview) return null;

  if (!gradcamBase64) {
    return (
      <p className="gradcam-error" role="alert">
        {ERROR_MESSAGES.GRADCAM}
      </p>
    );
  }

  const gradcamUrl = toDataUrl(gradcamBase64);
  const displaySrc = showOriginal ? originalPreview : gradcamUrl;

  const handleDownload = () => {
    try {
      const link = document.createElement("a");
      link.href = gradcamUrl;
      link.download = "gradcam_result.png";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch {
      onError?.({
        type: "error",
        message: ERROR_MESSAGES.DOWNLOAD,
        suggestion: ERROR_SUGGESTIONS.DOWNLOAD,
      });
    }
  };

  return (
    <div className="gradcam-viewer">
      <h3>Grad-CAM Heatmap</h3>
      <p className="gradcam-note">
        Warm regions (red/yellow) had the greatest influence on the prediction.
        JET colormap, 0.4 overlay opacity.
      </p>

      <div className="toggle-container">
        <button
          type="button"
          onClick={() => setShowOriginal((v) => !v)}
          aria-pressed={showOriginal}
        >
          {showOriginal ? "View Grad-CAM" : "View Original Image"}
        </button>
      </div>

      {displaySrc && (
        <img
          src={displaySrc}
          alt={showOriginal ? "Original MRI" : "MRI with Grad-CAM overlay"}
          className="preview-img gradcam-img"
        />
      )}

      {!showOriginal && (
        <button type="button" className="download-btn" onClick={handleDownload}>
          Download Map
        </button>
      )}
    </div>
  );
};

export default GradCamViewer;
