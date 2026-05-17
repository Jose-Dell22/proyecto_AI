import { createContext, useContext, useState, useCallback } from "react";
import {
  createInitialSessionMetrics,
  buildSessionMetrics,
} from "../utils/sessionMetrics";

const AppContext = createContext(null);

export function AppProvider({ children }) {
  const [section, setSection] = useState("home");
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [gradcamImage, setGradcamImage] = useState(null);
  const [analysisDateTime, setAnalysisDateTime] = useState(null);
  const [densnetSessionMetrics, setDensnetSessionMetrics] = useState(
    createInitialSessionMetrics
  );
  const [notifications, setNotifications] = useState([]);

  const addNotification = useCallback((notification) => {
    const id = Date.now() + Math.random();
    setNotifications((prev) => [...prev, { id, ...notification }]);

    if (!notification.persistent) {
      setTimeout(() => {
        setNotifications((prev) => prev.filter((n) => n.id !== id));
      }, notification.duration ?? 8000);
    }

    return id;
  }, []);

  const removeNotification = useCallback((id) => {
    setNotifications((prev) => prev.filter((n) => n.id !== id));
  }, []);

  const clearImage = useCallback(() => {
    if (imagePreview) URL.revokeObjectURL(imagePreview);
    setImageFile(null);
    setImagePreview(null);
    setPrediction(null);
    setGradcamImage(null);
    setAnalysisDateTime(null);
  }, [imagePreview]);

  const setValidatedImage = useCallback((file, previewUrl) => {
    if (imagePreview) URL.revokeObjectURL(imagePreview);
    setImageFile(file);
    setImagePreview(previewUrl);
    setPrediction(null);
    setGradcamImage(null);
    setAnalysisDateTime(null);
  }, [imagePreview]);

  const setPredictionResult = useCallback((data) => {
    const predictionPayload = data.prediction;
    setPrediction(predictionPayload);
    setGradcamImage(data.gradcamImage);
    setAnalysisDateTime(data.analysisDateTime || new Date().toISOString());

    if (predictionPayload) {
      setDensnetSessionMetrics((prev) =>
        buildSessionMetrics(prev, predictionPayload)
      );
    }
  }, []);

  const value = {
    section,
    setSection,
    imageFile,
    imagePreview,
    prediction,
    gradcamImage,
    analysisDateTime,
    densnetSessionMetrics,
    notifications,
    setValidatedImage,
    clearImage,
    setPredictionResult,
    addNotification,
    removeNotification,
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useApp must be used within AppProvider");
  return ctx;
}
