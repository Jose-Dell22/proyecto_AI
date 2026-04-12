import { useState } from "react";
import { predictModel } from "../api/modelApi";

export const usePrediction = () => {

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const predict = async (model, image) => {

    if (!image) {
      setError("Debes subir una imagen");
      return;
    }

    try {
      setLoading(true);
      setError(null);

      const response = await predictModel(model, image);

      setResult(response.prediction);

    } catch (err) {

      setError("Error al procesar la imagen");

    } finally {

      setLoading(false);

    }
  };

  const resetPrediction = () => {
    setResult(null);
    setError(null);
  };

  return {
    result,
    loading,
    error,
    predict,
    resetPrediction
  };
};