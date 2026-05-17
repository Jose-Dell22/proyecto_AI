import axios from "axios";
import { getDefaultMetricsPayload } from "../data/modelsMetrics";
import { ERROR_MESSAGES, ERROR_SUGGESTIONS } from "../utils/errors";

const API_URL = process.env.REACT_APP_API_URL ?? "";

const api = axios.create({
  baseURL: API_URL,
  timeout: 15000,
});

export const predictImage = async (image) => {
  const formData = new FormData();
  formData.append("image", image);

  const response = await api.post("/predict", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });

  return response.data;
};

export const fetchMetrics = async () => {
  try {
    const response = await api.get("/metrics");
    return response.data;
  } catch {
    try {
      const fallback = await fetch("/metrics.json");
      if (fallback.ok) return fallback.json();
    } catch {
      /* use bundled defaults */
    }
    return getDefaultMetricsPayload();
  }
};

export const parseApiError = (error) => {
  if (error.response) {
    const status = error.response.status;
    const message = error.response.data?.message;

    if (status === 400 && message) {
      return {
        type: "validation",
        message,
        suggestion: "Check the selected file.",
      };
    }
    if (status === 500) {
      return {
        type: "server",
        message: message || ERROR_MESSAGES.SERVER,
        suggestion: ERROR_SUGGESTIONS.SERVER,
      };
    }
    return {
      type: "server",
      message: ERROR_MESSAGES.SERVER,
      suggestion: "Please try again.",
    };
  }

  if (error.request || error.code === "ECONNABORTED") {
    return {
      type: "connection",
      message: ERROR_MESSAGES.CONNECTION,
      suggestion: ERROR_SUGGESTIONS.CONNECTION,
    };
  }

  return {
    type: "internal",
    message: "Unexpected error. Please try again.",
    suggestion: ERROR_SUGGESTIONS.PAGE_LOAD,
  };
};
