export const ERROR_MESSAGES = {
  UNSUPPORTED_FORMAT: "Unsupported format. Please use JPG or PNG.",
  FILE_TOO_LARGE: "File exceeds the maximum size of 10 MB.",
  INVALID_FILE: "Invalid or corrupted file.",
  CONNECTION: "Connection error with the server. Please try again.",
  SERVER: "Error processing the image. Please contact the administrator.",
  GRADCAM: "Could not generate the explanatory heatmap.",
  DOWNLOAD: "Could not generate the download. Please try again.",
  PDF: "Could not generate the report. Please try again.",
  PARTIAL_RESULTS: "Partial results. Some data is unavailable.",
  METRICS_UNAVAILABLE: "Metrics are not available at this time.",
  PAGE_LOAD: "Error loading the page. Please try again.",
};

export const ERROR_SUGGESTIONS = {
  UNSUPPORTED_FORMAT: "Select a JPG or PNG file.",
  FILE_TOO_LARGE: "Compress the image or choose a smaller file.",
  INVALID_FILE: "Verify that the file is not damaged.",
  CONNECTION: "Check your connection and try again.",
  SERVER: "If the problem persists, contact the administrator.",
  GRADCAM: "Classification results remain available.",
  DOWNLOAD: "Try again in a few seconds.",
  PDF: "Try again in a few seconds.",
  PARTIAL_RESULTS: "Review the available data with caution.",
  METRICS_UNAVAILABLE: "Try reloading the Model section.",
  PAGE_LOAD: "Reload the page if the problem continues.",
};
