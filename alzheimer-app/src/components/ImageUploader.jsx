import { useRef } from "react";
import { ALLOWED_EXTENSIONS, MAX_FILE_SIZE } from "../utils/constants";
import { ERROR_MESSAGES, ERROR_SUGGESTIONS } from "../utils/errors";

const ImageUploader = ({ onValidImage, onError }) => {
  const inputRef = useRef(null);

  const validateFile = async (file) => {
    const name = file.name.toLowerCase();
    const ext = name.includes(".") ? name.slice(name.lastIndexOf(".")) : "";

    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      return { valid: false, message: ERROR_MESSAGES.UNSUPPORTED_FORMAT, suggestion: ERROR_SUGGESTIONS.UNSUPPORTED_FORMAT };
    }

    if (file.size > MAX_FILE_SIZE) {
      return { valid: false, message: ERROR_MESSAGES.FILE_TOO_LARGE, suggestion: ERROR_SUGGESTIONS.FILE_TOO_LARGE };
    }

    const isValidImage = await new Promise((resolve) => {
      const img = new Image();
      const url = URL.createObjectURL(file);

      img.onload = () => {
        URL.revokeObjectURL(url);
        resolve(true);
      };

      img.onerror = () => {
        URL.revokeObjectURL(url);
        resolve(false);
      };

      img.src = url;
    });

    if (!isValidImage) {
      return { valid: false, message: ERROR_MESSAGES.INVALID_FILE, suggestion: ERROR_SUGGESTIONS.INVALID_FILE };
    }

    return { valid: true };
  };

  const handleChange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const result = await validateFile(file);

    if (!result.valid) {
      onError?.({
        type: "validation",
        message: result.message,
        suggestion: result.suggestion,
      });
      if (inputRef.current) inputRef.current.value = "";
      return;
    }

    const previewUrl = URL.createObjectURL(file);
    onValidImage(file, previewUrl);

    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div className="upload-section">
      <input
        ref={inputRef}
        id="mri-upload"
        type="file"
        accept=".jpg,.jpeg,.png,image/jpeg,image/png"
        onChange={handleChange}
        className="upload-input-hidden"
      />
      <label htmlFor="mri-upload" className="upload-button">
        Upload Image
      </label>
      <p className="upload-hint">
        JPG or PNG only. Max 10 MB. T1 axial MRI sequence.
      </p>
    </div>
  );
};

export default ImageUploader;
