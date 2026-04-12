import axios from "axios";

const API_URL = "http://localhost:5000";

export const predictModel = async (modelName, image) => {
  const formData = new FormData();
  formData.append("image", image);
  formData.append("model", modelName);

  const response = await axios.post(`${API_URL}/predict`, formData);

  return response.data;
};