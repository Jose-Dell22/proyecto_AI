import React from "react";

const ModelSelector = ({ model, setModel }) => {

  return (
    <select value={model} onChange={(e)=>setModel(e.target.value)}>
      <option value="DenseNet121">DenseNet121</option>
      <option value="EfficientNetV2S">EfficientNetV2S</option>
      <option value="MobileNetV3">MobileNetV3</option>
      <option value="ResNet50">ResNet50</option>
    </select>
  );
};

export default ModelSelector;