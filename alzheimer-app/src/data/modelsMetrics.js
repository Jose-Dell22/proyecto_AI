/** Test-set metrics from metricas.md — DenseNet121 + CBAM is the active inference model. */

export const ACTIVE_MODEL = "DenseNet121 + CBAM";

export const MODELS_COMPARISON = [
  {
    name: "DenseNet121 + CBAM",
    accuracy: 99.38,
    macro_f1: 99.55,
    balanced_accuracy: 99.49,
    active: true,
  },
  {
    name: "EfficientNetV2S + CBAM",
    accuracy: 99.17,
    macro_f1: 99.43,
    balanced_accuracy: 99.54,
    active: false,
  },
  {
    name: "ResNet50 + CBAM",
    accuracy: 98.96,
    macro_f1: 99.24,
    balanced_accuracy: 99.17,
    active: false,
  },
  {
    name: "MobileNetV3-Large + CBAM",
    accuracy: 94.17,
    macro_f1: 95.77,
    balanced_accuracy: 95.02,
    active: false,
  },
];

export const DENSENET121_METRICS = {
  model: ACTIVE_MODEL,
  dataset: "Mendelei Alzheimer MRI Dataset (test set)",
  evaluation_date: "2025-11-15",
  accuracy: 99.38,
  macro_f1: 99.55,
  balanced_accuracy: 99.49,
  macro_average: {
    precision: 99.55,
    recall: 99.38,
    f1_score: 99.46,
  },
  per_class: [
    { class: "Non Demented", precision: 99.2, recall: 99.5, f1_score: 99.35 },
    { class: "Very Mild Dementia", precision: 99.1, recall: 99.0, f1_score: 99.05 },
    { class: "Mild Dementia", precision: 99.8, recall: 99.6, f1_score: 99.7 },
    { class: "Moderate Dementia", precision: 99.9, recall: 99.4, f1_score: 99.65 },
  ],
};

export const getDefaultMetricsPayload = () => ({
  ...DENSENET121_METRICS,
  models_comparison: MODELS_COMPARISON,
});
