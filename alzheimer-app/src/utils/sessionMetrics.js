import { CLASSES } from "./constants";

const emptyClassCounts = () =>
  CLASSES.reduce((acc, cls) => ({ ...acc, [cls]: 0 }), {});

export const createInitialSessionMetrics = () => ({
  predictionCount: 0,
  lastUpdated: null,
  lastInference: null,
  averageConfidence: null,
  classDistribution: emptyClassCounts(),
  recentPredictions: [],
});

export const buildSessionMetrics = (previous, prediction) => {
  const { predicted_class, confidence, probabilities } = prediction;
  const count = previous.predictionCount + 1;
  const totalConfidence =
    (previous.averageConfidence ?? 0) * previous.predictionCount + Number(confidence);
  const classDistribution = {
    ...previous.classDistribution,
    [predicted_class]: (previous.classDistribution[predicted_class] ?? 0) + 1,
  };

  const entry = {
    predicted_class,
    confidence: Number(confidence),
    probabilities,
    at: new Date().toISOString(),
  };

  return {
    predictionCount: count,
    lastUpdated: entry.at,
    lastInference: entry,
    averageConfidence: totalConfidence / count,
    classDistribution,
    recentPredictions: [entry, ...previous.recentPredictions].slice(0, 10),
  };
};

export const formatSessionDate = (iso) => {
  if (!iso) return "—";
  return new Date(iso).toLocaleString("en-US", {
    dateStyle: "medium",
    timeStyle: "short",
  });
};
