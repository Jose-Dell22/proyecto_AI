import { CLASSES, CLASS_LABELS } from "../utils/constants";
import ProbabilitiesChart from "./ProbabilitiesChart";
import { ERROR_MESSAGES } from "../utils/errors";

const ResultsDisplay = ({ prediction }) => {
  if (!prediction) return null;

  const { predicted_class, confidence, probabilities } = prediction;

  const hasPartialData =
    !predicted_class ||
    confidence == null ||
    !probabilities ||
    CLASSES.some((c) => probabilities[c] == null);

  const rows = CLASSES.map((cls) => ({
    class: cls,
    label: CLASS_LABELS[cls],
    probability: probabilities?.[cls] ?? null,
    isPredicted: cls === predicted_class,
  }));

  return (
    <div className="results-display">
      {hasPartialData && (
        <p className="partial-warning" role="status">
          {ERROR_MESSAGES.PARTIAL_RESULTS}
        </p>
      )}

      <div className="predicted-summary">
        <h3>Analysis Result</h3>
        {predicted_class && (
          <p className="predicted-class-main">
            <strong>{predicted_class}</strong>
            <span className="class-note">({CLASS_LABELS[predicted_class]})</span>
          </p>
        )}
        {confidence != null && (
          <p className="confidence-value">
            Confidence: <strong>{Number(confidence).toFixed(2)}%</strong>
          </p>
        )}
      </div>

      <div className="probabilities-section">
        <h4>Class Probabilities</h4>

        <table className="prob-table" aria-label="Class probabilities">
          <thead>
            <tr>
              <th>Class</th>
              <th>Description</th>
              <th>Probability (%)</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr
                key={row.class}
                className={row.isPredicted ? "predicted-row" : ""}
              >
                <td>
                  <span className="class-name-cell">
                    {row.class}
                    {row.isPredicted && (
                      <span className="predicted-badge">Predicted</span>
                    )}
                  </span>
                </td>
                <td>{row.label}</td>
                <td>
                  {row.probability != null
                    ? `${Number(row.probability).toFixed(2)}`
                    : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        <ProbabilitiesChart probabilities={probabilities} predictedClass={predicted_class} />
      </div>
    </div>
  );
};

export default ResultsDisplay;
