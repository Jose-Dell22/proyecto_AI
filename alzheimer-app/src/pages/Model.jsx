import { useEffect, useState } from "react";
import { fetchMetrics } from "../api/modelApi";
import { useApp } from "../context/AppContext";
import { ACTIVE_MODEL } from "../data/modelsMetrics";
import { CLASSES } from "../utils/constants";
import { formatSessionDate } from "../utils/sessionMetrics";
import { ERROR_MESSAGES, ERROR_SUGGESTIONS } from "../utils/errors";
import "../styles/models.css";

const METRIC_TOOLTIPS = {
  accuracy: "Overall share of correct predictions on the test set.",
  macro_f1: "Macro-averaged F1 score across all classes.",
  balanced_accuracy: "Average of recall obtained on each class.",
  precision: "Share of positive predictions that are correct for each class.",
  recall: "Share of actual cases correctly detected per class.",
  f1_score: "Harmonic mean of precision and recall.",
};

const Model = () => {
  const { addNotification, densnetSessionMetrics } = useApp();
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);

  const session = densnetSessionMetrics;
  const hasSession = session.predictionCount > 0;

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      setLoading(true);
      try {
        const data = await fetchMetrics();
        if (!cancelled) setMetrics(data);
      } catch {
        if (!cancelled) {
          addNotification({
            type: "warning",
            message: ERROR_MESSAGES.METRICS_UNAVAILABLE,
            suggestion: ERROR_SUGGESTIONS.METRICS_UNAVAILABLE,
          });
          try {
            const res = await fetch("/metrics.json");
            if (res.ok && !cancelled) setMetrics(await res.json());
          } catch {
            if (!cancelled) setMetrics(null);
          }
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    load();
    return () => {
      cancelled = true;
    };
  }, [addNotification]);

  if (loading) {
    return (
      <div className="models-container">
        <p className="models-loading">Loading model metrics...</p>
      </div>
    );
  }

  if (!metrics) {
    return (
      <div className="models-container">
        <p className="metrics-unavailable">{ERROR_MESSAGES.METRICS_UNAVAILABLE}</p>
      </div>
    );
  }

  const comparison = metrics.models_comparison ?? [];

  return (
    <div className="models-container">
      <h2 className="models-title">Model Performance</h2>
      <p className="models-page-intro">
        Test-set benchmarks from evaluation runs. DenseNet121 + CBAM is used for
        inference; session metrics below refresh after each prediction.
      </p>

      <section className="metrics-section">
        <h3 className="section-heading">Model comparison (test set)</h3>
        <table className="models-table comparison-table" aria-label="Model comparison">
          <thead>
            <tr>
              <th>Model</th>
              <th title={METRIC_TOOLTIPS.accuracy}>Accuracy (%)</th>
              <th title={METRIC_TOOLTIPS.macro_f1}>Macro F1 (%)</th>
              <th title={METRIC_TOOLTIPS.balanced_accuracy}>Balanced accuracy (%)</th>
            </tr>
          </thead>
          <tbody>
            {comparison.map((row) => (
              <tr
                key={row.name}
                className={row.active || row.name === ACTIVE_MODEL ? "active-model-row" : ""}
              >
                <td>
                  {row.name}
                  {(row.active || row.name === ACTIVE_MODEL) && (
                    <span className="model-badge">In use</span>
                  )}
                </td>
                <td>{Number(row.accuracy).toFixed(2)}</td>
                <td>{Number(row.macro_f1).toFixed(2)}</td>
                <td>{Number(row.balanced_accuracy).toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="metrics-section">
        <h3 className="section-heading">{ACTIVE_MODEL} — test set</h3>
        <div className="model-meta">
          <p><strong>Dataset:</strong> {metrics.dataset}</p>
          <p><strong>Evaluation date:</strong> {metrics.evaluation_date}</p>
        </div>

        <div className="metrics-summary-grid">
          <div className="metric-card" title={METRIC_TOOLTIPS.accuracy}>
            <span className="metric-card-label">Accuracy</span>
            <span className="metric-card-value">{Number(metrics.accuracy).toFixed(2)}%</span>
          </div>
          <div className="metric-card" title={METRIC_TOOLTIPS.macro_f1}>
            <span className="metric-card-label">Macro F1</span>
            <span className="metric-card-value">
              {Number(metrics.macro_f1 ?? metrics.macro_average?.f1_score).toFixed(2)}%
            </span>
          </div>
          <div className="metric-card" title={METRIC_TOOLTIPS.balanced_accuracy}>
            <span className="metric-card-label">Balanced accuracy</span>
            <span className="metric-card-value">
              {Number(metrics.balanced_accuracy).toFixed(2)}%
            </span>
          </div>
        </div>

        <table className="models-table" aria-label="DenseNet121 per-class metrics">
          <thead>
            <tr>
              <th>Class</th>
              <th title={METRIC_TOOLTIPS.precision}>Precision (%)</th>
              <th title={METRIC_TOOLTIPS.recall}>Recall (%)</th>
              <th title={METRIC_TOOLTIPS.f1_score}>F1-Score (%)</th>
            </tr>
          </thead>
          <tbody>
            {metrics.per_class?.map((row) => (
              <tr key={row.class}>
                <td>{row.class}</td>
                <td>{Number(row.precision).toFixed(2)}</td>
                <td>{Number(row.recall).toFixed(2)}</td>
                <td>{Number(row.f1_score).toFixed(2)}</td>
              </tr>
            ))}
            {metrics.macro_average && (
              <tr className="macro-row">
                <td><strong>Macro-average</strong></td>
                <td><strong>{Number(metrics.macro_average.precision).toFixed(2)}</strong></td>
                <td><strong>{Number(metrics.macro_average.recall).toFixed(2)}</strong></td>
                <td><strong>{Number(metrics.macro_average.f1_score).toFixed(2)}</strong></td>
              </tr>
            )}
          </tbody>
        </table>
      </section>

      <section className="metrics-section session-section">
        <h3 className="section-heading">
          {ACTIVE_MODEL} — current session
          {hasSession && (
            <span className="session-updated">
              Updated {formatSessionDate(session.lastUpdated)}
            </span>
          )}
        </h3>

        {!hasSession ? (
          <p className="session-empty">
            Run a prediction on the Analysis tab to populate session metrics.
          </p>
        ) : (
          <>
            <div className="metrics-summary-grid session-grid">
              <div className="metric-card session-card">
                <span className="metric-card-label">Predictions</span>
                <span className="metric-card-value">{session.predictionCount}</span>
              </div>
              <div className="metric-card session-card">
                <span className="metric-card-label">Avg. confidence</span>
                <span className="metric-card-value">
                  {session.averageConfidence != null
                    ? `${session.averageConfidence.toFixed(2)}%`
                    : "—"}
                </span>
              </div>
              <div className="metric-card session-card highlight">
                <span className="metric-card-label">Last predicted class</span>
                <span className="metric-card-value metric-card-value-sm">
                  {session.lastInference?.predicted_class ?? "—"}
                </span>
                <span className="metric-card-sub">
                  {session.lastInference?.confidence != null
                    ? `${Number(session.lastInference.confidence).toFixed(2)}% confidence`
                    : ""}
                </span>
              </div>
            </div>

            <h4 className="subsection-heading">Class distribution (session)</h4>
            <table className="models-table" aria-label="Session class distribution">
              <thead>
                <tr>
                  <th>Class</th>
                  <th>Count</th>
                  <th>Share (%)</th>
                </tr>
              </thead>
              <tbody>
                {CLASSES.map((cls) => {
                  const count = session.classDistribution[cls] ?? 0;
                  const share =
                    session.predictionCount > 0
                      ? (count / session.predictionCount) * 100
                      : 0;
                  const isLast = session.lastInference?.predicted_class === cls;
                  return (
                    <tr key={cls} className={isLast ? "active-model-row" : ""}>
                      <td>{cls}</td>
                      <td>{count}</td>
                      <td>{share.toFixed(2)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>

            {session.recentPredictions.length > 0 && (
              <>
                <h4 className="subsection-heading">Recent inferences</h4>
                <table className="models-table recent-table" aria-label="Recent predictions">
                  <thead>
                    <tr>
                      <th>Time</th>
                      <th>Class</th>
                      <th>Confidence (%)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {session.recentPredictions.map((item) => (
                      <tr key={item.at}>
                        <td>{formatSessionDate(item.at)}</td>
                        <td>{item.predicted_class}</td>
                        <td>{Number(item.confidence).toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </>
            )}
          </>
        )}
      </section>
    </div>
  );
};

export default Model;
