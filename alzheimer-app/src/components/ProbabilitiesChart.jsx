import { CLASSES } from "../utils/constants";

const ProbabilitiesChart = ({ probabilities, predictedClass }) => {
  if (!probabilities) return null;

  const maxVal = Math.max(...CLASSES.map((c) => Number(probabilities[c]) || 0), 1);

  return (
    <div className="bar-chart" aria-label="Probability bar chart">
      <h5>Probability Distribution</h5>
      {CLASSES.map((cls) => {
        const value = Number(probabilities[cls]) || 0;
        const width = (value / maxVal) * 100;
        return (
          <div key={cls} className={`bar-row ${cls === predictedClass ? "bar-row-predicted" : ""}`}>
            <span className="bar-label" title={cls}>{cls}</span>
            <div className="bar-track">
              <div
                className="bar-fill"
                style={{ width: `${width}%` }}
                role="presentation"
              />
            </div>
            <span className="bar-value">{value.toFixed(2)}%</span>
          </div>
        );
      })}
    </div>
  );
};

export default ProbabilitiesChart;
