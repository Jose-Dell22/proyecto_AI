import "../styles/progress.css";

const ProgressIndicator = ({ visible, label = "Running prediction..." }) => {
  if (!visible) return null;

  return (
    <div className="progress-overlay" role="status" aria-live="polite">
      <div className="progress-card">
        <div className="progress-spinner" aria-hidden="true" />
        <p>{label}</p>
      </div>
    </div>
  );
};

export default ProgressIndicator;
