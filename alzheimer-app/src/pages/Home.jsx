import { useApp } from "../context/AppContext";
import "../styles/home.css";

const FEATURES = [
  {
    index: "01",
    title: "MRI Analysis",
    description: "Upload and validate T1 axial brain MRI images for automated review.",
  },
  {
    index: "02",
    title: "DenseNet121 + CBAM",
    description: "Classification powered by a convolutional network with attention modules.",
  },
  {
    index: "03",
    title: "Explainable Output",
    description: "Class probabilities, confidence scores, and Grad-CAM heatmaps for interpretation.",
  },
];

const Home = () => {
  const { setSection } = useApp();

  return (
    <div className="home-container">
      <div className="home-content">
        <header className="home-header">
          <p className="home-eyebrow">Clinical decision support</p>
          <h1 className="home-title">Alzheimer MRI Diagnosis</h1>
          <p className="home-subtitle">
            Assisted screening from structural MRI using deep learning. Results support
            clinical review and do not replace professional judgment.
          </p>
        </header>

        <div className="home-features">
          {FEATURES.map((feature) => (
            <article key={feature.index} className="feature-card">
              <span className="feature-index">{feature.index}</span>
              <h3>{feature.title}</h3>
              <p>{feature.description}</p>
            </article>
          ))}
        </div>

        <div className="home-actions">
          <button
            type="button"
            className="action-button primary"
            onClick={() => setSection("analysis")}
          >
            Start Analysis
          </button>
          <button
            type="button"
            className="action-button secondary"
            onClick={() => setSection("model")}
          >
            Model Performance
          </button>
        </div>

        <div className="home-info">
          <section className="info-section">
            <h4>Workflow</h4>
            <ol>
              <li>Upload a T1 axial brain MRI (JPG or PNG, max 10 MB)</li>
              <li>Run prediction with the DenseNet121 + CBAM model</li>
              <li>Review predicted class, confidence, and per-class probabilities</li>
              <li>Inspect Grad-CAM regions and export reports if needed</li>
            </ol>
          </section>

          <section className="info-section">
            <h4>Capabilities</h4>
            <ul>
              <li>Server-side inference with structured JSON responses</li>
              <li>Input validation and clear error messaging</li>
              <li>PDF report and PNG heatmap download</li>
              <li>Published test-set metrics for model transparency</li>
            </ul>
          </section>
        </div>
      </div>
    </div>
  );
};

export default Home;
