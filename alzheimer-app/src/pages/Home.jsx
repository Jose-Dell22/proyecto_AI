import React from "react";
import { Link } from "react-router-dom";
import "../styles/Home.css";

const Inicio = () => {
  return (
    <div className="inicio-container">
      <div className="inicio-content">
        <div className="inicio-header">
          <h1 className="inicio-title">
            🏥 AI-Powered Diagnosis System
          </h1>
          <p className="inicio-subtitle">
            Alzheimer's disease assisted diagnosis using artificial intelligence
          </p>
        </div>

        <div className="inicio-features">
          <div className="feature-card">
            <div className="feature-icon">🧠</div>
            <h3>MRI Analysis</h3>
            <p>Advanced processing of brain magnetic resonance imaging</p>
          </div>

          <div className="feature-card">
            <div className="feature-icon">🤖</div>
            <h3>Advanced AI</h3>
            <p>DenseNet121 model optimized for early detection</p>
          </div>

          <div className="feature-card">
            <div className="feature-icon">📊</div>
            <h3>Precise Results</h3>
            <p>Detailed metrics and Grad-CAM visualization for interpretation</p>
          </div>
        </div>

        <div className="inicio-actions">
          <Link to="/analysis" className="action-button primary">
            🚀 Start Analysis
          </Link>
          <Link to="/models" className="action-button secondary">
            📈 View Models
          </Link>
        </div>

        <div className="inicio-info">
          <div className="info-section">
            <h4>🎯 How it works?</h4>
            <ol>
              <li>Upload a brain MRI image</li>
              <li>Our AI analyzes the image automatically</li>
              <li>Receive detailed results with metrics</li>
              <li>Visualize areas of interest with Grad-CAM</li>
            </ol>
          </div>

          <div className="info-section">
            <h4>⚡ Key Features</h4>
            <ul>
              <li>Fast and secure processing</li>
              <li>Professional medical interface</li>
              <li>Results with confidence and metrics</li>
              <li>Advanced result visualization</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Inicio;
