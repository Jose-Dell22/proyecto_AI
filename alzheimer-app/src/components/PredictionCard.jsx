import React from "react";

const PredictionCard = ({ result }) => {
  return (
    <div className="card">
      <h3>Resultado</h3>
      <p>{result}</p>
    </div>
  );
};

export default PredictionCard;