import React from "react";

const PredictionCard = ({ result }) => {

  if (!result) return null;

  return (
    <div className="card">

      <h3>Resultado del diagnóstico</h3>

      {/* Clase detectada */}
      <p>
        <strong>Clase detectada:</strong>{" "}
        {result.prediction}
      </p>

      {/* Confianza */}
      {result.confidence && (
        <p>
          <strong>Confianza:</strong>{" "}
          {(result.confidence * 100).toFixed(2)}%
        </p>
      )}

      {/* Probabilidades */}
      {result.probabilities && (
        <div className="class-probabilities">

          <h4>Probabilidades por clase</h4>

          <table className="prob-table">
            <tbody>

              {Object.entries(result.probabilities).map(
                ([cls, prob]) => (

                  <tr key={cls}>

                    <td>{cls}</td>

                    <td>
                      {(prob * 100).toFixed(2)}%
                    </td>

                  </tr>

                )
              )}

            </tbody>
          </table>

        </div>
      )}

    </div>
  );
};

export default PredictionCard;