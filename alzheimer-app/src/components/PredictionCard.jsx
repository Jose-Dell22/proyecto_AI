import React from "react";

const PredictionCard = ({ result }) => {

  if (!result) return null;

  return (
    <div className="card">

      <h3>Diagnosis Result</h3>

      {/* Detected Class */}
      <p>
        <strong>Detected Class:</strong>{" "}
        {result.prediction}
      </p>

      {/* Confidence */}
      {result.confidence && (
        <p>
          <strong>Confidence:</strong>{" "}
          {(result.confidence * 100).toFixed(2)}%
        </p>
      )}

      {/* Probabilities */}
      {result.probabilities && (
        <div className="class-probabilities">

          <h4>Class Probabilities</h4>

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