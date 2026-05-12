import "../styles/models.css";

const Models = () => {

  const models = [
    { name: "DenseNet121", accuracy: "92%", precision: "89%", recall: "91%" },
    { name: "EfficienNetV25", accuracy: "94%", precision: "92%", recall: "93%" },
    { name: "MobileNetv3", accuracy: "93%", precision: "90%", recall: "92%" },
    { name: "ResNet50", accuracy: "95%", precision: "93%", recall: "94%" }
  ].sort((a, b) => parseFloat(b.accuracy) - parseFloat(a.accuracy));

  return (
    <div className="models-container">

      <h2 className="models-title">Available Models</h2>

      <table className="models-table">

        <thead>
          <tr>
            <th>Model</th>
            <th>Accuracy</th>
            <th>Precision</th>
            <th>Recall</th>
          </tr>
        </thead>

        <tbody>
          {models.map((model, index) => (
            <tr key={index}>
              <td>{model.name}</td>
              <td>{model.accuracy}</td>
              <td>{model.precision}</td>
              <td>{model.recall}</td>
            </tr>
          ))}
        </tbody>

      </table>

    </div>
  );
};

export default Models;