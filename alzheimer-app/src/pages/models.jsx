import "../styles/models.css";

const Models = () => {

  const models = [
    { name: "DenseNet121", accuracy: "92%" },
    { name: "EfficienNetV25", accuracy: "94%" },
    { name: "MobileNetv3", accuracy: "93%" },
    { name: "ResNet50", accuracy: "95%" }
  ];

  return (
    <div className="models-container">

      <h2 className="models-title">Modelos disponibles</h2>

      <table className="models-table">

        <thead>
          <tr>
            <th>Modelo</th>
            <th>Accuracy</th>
          </tr>
        </thead>

        <tbody>
          {models.map((model, index) => (
            <tr key={index}>
              <td>{model.name}</td>
              <td>{model.accuracy}</td>
            </tr>
          ))}
        </tbody>

      </table>

    </div>
  );
};

export default Models;