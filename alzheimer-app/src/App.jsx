import { BrowserRouter, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";

import Home from "./pages/Home";
import Analysis from "./pages/analysis..jsx";
import Models from "./pages/models";

function App() {
  return (
    <BrowserRouter>

      <Navbar />

      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/analysis" element={<Analysis />} />
        <Route path="/models" element={<Models />} />
      </Routes>

    </BrowserRouter>
  );
}

export default App;