import { AppProvider, useApp } from "./context/AppContext";
import Navbar from "./components/Navbar";
import ErrorNotification from "./components/ErrorNotification";
import Home from "./pages/Home";
import Analysis from "./pages/Analysis";
import Model from "./pages/Model";
import "./styles/main.css";

function AppContent() {
  const { section } = useApp();

  return (
    <>
      <Navbar />
      <ErrorNotification />
      <main>
        {section === "home" && <Home />}
        {section === "analysis" && <Analysis />}
        {section === "model" && <Model />}
      </main>
    </>
  );
}

function App() {
  return (
    <AppProvider>
      <AppContent />
    </AppProvider>
  );
}

export default App;
