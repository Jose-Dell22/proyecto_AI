import { useApp } from "../context/AppContext";
import "../styles/navbar.css";

const NAV_ITEMS = [
  { id: "home", label: "Home" },
  { id: "analysis", label: "Analysis" },
  { id: "model", label: "Model" },
];

const Navbar = () => {
  const { section, setSection } = useApp();

  return (
    <nav className="navbar" aria-label="Main navigation">
      <div className="navbar-logo">Alzheimer MRI Diagnosis</div>

      <ul className="navbar-links">
        {NAV_ITEMS.map(({ id, label }) => (
          <li key={id}>
            <button
              type="button"
              className={section === id ? "active" : ""}
              onClick={() => setSection(id)}
              aria-label={`Go to ${label}`}
              aria-current={section === id ? "page" : undefined}
            >
              {label}
            </button>
          </li>
        ))}
      </ul>
    </nav>
  );
};

export default Navbar;
