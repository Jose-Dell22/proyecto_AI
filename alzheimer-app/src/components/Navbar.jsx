import { Link, useLocation } from "react-router-dom";
import "../styles/navbar.css";

const Navbar = () => {
  const location = useLocation();

  return (
    <nav className="navbar">

      <div className="navbar-logo">
        Vision AI
      </div>

      <ul className="navbar-links">

        <li>
          <Link to="/" className={location.pathname === "/" ? "active" : ""}>
            🏥 Home
          </Link>
        </li>

        <li>
          <Link to="/analysis" className={location.pathname === "/analysis" ? "active" : ""}>
            🔬 Analysis
          </Link>
        </li>

        <li>
          <Link to="/models" className={location.pathname === "/models" ? "active" : ""}>
            📈 Models
          </Link>
        </li>

      </ul>

    </nav>
  );
};

export default Navbar;