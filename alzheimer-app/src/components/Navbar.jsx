import { Link } from "react-router-dom";
import "../styles/navbar.css";

const Navbar = () => {
  return (
    <nav className="navbar">

      <div className="navbar-logo">
        Vision AI
      </div>

      <ul className="navbar-links">

        <li>
          <Link to="/">Inicio</Link>
        </li>

        <li>
          <Link to="/models">Modelos</Link>
        </li>

      </ul>

    </nav>
  );
};

export default Navbar;