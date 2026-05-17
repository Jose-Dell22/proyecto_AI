import { useApp } from "../context/AppContext";
import "../styles/notifications.css";

const ErrorNotification = () => {
  const { notifications, removeNotification } = useApp();

  if (!notifications.length) return null;

  return (
    <div className="notifications-container" aria-live="polite">
      {notifications.map((n) => (
        <div
          key={n.id}
          className={`notification notification-${n.type || "error"}`}
          role="alert"
        >
          <div className="notification-content">
            <p className="notification-message">{n.message}</p>
            {n.suggestion && (
              <p className="notification-suggestion">{n.suggestion}</p>
            )}
            {n.critical && (
              <button
                type="button"
                className="notification-reload"
                onClick={() => window.location.reload()}
              >
                Reload Page
              </button>
            )}
          </div>
          <button
            type="button"
            className="notification-close"
            onClick={() => removeNotification(n.id)}
            aria-label="Close notification"
          >
            ×
          </button>
        </div>
      ))}
    </div>
  );
};

export default ErrorNotification;
