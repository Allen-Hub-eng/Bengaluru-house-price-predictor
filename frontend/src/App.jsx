import { useEffect, useState } from "react";
import api, { getErrorMessage } from "./api";

// --------------------------------------------------
// HELPER FUNCTIONS
// --------------------------------------------------

function formatPrice(value) {
  const number = Number(value);

  if (!Number.isFinite(number)) {
    return "Not available";
  }

  return number.toLocaleString("en-IN", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

function formatDate(value) {
  if (!value) {
    return "Date unavailable";
  }

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleString("en-IN", {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

// --------------------------------------------------
// MAIN APPLICATION
// --------------------------------------------------

export default function App() {
  const [page, setPage] = useState("login");

  const [user, setUser] = useState(null);

  const [authLoading, setAuthLoading] = useState(true);

  const [authMode, setAuthMode] = useState("login");

  const [authForm, setAuthForm] = useState({
    username: "",
    email: "",
    password: "",
  });

  const [houseForm, setHouseForm] = useState({
    location: "",
    total_sqft: "",
    bhk: "2",
    bath: "2",
    balcony: "",
  });

  const [prediction, setPrediction] = useState(null);

  const [history, setHistory] = useState([]);

  const [loading, setLoading] = useState(false);

  const [historyLoading, setHistoryLoading] = useState(false);

  const [message, setMessage] = useState("");

  const [error, setError] = useState("");

  // ------------------------------------------------
  // CHECK EXISTING LOGIN SESSION
  // ------------------------------------------------

  useEffect(() => {
    async function restoreSession() {
      const token = localStorage.getItem("access_token");

      if (!token) {
        setAuthLoading(false);
        return;
      }

      try {
        const response = await api.get("/auth/me");

        setUser(response.data);
        setPage("dashboard");
      } catch {
        localStorage.removeItem("access_token");
        setUser(null);
        setPage("login");
      } finally {
        setAuthLoading(false);
      }
    }

    restoreSession();
  }, []);

  // ------------------------------------------------
  // HANDLE AUTH FORM CHANGES
  // ------------------------------------------------

  function handleAuthChange(event) {
    const { name, value } = event.target;

    setAuthForm((previous) => ({
      ...previous,
      [name]: value,
    }));
  }

  // ------------------------------------------------
  // REGISTER OR LOGIN
  // ------------------------------------------------

  async function handleAuthSubmit(event) {
    event.preventDefault();

    setError("");
    setMessage("");
    setLoading(true);

    try {
      if (authMode === "register") {
        await api.post("/auth/register", {
          username: authForm.username.trim(),
          email: authForm.email.trim(),
          password: authForm.password,
        });

        setMessage(
          "Registration successful! Please log in with your new account."
        );

        setAuthMode("login");

        setAuthForm((previous) => ({
          ...previous,
          password: "",
        }));

        setLoading(false);
        return;
      }

      const response = await api.post("/auth/login", {
        email: authForm.email.trim(),
        password: authForm.password,
      });

      const token = response.data.access_token;

      if (!token) {
        throw new Error("The login response did not contain an access token.");
      }

      localStorage.setItem("access_token", token);

      const userResponse = await api.get("/auth/me");

      setUser(userResponse.data);

      setPage("dashboard");

      setPrediction(null);

      setMessage("You have successfully logged in.");

      setAuthForm({
        username: "",
        email: "",
        password: "",
      });

      await loadHistory();
    } catch (requestError) {
      localStorage.removeItem("access_token");

      setError(
        requestError.response
          ? getErrorMessage(requestError)
          : requestError.message || getErrorMessage(requestError)
      );
    } finally {
      setLoading(false);
    }
  }

  // ------------------------------------------------
  // LOGOUT
  // ------------------------------------------------

  function handleLogout() {
    localStorage.removeItem("access_token");

    setUser(null);
    setPrediction(null);
    setHistory([]);

    setHouseForm({
      location: "",
      total_sqft: "",
      bhk: "2",
      bath: "2",
      balcony: "",
    });

    setAuthForm({
      username: "",
      email: "",
      password: "",
    });

    setAuthMode("login");
    setPage("login");

    setError("");
    setMessage("You have been logged out successfully.");
  }

  // ------------------------------------------------
  // HANDLE HOUSE FORM CHANGES
  // ------------------------------------------------

  function handleHouseChange(event) {
    const { name, value } = event.target;

    setHouseForm((previous) => ({
      ...previous,
      [name]: value,
    }));
  }

  // ------------------------------------------------
  // PREDICT HOUSE PRICE
  // ------------------------------------------------

  async function handlePrediction(event) {
    event.preventDefault();

    setError("");
    setMessage("");
    setPrediction(null);

    const totalSqft = Number(houseForm.total_sqft);
    const bhk = Number(houseForm.bhk);
    const bath = Number(houseForm.bath);

    const balcony =
      houseForm.balcony === ""
        ? null
        : Number(houseForm.balcony);

    if (!houseForm.location.trim()) {
      setError("Please enter a house location.");
      return;
    }

    if (!Number.isFinite(totalSqft) || totalSqft <= 0 || totalSqft > 20000) {
      setError("Enter a valid area between 1 and 20,000 square feet.");
      return;
    }

    if (!Number.isInteger(bhk) || bhk < 1 || bhk > 10) {
      setError("BHK must be between 1 and 10.");
      return;
    }

    if (!Number.isInteger(bath) || bath < 1 || bath > 10) {
      setError("Bathrooms must be between 1 and 10.");
      return;
    }

    if (
      balcony !== null &&
      (!Number.isInteger(balcony) || balcony < 0 || balcony > 3)
    ) {
      setError("Balconies must be between 0 and 3.");
      return;
    }

    setLoading(true);

    try {
      const response = await api.post("/predict", {
        location: houseForm.location.trim(),
        total_sqft: totalSqft,
        bhk,
        bath,
        balcony,
      });

      setPrediction(response.data);

      setMessage("Your house price prediction is ready.");

      await loadHistory();
    } catch (requestError) {
      setError(getErrorMessage(requestError));
    } finally {
      setLoading(false);
    }
  }

  // ------------------------------------------------
  // LOAD PREDICTION HISTORY
  // ------------------------------------------------

  async function loadHistory() {
    setHistoryLoading(true);

    try {
      const response = await api.get("/predictions/history");

      setHistory(Array.isArray(response.data) ? response.data : []);
    } catch (requestError) {
      if (requestError.response?.status === 401) {
        localStorage.removeItem("access_token");

        setUser(null);
        setPage("login");

        setError("Your session has expired. Please log in again.");
      } else {
        setError(getErrorMessage(requestError));
      }
    } finally {
      setHistoryLoading(false);
    }
  }

  // ------------------------------------------------
  // DELETE A HISTORY RECORD
  // ------------------------------------------------

  async function deleteHistoryItem(predictionId) {
    const confirmed = window.confirm(
      "Are you sure you want to delete this prediction record?"
    );

    if (!confirmed) {
      return;
    }

    setError("");
    setMessage("");

    try {
      await api.delete(`/predictions/${predictionId}`);

      setHistory((previous) =>
        previous.filter((item) => item.id !== predictionId)
      );

      setMessage("Prediction record deleted successfully.");
    } catch (requestError) {
      setError(getErrorMessage(requestError));
    }
  }

  // ------------------------------------------------
  // LOADING SCREEN
  // ------------------------------------------------

  if (authLoading) {
    return (
      <div className="loading-screen">
        <div className="spinner"></div>

        <p>Preparing your dashboard...</p>
      </div>
    );
  }

  // ------------------------------------------------
  // AUTHENTICATION PAGE
  // ------------------------------------------------

  if (!user) {
    return (
      <div className="auth-layout">
        <section className="auth-visual">
          <div className="visual-content">
            <div className="brand brand-light">
              <span className="brand-icon">H</span>

              <span>HomeValue</span>
            </div>

            <div className="visual-copy">
              <span className="eyebrow">SMART PROPERTY ESTIMATES</span>

              <h1>
                Discover the
                <br />
                value of your
                <br />
                next home.
              </h1>

              <p>
                Explore estimated property prices across Bengaluru with
                machine learning.
              </p>
            </div>

            <div className="visual-stat">
              <span className="stat-symbol">₹</span>

              <div>
                <strong>Smarter decisions.</strong>

                <p>Powered by machine learning.</p>
              </div>
            </div>
          </div>
        </section>

        <section className="auth-panel">
          <div className="auth-card">
            <div className="mobile-brand brand">
              <span className="brand-icon">H</span>

              <span>HomeValue</span>
            </div>

            <div className="auth-heading">
              <span className="eyebrow">
                {authMode === "login"
                  ? "WELCOME BACK"
                  : "JOIN HOMEVALUE"}
              </span>

              <h2>
                {authMode === "login"
                  ? "Sign in to your account"
                  : "Create your account"}
              </h2>

              <p>
                {authMode === "login"
                  ? "Enter your details to access your dashboard."
                  : "Create an account to start estimating property prices."}
              </p>
            </div>

            {error && (
              <div className="alert alert-error" role="alert">
                {error}
              </div>
            )}

            {message && (
              <div className="alert alert-success" role="status">
                {message}
              </div>
            )}

            <form onSubmit={handleAuthSubmit} className="form">
              {authMode === "register" && (
                <div className="form-group">
                  <label htmlFor="username">Full name</label>

                  <input
                    id="username"
                    name="username"
                    type="text"
                    placeholder="Enter your name"
                    value={authForm.username}
                    onChange={handleAuthChange}
                    minLength={3}
                    maxLength={50}
                    autoComplete="name"
                    required
                  />
                </div>
              )}

              <div className="form-group">
                <label htmlFor="email">Email address</label>

                <input
                  id="email"
                  name="email"
                  type="email"
                  placeholder="you@example.com"
                  value={authForm.email}
                  onChange={handleAuthChange}
                  autoComplete="email"
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="password">Password</label>

                <input
                  id="password"
                  name="password"
                  type="password"
                  placeholder="Enter your password"
                  value={authForm.password}
                  onChange={handleAuthChange}
                  minLength={authMode === "register" ? 12 : 1}
                  maxLength={128}
                  autoComplete={
                    authMode === "login"
                      ? "current-password"
                      : "new-password"
                  }
                  required
                />

                {authMode === "register" && (
                  <small>
                    Use at least 12 characters.
                  </small>
                )}
              </div>

              <button
                className="button button-primary button-full"
                type="submit"
                disabled={loading}
              >
                {loading
                  ? "Please wait..."
                  : authMode === "login"
                    ? "Sign in"
                    : "Create account"}

                {!loading && <span aria-hidden="true"> →</span>}
              </button>
            </form>

            <div className="auth-switch">
              {authMode === "login" ? (
                <>
                  Don't have an account?{" "}

                  <button
                    type="button"
                    onClick={() => {
                      setAuthMode("register");
                      setError("");
                      setMessage("");
                    }}
                  >
                    Create one
                  </button>
                </>
              ) : (
                <>
                  Already have an account?{" "}

                  <button
                    type="button"
                    onClick={() => {
                      setAuthMode("login");
                      setError("");
                      setMessage("");
                    }}
                  >
                    Sign in
                  </button>
                </>
              )}
            </div>

            <p className="auth-footer">
              Bengaluru House Price Prediction Project
            </p>
          </div>
        </section>
      </div>
    );
  }

  // ------------------------------------------------
  // AUTHENTICATED DASHBOARD
  // ------------------------------------------------

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-icon">H</span>

          <span>HomeValue</span>
        </div>

        <nav className="topbar-actions">
          <a href="#predict">Predict</a>

          <a href="#history">History</a>

          <div className="user-chip">
            <div className="avatar">
              {(user.username || user.email || "U")
                .charAt(0)
                .toUpperCase()}
            </div>

            <span>{user.username || user.email}</span>
          </div>

          <button
            className="button button-outline button-small"
            onClick={handleLogout}
          >
            Logout
          </button>
        </nav>
      </header>

      <main className="dashboard">
        <section className="welcome-section">
          <div>
            <span className="eyebrow">YOUR PROPERTY WORKSPACE</span>

            <h1>
              Welcome back,{" "}
              {(user.username || "there").split(" ")[0]}.
            </h1>

            <p>
              Estimate property values, explore your results, and review
              your prediction history.
            </p>
          </div>

          <div className="welcome-badge">
            <span className="status-dot"></span>

            ML prediction system
          </div>
        </section>

        {error && (
          <div className="alert alert-error" role="alert">
            {error}

            <button
              className="alert-close"
              onClick={() => setError("")}
              aria-label="Dismiss error"
            >
              ×
            </button>
          </div>
        )}

        {message && (
          <div className="alert alert-success" role="status">
            {message}

            <button
              className="alert-close"
              onClick={() => setMessage("")}
              aria-label="Dismiss message"
            >
              ×
            </button>
          </div>
        )}

        <section className="summary-grid">
          <article className="summary-card">
            <div className="summary-icon icon-purple">₹</div>

            <div>
              <span className="summary-label">
                Latest estimated price
              </span>

              <strong>
                {prediction
                  ? `₹ ${formatPrice(prediction.predicted_price_lakhs)} L`
                  : "—"}
              </strong>

              <small>
                {prediction
                  ? `₹ ${formatPrice(prediction.predicted_price_crores)} Cr`
                  : "Make a prediction to see your result"}
              </small>
            </div>
          </article>

          <article className="summary-card">
            <div className="summary-icon icon-blue">⌂</div>

            <div>
              <span className="summary-label">
                Saved predictions
              </span>

              <strong>{history.length}</strong>

              <small>Records available in your history</small>
            </div>
          </article>

          <article className="summary-card">
            <div className="summary-icon icon-green">✓</div>

            <div>
              <span className="summary-label">
                Account status
              </span>

              <strong>Active</strong>

              <small>Your authenticated session is active</small>
            </div>
          </article>
        </section>

        <section className="workspace-grid">
          <article className="panel prediction-panel" id="predict">
            <div className="panel-heading">
              <div>
                <span className="eyebrow">PROPERTY ANALYSIS</span>

                <h2>Estimate house price</h2>

                <p>
                  Enter the property details to generate an estimate.
                </p>
              </div>

              <div className="panel-icon">⌂</div>
            </div>

            <form onSubmit={handlePrediction} className="form">
              <div className="form-group">
                <label htmlFor="location">Property location</label>

                <input
                  id="location"
                  name="location"
                  type="text"
                  placeholder="e.g. Whitefield, Sarjapur Road"
                  value={houseForm.location}
                  onChange={handleHouseChange}
                  maxLength={150}
                  required
                />

                <small>
                  Enter a location that your trained model recognises.
                </small>
              </div>

              <div className="form-group">
                <label htmlFor="total_sqft">Total area (sq. ft.)</label>

                <input
                  id="total_sqft"
                  name="total_sqft"
                  type="number"
                  placeholder="e.g. 1200"
                  value={houseForm.total_sqft}
                  onChange={handleHouseChange}
                  min="1"
                  max="20000"
                  step="any"
                  required
                />
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label htmlFor="bhk">Bedrooms (BHK)</label>

                  <select
                    id="bhk"
                    name="bhk"
                    value={houseForm.bhk}
                    onChange={handleHouseChange}
                    required
                  >
                    {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((number) => (
                      <option key={number} value={number}>
                        {number} BHK
                      </option>
                    ))}
                  </select>
                </div>

                <div className="form-group">
                  <label htmlFor="bath">Bathrooms</label>

                  <select
                    id="bath"
                    name="bath"
                    value={houseForm.bath}
                    onChange={handleHouseChange}
                    required
                  >
                    {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((number) => (
                      <option key={number} value={number}>
                        {number}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="form-group">
                <label htmlFor="balcony">Balconies</label>

                <select
                  id="balcony"
                  name="balcony"
                  value={houseForm.balcony}
                  onChange={handleHouseChange}
                >
                  <option value="">Not specified</option>

                  <option value="0">0 balconies</option>

                  <option value="1">1 balcony</option>

                  <option value="2">2 balconies</option>

                  <option value="3">3 balconies</option>
                </select>
              </div>

              <div className="form-note">
                <span>ⓘ</span>

                <p>
                  The estimate is generated by your trained ML model.
                  Actual market prices may differ.
                </p>
              </div>

              <button
                className="button button-primary button-full"
                type="submit"
                disabled={loading}
              >
                {loading ? (
                  "Calculating..."
                ) : (
                  <>
                    Predict house price <span aria-hidden="true">→</span>
                  </>
                )}
              </button>
            </form>
          </article>

          <article className="panel result-panel">
            <div className="panel-heading">
              <div>
                <span className="eyebrow">PREDICTION RESULT</span>

                <h2>Your estimate</h2>

                <p>
                  Your model's estimated value for the property.
                </p>
              </div>
            </div>

            {prediction ? (
              <div className="result-content">
                <div className="result-label">
                  Estimated property value
                </div>

                <div className="result-price">
                  <span>₹</span>

                  {formatPrice(prediction.predicted_price_lakhs)}
                </div>

                <div className="result-unit">
                  Lakh
                </div>

                <div className="result-divider"></div>

                <div className="crore-result">
                  <span>Equivalent in crores</span>

                  <strong>
                    ₹ {formatPrice(prediction.predicted_price_crores)} Cr
                  </strong>
                </div>

                <div className="property-summary">
                  <div>
                    <span>Location</span>

                    <strong>{prediction.location}</strong>
                  </div>

                  <div>
                    <span>Area</span>

                    <strong>{houseForm.total_sqft} sq. ft.</strong>
                  </div>

                  <div>
                    <span>Configuration</span>

                    <strong>
                      {houseForm.bhk} BHK · {houseForm.bath} Bath
                    </strong>
                  </div>
                </div>

                <div className="result-disclaimer">
                  This is a machine-learning estimate, not a certified
                  valuation or guaranteed selling price.
                </div>
              </div>
            ) : (
              <div className="empty-result">
                <div className="empty-illustration">
                  <span>₹</span>

                  <div className="floating-square square-one"></div>

                  <div className="floating-square square-two"></div>
                </div>

                <h3>Your result will appear here</h3>

                <p>
                  Fill in the property details and select Predict house
                  price to see the estimated value.
                </p>

                <div className="empty-steps">
                  <span><b>1</b> Enter details</span>

                  <span><b>2</b> Run prediction</span>

                  <span><b>3</b> Review estimate</span>
                </div>
              </div>
            )}
          </article>
        </section>

        <section className="panel history-panel" id="history">
          <div className="history-heading">
            <div>
              <span className="eyebrow">YOUR ACTIVITY</span>

              <h2>Prediction history</h2>

              <p>
                Review the most recent predictions saved to your account.
              </p>
            </div>

            <button
              className="button button-outline"
              onClick={loadHistory}
              disabled={historyLoading}
            >
              {historyLoading ? "Refreshing..." : "↻ Refresh"}
            </button>
          </div>

          {historyLoading && history.length === 0 ? (
            <div className="table-message">
              Loading prediction history...
            </div>
          ) : history.length === 0 ? (
            <div className="table-message">
              <div className="table-empty-icon">▤</div>

              <h3>No predictions yet</h3>

              <p>
                Your saved predictions will appear here after your first
                successful estimate.
              </p>
            </div>
          ) : (
            <div className="table-wrapper">
              <table>
                <thead>
                  <tr>
                    <th>Location</th>

                    <th>Area</th>

                    <th>Bedrooms</th>

                    <th>Bathrooms</th>

                    <th>Balconies</th>

                    <th>Price (Lakhs)</th>

                    <th>Price (Crores)</th>

                    <th>Date</th>

                    <th>Action</th>
                  </tr>
                </thead>

                <tbody>
                  {history.map((item) => (
                    <tr key={item.id}>
                      <td className="location-cell">
                        {item.location}
                      </td>

                      <td>{Number(item.total_sqft).toLocaleString("en-IN")}</td>

                      <td>{item.bhk}</td>

                      <td>{item.bath}</td>

                      <td>
                        {item.balcony ?? "—"}
                      </td>

                      <td className="price-cell">
                        ₹ {formatPrice(item.predicted_price_lakhs)}
                      </td>

                      <td>
                        ₹ {formatPrice(item.predicted_price_crores)}
                      </td>

                      <td>{formatDate(item.created_at)}</td>

                      <td>
                        <button
                          className="delete-button"
                          onClick={() => deleteHistoryItem(item.id)}
                          aria-label={`Delete prediction ${item.id}`}
                          title="Delete this prediction"
                        >
                          Delete
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {history.length > 0 && (
            <div className="table-footer">
              Showing {history.length} most recent saved prediction
              {history.length === 1 ? "" : "s"}.
            </div>
          )}
        </section>

        <footer className="dashboard-footer">
          <div className="brand">
            <span className="brand-icon">H</span>

            <span>HomeValue</span>
          </div>

          <p>
            Bengaluru House Price Predictor · Machine Learning Project
          </p>

          <span>For estimation purposes only</span>
        </footer>
      </main>
    </div>
  );
}