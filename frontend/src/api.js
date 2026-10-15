import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000",
  timeout: 15000,
});

// Attach the JWT access token to protected API requests.
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem("access_token");

    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }

    return config;
  },
  (error) => Promise.reject(error)
);

// Return a readable error message from FastAPI responses.
export function getErrorMessage(error) {
  if (error.response) {
    const detail = error.response.data?.detail;

    if (typeof detail === "string") {
      return detail;
    }

    if (Array.isArray(detail)) {
      return detail
        .map((item) => item.msg || "Invalid input")
        .join(", ");
    }

    return "The server could not process your request.";
  }

  if (error.code === "ECONNABORTED") {
    return "The request timed out. Please try again.";
  }

  if (error.request) {
    return "Cannot connect to the backend. Make sure FastAPI is running.";
  }

  return error.message || "An unexpected error occurred.";
}

export default api;