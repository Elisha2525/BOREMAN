const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:9001";

export function getToken() {
  return sessionStorage.getItem("boreman_access_token");
}

export function setToken(token) {
  sessionStorage.setItem("boreman_access_token", token);
}

export function clearToken() {
  sessionStorage.removeItem("boreman_access_token");
}

export async function apiRequest(path, options = {}) {
  const token = getToken();

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...options.headers,
    },
  });

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(data.detail || `Request failed (${response.status})`);
  }

  return data;
}

export const authApi = {
  signup: (payload) =>
    apiRequest("/api/auth/signup", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  login: (payload) =>
    apiRequest("/api/auth/login", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  me: () => apiRequest("/api/auth/me"),

  logout: () =>
    apiRequest("/api/auth/logout", { method: "POST" }),

  users: () => apiRequest("/api/auth/users"),

  updateUser: (id, payload) =>
    apiRequest(`/api/auth/users/${id}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),
};
