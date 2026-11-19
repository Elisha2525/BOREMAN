import React, { useEffect, useState } from "react";
import "./auth.css";

const API = import.meta.env.VITE_AUTH_API_URL || "http://127.0.0.1:9001";
const KEY = "boreman_access_token";

async function api(path, options = {}, token = "") {
  const res = await fetch(`${API}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...(token ? { Authorization: `Bearer ${token}` } : {}), ...(options.headers || {}) }
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
  return data;
}

export default function AuthPanel({ children, onAuthenticated }) {
  const [mode, setMode] = useState("login");
  const [form, setForm] = useState({ full_name: "", email: "", password: "" });
  const [token, setToken] = useState(() => localStorage.getItem(KEY) || "");
  const [user, setUser] = useState(null);
  const [users, setUsers] = useState([]);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const set = (k, v) => setForm(old => ({ ...old, [k]: v }));

  async function loadMe(t) {
    const me = await api("/api/auth/me", {}, t);
    setUser(me); onAuthenticated?.({ user: me, token: t });
  }

  useEffect(() => {
    if (token) loadMe(token).catch(() => { localStorage.removeItem(KEY); setToken(""); setUser(null); });
  }, []);

  async function submit(e) {
    e.preventDefault(); setBusy(true); setError(""); setMessage("");
    try {
      if (mode === "signup") {
        const result = await api("/api/auth/signup", { method: "POST", body: JSON.stringify(form) });
        setMessage(result.message); setMode("login");
      } else {
        const result = await api("/api/auth/login", { method: "POST", body: JSON.stringify({ email: form.email, password: form.password }) });
        localStorage.setItem(KEY, result.access_token); setToken(result.access_token);
        setUser(result.user); onAuthenticated?.({ user: result.user, token: result.access_token });
      }
    } catch (err) { setError(err.message); }
    finally { setBusy(false); }
  }

  async function logout() {
    try { if (token) await api("/api/auth/logout", { method: "POST" }, token); } catch {}
    localStorage.removeItem(KEY); setToken(""); setUser(null); setUsers([]);
  }

  async function loadUsers() {
    setError("");
    try { setUsers(await api("/api/auth/users", {}, token)); }
    catch (err) { setError(err.message); }
  }

  async function updateUser(id, patch) {
    setError(""); setMessage("");
    try {
      await api(`/api/auth/users/${id}`, { method: "PATCH", body: JSON.stringify(patch) }, token);
      setMessage("User updated."); await loadUsers();
    } catch (err) { setError(err.message); }
  }

  if (user) return (
    <div className="boreman-auth-shell">
      <header className="boreman-auth-top"><div><b>BOREMAN</b><small>Secure workspace</small></div><div>{user.full_name} · {user.role} <button onClick={logout}>Log out</button></div></header>
      {children ? <main>{children}</main> : <main className="boreman-auth-card wide">
        <h2>Welcome, {user.full_name}</h2><p>Authenticated BOREMAN account · {user.role}</p>
        {user.role === "administrator" && <section><div className="boreman-auth-row"><h3>User management</h3><button onClick={loadUsers}>Load users</button></div>
          {users.map(u => <div className="boreman-auth-user" key={u.id}>
            <span><b>{u.full_name}</b><small>{u.email} · {u.status} · {u.is_active ? "Active" : "Deactivated"}</small></span>
            <select value={u.role} onChange={e => updateUser(u.id, { role: e.target.value })}><option value="administrator">Administrator</option><option value="researcher">Researcher</option><option value="viewer">Viewer</option></select>
            {u.status === "pending" && <button onClick={() => updateUser(u.id, { status: "approved" })}>Approve</button>}
            <button className="muted" onClick={() => updateUser(u.id, { is_active: !u.is_active })}>{u.is_active ? "Deactivate" : "Activate"}</button>
          </div>)}
        </section>}
        {message && <p className="success">{message}</p>}{error && <p className="error">{error}</p>}
      </main>}
    </div>
  );

  return <div className="boreman-auth-page"><form className="boreman-auth-card" onSubmit={submit}>
    <div className="boreman-auth-brand"><span>B</span><div><h1>BOREMAN</h1><small>Water quality monitoring</small></div></div>
    <h2>{mode === "login" ? "Sign in" : "Request an account"}</h2>
    <p className="hint">{mode === "login" ? "Use your approved BOREMAN account." : "New accounts require administrator approval."}</p>
    {mode === "signup" && <label>Full name<input required minLength="2" value={form.full_name} onChange={e => set("full_name", e.target.value)} /></label>}
    <label>Email<input required type="email" value={form.email} onChange={e => set("email", e.target.value)} /></label>
    <label>Password<input required type="password" minLength={mode === "signup" ? 12 : 1} value={form.password} onChange={e => set("password", e.target.value)} /></label>
    {error && <p className="error">{error}</p>}{message && <p className="success">{message}</p>}
    <button className="primary" disabled={busy}>{busy ? "Please wait…" : mode === "login" ? "Sign in" : "Submit signup request"}</button>
    <button type="button" className="link" onClick={() => { setMode(mode === "login" ? "signup" : "login"); setError(""); setMessage(""); }}>
      {mode === "login" ? "New to BOREMAN? Request access" : "Already registered? Sign in"}
    </button>
  </form></div>;
}
