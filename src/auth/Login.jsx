
import React, { useState } from "react";
import { Eye, EyeOff, ShieldCheck, Droplets } from "lucide-react";
import { useAuth } from "./AuthContext.jsx";

export default function Login() {
  const { login, signup } = useAuth();

  const [mode, setMode] = useState("login");
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [submitting, setSubmitting] = useState(false);

  function changeMode(nextMode) {
    setMode(nextMode);
    setError("");
    setMessage("");
    setPassword("");
    setConfirmPassword("");
    setShowPassword(false);
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");
    setMessage("");

    const cleanEmail = email.trim().toLowerCase();

    if (mode === "signup") {
      if (fullName.trim().length < 2) {
        setError("Please enter your full name.");
        return;
      }

      if (password.length < 12) {
        setError("Password must contain at least 12 characters.");
        return;
      }

      if (password !== confirmPassword) {
        setError("Passwords do not match.");
        return;
      }
    }

    setSubmitting(true);

    try {
      if (mode === "login") {
        await login(cleanEmail, password);
      } else {
        await signup(fullName.trim(), cleanEmail, password);

        setMessage(
          "Registration submitted successfully. Your account is awaiting administrator approval. You can sign in once it has been approved."
        );

        setMode("login");
        setPassword("");
        setConfirmPassword("");
        setFullName("");
      }
    } catch (err) {
      setError(err.message || "Something went wrong. Please try again.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="boreman-auth-page">
      <div className="boreman-auth-card">
        <div className="boreman-auth-brand">
          <div className="boreman-auth-logo">
            <Droplets size={30} />
          </div>
          <h1>BOREMAN</h1>
          <p>Water Quality Intelligence Platform</p>
        </div>

        <div className="boreman-auth-tabs">
          <button
            type="button"
            className={mode === "login" ? "active" : ""}
            onClick={() => changeMode("login")}
          >
            Login
          </button>
          <button
            type="button"
            className={mode === "signup" ? "active" : ""}
            onClick={() => changeMode("signup")}
          >
            Sign Up
          </button>
        </div>

        <h2>{mode === "login" ? "Welcome back" : "Create your account"}</h2>

        <p className="boreman-auth-subtitle">
          {mode === "login"
            ? "Sign in to access your water quality dashboard."
            : "Register for access to the BOREMAN platform."}
        </p>

        {error && (
          <div className="boreman-auth-error" role="alert">
            {error}
          </div>
        )}

        {message && (
          <div className="boreman-auth-success" role="status">
            {message}
          </div>
        )}

        <form onSubmit={handleSubmit}>
          {mode === "signup" && (
            <label>
              Full name
              <input
                type="text"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                placeholder="Enter your full name"
                minLength={2}
                maxLength={120}
                autoComplete="name"
                required
              />
            </label>
          )}

          <label>
            Email address
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              autoComplete="username"
              autoCapitalize="none"
              required
            />
          </label>

          <label>
            Password
            <div className="boreman-password-field">
              <input
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter your password"
                autoComplete={
                  mode === "login" ? "current-password" : "new-password"
                }
                minLength={mode === "signup" ? 12 : undefined}
                required
              />
              <button
                type="button"
                className="boreman-password-toggle"
                onClick={() => setShowPassword(!showPassword)}
                aria-label={showPassword ? "Hide password" : "Show password"}
              >
                {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
              </button>
            </div>
            {mode === "signup" && (
              <small className="boreman-auth-hint">
                Use at least 12 characters.
              </small>
            )}
          </label>

          {mode === "signup" && (
            <label>
              Confirm password
              <input
                type={showPassword ? "text" : "password"}
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                placeholder="Enter your password again"
                autoComplete="new-password"
                minLength={12}
                required
              />
            </label>
          )}

          <button
            className="boreman-auth-submit"
            type="submit"
            disabled={submitting}
          >
            {submitting
              ? "Please wait..."
              : mode === "login"
                ? "Sign In"
                : "Create Account"}
          </button>
        </form>

        <div className="boreman-auth-footer">
          <ShieldCheck size={15} />
          Secure access · Administrator-approved accounts
        </div>
      </div>
    </div>
  );
}
