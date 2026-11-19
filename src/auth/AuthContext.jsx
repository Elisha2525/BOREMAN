import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
} from "react";

import {
  authApi,
  clearToken,
  getToken,
  setToken,
} from "./api.js";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  const refreshUser = useCallback(async () => {
    if (!getToken()) {
      setUser(null);
      return null;
    }

    try {
      const currentUser = await authApi.me();
      setUser(currentUser);
      return currentUser;
    } catch (error) {
      clearToken();
      setUser(null);
      return null;
    }
  }, []);

  useEffect(() => {
    refreshUser().finally(() => setLoading(false));
  }, [refreshUser]);

  const login = async (email, password) => {
    const result = await authApi.login({ email, password });
    setToken(result.access_token);

    try {
      const currentUser = await authApi.me();
      setUser(currentUser);
      return currentUser;
    } catch (error) {
      clearToken();
      setUser(null);
      throw error;
    }
  };

  const signup = async (full_name, email, password) => {
    return authApi.signup({ full_name, email, password });
  };

  const logout = async () => {
    try {
      if (getToken()) {
        await authApi.logout();
      }
    } finally {
      clearToken();
      setUser(null);
    }
  };

  const value = {
    user,
    loading,
    login,
    signup,
    logout,
    refreshUser,
    isAdmin: user?.role === "administrator",
    isResearcher: user?.role === "researcher",
    isViewer: user?.role === "viewer",
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);

  if (!context) {
    throw new Error("useAuth must be used inside AuthProvider");
  }

  return context;
}
