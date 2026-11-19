import React, { useCallback, useEffect, useState } from "react";
import { authApi } from "./api.js";

export default function AdminUsers() {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [savingId, setSavingId] = useState(null);

  const loadUsers = useCallback(async () => {
    setLoading(true);
    setError("");

    try {
      const data = await authApi.users();
      setUsers(Array.isArray(data) ? data : data.users || []);
    } catch (err) {
      setError(err.message || "Unable to load users.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadUsers();
  }, [loadUsers]);

  async function updateUser(id, payload) {
    setSavingId(id);
    setError("");
    setMessage("");

    try {
      await authApi.updateUser(id, payload);
      setMessage("User updated successfully.");
      await loadUsers();
    } catch (err) {
      setError(err.message || "Unable to update user.");
    } finally {
      setSavingId(null);
    }
  }

  const pendingUsers = users.filter((u) => u.status === "pending");
  const otherUsers = users.filter((u) => u.status !== "pending");

  function UserRow({ user }) {
    return (
      <tr>
        <td>
          <strong>{user.full_name}</strong>
          <div className="admin-user-email">{user.email}</div>
        </td>
        <td>
          <select
            value={user.role}
            disabled={savingId === user.id}
            onChange={(e) =>
              updateUser(user.id, { role: e.target.value })
            }
          >
            <option value="administrator">Administrator</option>
            <option value="researcher">Researcher</option>
            <option value="viewer">Viewer</option>
          </select>
        </td>
        <td>
          <span className={`admin-status ${user.status}`}>
            {user.status}
          </span>
        </td>
        <td>{user.is_active ? "Active" : "Inactive"}</td>
        <td>
          {user.status === "pending" ? (
            <div className="admin-actions">
              <button
                disabled={savingId === user.id}
                onClick={() =>
                  updateUser(user.id, {
                    status: "approved",
                    is_active: true,
                  })
                }
              >
                Approve
              </button>
              <button
                className="admin-reject"
                disabled={savingId === user.id}
                onClick={() =>
                  updateUser(user.id, { status: "rejected" })
                }
              >
                Reject
              </button>
            </div>
          ) : (
            <button
              disabled={savingId === user.id}
              onClick={() =>
                updateUser(user.id, {
                  is_active: !user.is_active,
                })
              }
            >
              {user.is_active ? "Deactivate" : "Activate"}
            </button>
          )}
        </td>
      </tr>
    );
  }

  return (
    <div className="admin-users-page">
      <section className="section-intro">
        <div>
          <h2>User administration</h2>
          <p>Review registration requests and manage platform access.</p>
        </div>
        <button onClick={loadUsers}>Refresh users</button>
      </section>

      {error && <div className="boreman-auth-error">{error}</div>}
      {message && <div className="boreman-auth-success">{message}</div>}

      <div className="admin-summary">
        <article className="panel">
          <span>All users</span>
          <strong>{users.length}</strong>
        </article>
        <article className="panel">
          <span>Pending approvals</span>
          <strong>{pendingUsers.length}</strong>
        </article>
        <article className="panel">
          <span>Approved users</span>
          <strong>{users.filter((u) => u.status === "approved").length}</strong>
        </article>
      </div>

      {loading ? (
        <div className="panel">Loading users...</div>
      ) : (
        <>
          <article className="panel">
            <h3>Pending registration requests</h3>
            {pendingUsers.length === 0 ? (
              <p>No pending registration requests.</p>
            ) : (
              <div className="admin-table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>User</th>
                      <th>Role</th>
                      <th>Status</th>
                      <th>Account</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {pendingUsers.map((user) => (
                      <UserRow key={user.id} user={user} />
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </article>

          <article className="panel">
            <h3>All registered users</h3>
            <div className="admin-table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>User</th>
                    <th>Role</th>
                    <th>Status</th>
                    <th>Account</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {otherUsers.map((user) => (
                    <UserRow key={user.id} user={user} />
                  ))}
                </tbody>
              </table>
            </div>
          </article>
        </>
      )}
    </div>
  );
}
