# Connect to the current Vite frontend

1. Copy `src/auth/AuthPanel.jsx` and `src/auth/auth.css` into your existing frontend `src/auth/`.
2. Add `.env.local` at the Vite project root. Set `VITE_AUTH_API_URL=http://127.0.0.1:9000` after the auth router is mounted in the current API.
3. In `App.jsx`, import `AuthPanel` and wrap the existing dashboard component:

```jsx
import AuthPanel from "./auth/AuthPanel";
import ExistingBoremanDashboard from "./ExistingBoremanDashboard";

export default function App() {
  return <AuthPanel><ExistingBoremanDashboard /></AuthPanel>;
}
```

Replace the example import with the component currently rendered by your `App.jsx`.
4. Protected API calls must send the token:

```js
const token = localStorage.getItem("boreman_access_token");
fetch(`${API_BASE_URL}/api/live`, {
  headers: { Authorization: `Bearer ${token}` }
});
```

5. The FastAPI route itself must also require authentication. Hiding a page in React alone does not secure the API.

For a standalone UI test, temporarily render `<AuthPanel />` without dashboard children. The UI includes an administrator user list, approval, role change, and activate/deactivate controls.
