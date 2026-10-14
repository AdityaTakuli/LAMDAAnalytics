# LAMDA dashboard

React (Vite) front end for the LAMDA risk-scoring service. Enter a component type, seller and import
locations, and a seller name; the dashboard sends them to the backend's `POST /analyze` and shows the
risk score, the contributing factors and the suggested mitigations on a map.

```bash
npm install
npm run dev        # then open http://localhost:5173/dashboard
```

The backend must be running at `http://127.0.0.1:8000` (set in `src/services/api.js`). See the
repository README for how to start it.
