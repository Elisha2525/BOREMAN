
import React, { useCallback, useEffect, useMemo, useState } from "react";

import modelEvaluation from "./data/modelEvaluation.json";
import step40Comparison from "./data/step40Comparison.json";
import actualVsPredicted from "./data/actualVsPredicted.json";

import {
  Activity,
  AlertTriangle,
  Bell,
  CheckCircle2,
  ChevronDown,
  CircleHelp,
  Droplets,
  Gauge,
  LayoutDashboard,
  Menu,
  Radio,
  Settings,
  ShieldCheck,
  Thermometer,
  Waves,
  Zap,
  Clock3,
  BrainCircuit,
  Download,
  Wifi,
  RefreshCw,
  MapPin,
  SlidersHorizontal,
  ArrowUpRight,
  FlaskConical,
  FileSpreadsheet,
  Database,
  CloudDownload,
  TrendingUp,
  History,
  Save,
  ExternalLink,
  X,
  Search,
  CalendarDays,
  CircleGauge,
  MessageCircle,
  Send,
  Bot,
  Minimize2
} from "lucide-react";

import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

// API configuration
const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "https://boreman-ai-api-10135309602.development.catalystappsail.com/";

const TIME_ZONE = "Africa/Windhoek";

// Forecast horizons
const FORECAST_HORIZONS = [5, 15, 30, 60];

// Normalise target names consistently
const normaliseTarget = (target) =>
  String(target || "").toLowerCase();

// Step 40 model comparison chart data
const step40ChartData = FORECAST_HORIZONS.map((minutes) => {
  const row = { horizon: `${minutes} min` };

  ["tds", "humidity", "temperature"].forEach((target) => {
    const metric = step40Comparison.find(
      (item) =>
        normaliseTarget(item.target) === target &&
        Number(item.horizon_minutes) === minutes
    );

    if (metric) {
      row[`${target}_selected_mae`] = metric.selected_model_mae;
      row[`${target}_persistence_mae`] = metric.persistence_mae;
      row[`${target}_selected_rmse`] = metric.selected_model_rmse;
      row[`${target}_persistence_rmse`] = metric.persistence_rmse;
    }
  });

  return row;
});

// Model evaluation chart data
const evaluationChartData = FORECAST_HORIZONS.map((minutes) => {
  const row = { horizon: `${minutes} min` };

  ["tds", "humidity", "temperature"].forEach((target) => {
    const metric = modelEvaluation.find(
      (item) =>
        normaliseTarget(item.target) === target &&
        Number(item.horizon_minutes) === minutes
    );

    if (metric) {
      row[`${target}_mae`] = metric.mae;
      row[`${target}_rmse`] = metric.rmse;
      row[`${target}_r2`] = metric.r2;
    }
  });

  return row;
});

// Actual versus predicted evaluation data
const actualVsPredictedGroups = actualVsPredicted.reduce(
  (groups, item) => {
    const target = normaliseTarget(item.target);
    const key = `${target}_${item.horizon_minutes}`;

    if (!groups[key]) {
      groups[key] = [];
    }

    const timestamp = new Date(item.timestamp);

    groups[key].push({
      time: Number.isNaN(timestamp.getTime())
        ? "Invalid date"
        : timestamp.toLocaleString("en-GB", {
            timeZone: TIME_ZONE,
            day: "2-digit",
            month: "short",
            hour: "2-digit",
            minute: "2-digit"
          }),
      actual: item.actual,
      prediction: item.prediction,
      persistence: item.persistence
    });

    return groups;
  },
  {}
);

// Clearly labelled sample data.
// This is not live sensor data.
const DEMO_READINGS = [
  { time: "00:00", tds: 332, temp: 24.1, humidity: null },
  { time: "02:00", tds: 326, temp: 24.0, humidity: null },
  { time: "04:00", tds: 329, temp: 23.8, humidity: null },
  { time: "06:00", tds: 336, temp: 24.2, humidity: null },
  { time: "08:00", tds: 341, temp: 24.5, humidity: null },
  { time: "10:00", tds: 338, temp: 24.7, humidity: null },
  { time: "12:00", tds: 344, temp: 24.8, humidity: null },
  { time: "14:00", tds: 347, temp: 24.6, humidity: null },
  { time: "16:00", tds: 343, temp: 24.5, humidity: null },
  { time: "18:00", tds: 339, temp: 24.3, humidity: null },
  { time: "20:00", tds: 341, temp: 24.2, humidity: null },
  { time: "22:00", tds: 344, temp: 24.4, humidity: null }
];

// Sidebar navigation
const NAV = [
  {
    label: "Dashboard",
    icon: LayoutDashboard,
    desc: "Overview of the latest borewell readings."
  },
  {
    label: "Live readings",
    icon: Activity,
    desc: "Inspect the latest sensor values and recent history."
  },
  {
    label: "Predictions",
    icon: BrainCircuit,
    desc: "View trend-based estimates and derived indicators."
  },
  {
    label: "Reports",
    icon: FlaskConical,
    desc: "Review and export historical sensor records."
  },
  {
    label: "Settings",
    icon: Settings,
    desc: "Monitor the live API connection and configure dashboard thresholds."
  }
];

// Format timestamps using Namibia time
function formatTime(value) {
  if (!value) return "N/A";

  const date = new Date(value);

  if (!Number.isNaN(date.getTime())) {
    return date.toLocaleString("en-GB", {
      timeZone: TIME_ZONE,
      day: "2-digit",
      month: "short",
      hour: "2-digit",
      minute: "2-digit"
    });
  }

  return String(value);
}

// Calculate mean while ignoring missing values
function mean(rows, key) {
  const values = rows
    .map((row) => Number(row[key]))
    .filter((value) => Number.isFinite(value));

  if (!values.length) return null;

  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

// Reusable metric card
function MetricCard({
  icon: Icon,
  label,
  value,
  unit,
  status,
  color,
  detail,
  tag
}) {
  return (
    <article className={`metric-card ${color}`}>
      <div className="metric-icon">
        <Icon size={22} />
      </div>

      <div className="metric-content">
        <div className="metric-label">{label}</div>

        <div className="metric-value">
          {value ?? "N/A"}
          <span>{unit}</span>
        </div>

        <div className="metric-foot">
          <span className="status-dot" />
          {status}
          <span className="metric-detail">{detail}</span>
        </div>
      </div>

      <div className="metric-change">{tag}</div>
    </article>
  );
}

// Reusable panel heading
function PanelHeader({ icon: Icon, title, subtitle, action }) {
  return (
    <div className="panel-header">
      <div>
        <h3>
          <Icon size={18} />
          {title}
        </h3>
        <p>{subtitle}</p>
      </div>

      {action}
    </div>
  );
}

function App() {
  // --------------------------------------------------
  // 1. APPLICATION STATE
  // --------------------------------------------------

  const [active, setActive] = useState("Dashboard");

  const [readings, setReadings] = useState([]);
  const [liveData, setLiveData] = useState(null);
  const [forecasts, setForecasts] = useState({});
  const [historyReadings, setHistoryReadings] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState("");

  const [source, setSource] = useState("live");
  const [refreshSeconds, setRefreshSeconds] = useState("30");

  const [loading, setLoading] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [error, setError] = useState("");
  const [healthStatus, setHealthStatus] = useState(null);
  const [modelStatus, setModelStatus] = useState(null);
  const [healthLoading, setHealthLoading] = useState(false);
  const [healthError, setHealthError] = useState("");
  const [toast, setToast] = useState("");

  const [mobileNav, setMobileNav] = useState(false);
  const [search, setSearch] = useState("");

  // Display thresholds only. These are not safety standards.
  const [thresholdTds, setThresholdTds] = useState("500");
  const [thresholdTemp, setThresholdTemp] = useState("35");

  // Research evaluation controls
  const [selectedEvalTarget, setSelectedEvalTarget] =
    useState("tds");

  const [selectedEvalHorizon, setSelectedEvalHorizon] =
    useState("5");

  // Chatbot state
  const [chatOpen, setChatOpen] = useState(false);
  const [chatInput, setChatInput] = useState("");

  const [chatMessages, setChatMessages] = useState([
    {
      role: "assistant",
      content:
        "Hello! I am the BOREMAN Assistant. I can help you understand live TDS, water temperature, ambient humidity, historical readings and experimental forecasts. These readings are for research and do not certify water safety."
    }
  ]);

  // --------------------------------------------------
  // 2. DERIVED DATA
  // --------------------------------------------------

  const latest = readings[readings.length - 1] || {};

  const prev =
    readings.length > 1
      ? readings[readings.length - 2]
      : latest;

  const tdsValue = latest.tds ?? null;
  const tempValue = latest.temp ?? null;
  const humidityValue = latest.humidity ?? null;

  const tdsLimit = Number(thresholdTds) || 500;
  const tempLimit = Number(thresholdTemp) || 35;

  const tdsStatus =
    tdsValue == null
      ? "No data"
      : tdsValue <= tdsLimit
        ? "Within display limit"
        : "Above display limit";

  const tempStatus =
    tempValue == null
      ? "No data"
      : tempValue <= tempLimit
        ? "Within display limit"
        : "Above display limit";

  const humidityStatus =
    humidityValue == null
      ? "No data"
      : "Environmental humidity reading";

  const averageTds = mean(readings, "tds");
  const averageTemp = mean(readings, "temp");
  const averageHumidity = mean(readings, "humidity");

  const overall = [
    tdsValue != null && tdsValue > tdsLimit,
    tempValue != null && tempValue > tempLimit
  ].some(Boolean)
    ? "Review readings"
    : "Within display limits";

  const sourceLabel =
    source === "live"
      ? "Adafruit IO Live"
      : "Simulated data";

  const actualVsPredictedData =
    actualVsPredictedGroups[
      `${selectedEvalTarget}_${selectedEvalHorizon}`
    ] || [];

  // --------------------------------------------------
  // 3. NOTIFICATION
  // --------------------------------------------------

  const notify = useCallback((message) => {
    setToast(message);

    window.setTimeout(() => {
      setToast("");
    }, 3000);
  }, []);

  // --------------------------------------------------
  // 4. LIVE API CONNECTION
  // --------------------------------------------------

  const fetchLive = useCallback(async () => {
    setLoading(true);
    setError("");

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/live`
      );

      if (!response.ok) {
        const detail = await response
          .json()
          .catch(() => ({}));

        throw new Error(
          detail.detail ||
          `API returned HTTP ${response.status}`
        );
      }

      const data = await response.json();

      if (!Array.isArray(data.readings)) {
        throw new Error(
          "The API response does not contain a valid readings array."
        );
      }

      const parsed = data.readings
        .filter((item) => item.timestamp)
        .map((item) => {
          const timestamp = new Date(item.timestamp);

          return {
            timestamp: item.timestamp,

            time: Number.isNaN(timestamp.getTime())
              ? "Invalid time"
              : timestamp.toLocaleTimeString("en-GB", {
                  timeZone: TIME_ZONE,
                  hour: "2-digit",
                  minute: "2-digit"
                }),

            tds:
              item.tds == null
                ? null
                : Number(item.tds),

            temp:
              item.temperature == null
                ? null
                : Number(item.temperature),

            humidity:
              item.humidity == null
                ? null
                : Number(item.humidity)
          };
        });

      setReadings(parsed);
      setLiveData(data);
      setForecasts(data.forecasts || {});
      setSource("live");

      const updateTime = data.updated_at_utc
        ? new Date(data.updated_at_utc)
        : new Date();

      setLastUpdated(
        Number.isNaN(updateTime.getTime())
          ? new Date()
          : updateTime
      );

      setError("");
    } catch (err) {
      setError(
        `Live API connection failed: ${err.message}`
      );
    } finally {
      setLoading(false);
    }
  }, []);

  // --------------------------------------------------
  // 5. AUTOMATIC REFRESH
  // --------------------------------------------------

  useEffect(() => {
    fetchLive();

    const seconds = Math.max(
      15,
      Number(refreshSeconds) || 30
    );

    const timer = window.setInterval(() => {
      fetchLive();
    }, seconds * 1000);

    return () => {
      window.clearInterval(timer);
    };
  }, [fetchLive, refreshSeconds]);

  

  const fetchHistory = useCallback(async () => {
  setHistoryLoading(true);
  setHistoryError("");

  try {
    const response = await fetch(`${API_BASE_URL}/api/history`);

    if (!response.ok) {
      throw new Error(`API returned HTTP ${response.status}`);
    }

    const data = await response.json();

    if (!Array.isArray(data)) {
      throw new Error("History API did not return a valid array.");
    }

    const parsed = data
      .filter((item) => item.timestamp)
      .map((item) => ({
        timestamp: item.timestamp,
        time: new Date(item.timestamp).toLocaleString("en-GB", {
          timeZone: TIME_ZONE,
          day: "2-digit",
          month: "2-digit",
          year: "numeric",
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit"
        }),
        tds: item.tds == null ? null : Number(item.tds),
        temp: item.temperature == null ? null : Number(item.temperature),
        humidity: item.humidity == null ? null : Number(item.humidity)
      }))
      .sort(
        (a, b) =>
          new Date(a.timestamp).getTime() -
          new Date(b.timestamp).getTime()
      );

    setHistoryReadings(parsed);
  } catch (err) {
    setHistoryError(err.message);
  } finally {
    setHistoryLoading(false);
  }
}, []);


  useEffect(() => {
  fetchHistory();
  }, [fetchHistory]);


  const checkBackendHealth = useCallback(async () => {
  setHealthLoading(true);
  setHealthError("");

  try {
    const [healthResponse, modelsResponse] = await Promise.all([
      fetch(`${API_BASE_URL}/health`),
      fetch(`${API_BASE_URL}/models`)
    ]);

    if (!healthResponse.ok) {
      throw new Error(`Health API returned HTTP ${healthResponse.status}`);
    }

    if (!modelsResponse.ok) {
      throw new Error(`Models API returned HTTP ${modelsResponse.status}`);
    }

    const health = await healthResponse.json();
    const models = await modelsResponse.json();

    setHealthStatus(health);
    setModelStatus(models);
  } catch (err) {
    setHealthError(err.message);
    setHealthStatus(null);
    setModelStatus(null);
  } finally {
    setHealthLoading(false);
  }
}, []);

  useEffect(() => {
    checkBackendHealth();
  }, [checkBackendHealth]);
  // --------------------------------------------------
  // 6. CSV EXPORT
  // --------------------------------------------------

  const exportCsv = () => {
    const escapeCsv = (value) => {
      if (value == null) return "";

      return `"${String(value).replace(/"/g, '""')}"`;
    };

    const rows = [
      [
        "Timestamp",
        "TDS (ppm)",
        "Ambient humidity (%)",
        "Water temperature (C)"
      ],

      ...historyReadings.map((row) => [
        row.timestamp || row.time,
        row.tds,
        row.humidity,
        row.temp
      ])
    ];

    const csv = rows
      .map((row) => row.map(escapeCsv).join(","))
      .join("\n");

    const blob = new Blob(
      ["\uFEFF", csv],
      { type: "text/csv;charset=utf-8;" }
    );

    const url = URL.createObjectURL(blob);

    const link = document.createElement("a");
    link.href = url;
    link.download = "boreman-water-quality-readings.csv";

    document.body.appendChild(link);
    link.click();
    link.remove();

    URL.revokeObjectURL(url);

    notify("CSV report downloaded");
  };

  // --------------------------------------------------
  // 7. DEMO MODE
  // --------------------------------------------------

  const switchToDemo = () => {
    setSource("demo");
    setReadings(DEMO_READINGS);
    setLiveData(null);
    setForecasts({});
    setError("");
    setLastUpdated(new Date());

    notify("Sample data mode activated");
  };

  // --------------------------------------------------
  // 8. HISTORICAL CHART DATA
  // --------------------------------------------------

  const trendData = useMemo(
    () => readings.slice(-24),
    [readings]
  );

  // --------------------------------------------------
  // 9. FORECAST CHART PREPARATION
  // --------------------------------------------------

  const buildForecastChart = useCallback(
    (field, forecastPrefix) => {
      const history = readings.slice(-24);

      if (!history.length) return [];

      const historical = history.map((row) => {
        const value = Number(row[field]);

        return {
          time: row.time || formatTime(row.timestamp),
          actual: Number.isFinite(value) ? value : null,
          forecast: null
        };
      });

      const chart = [...historical];
      const latestReading = history[history.length - 1];

      const currentValue = Number(latestReading[field]);

      if (Number.isFinite(currentValue)) {
        chart[chart.length - 1] = {
          ...chart[chart.length - 1],
          actual: currentValue,
          forecast: currentValue
        };
      }

      const latestTime = latestReading.timestamp
        ? new Date(latestReading.timestamp)
        : null;

      FORECAST_HORIZONS.forEach((minutes) => {
        const prediction =
          forecasts[`${forecastPrefix}_${minutes}m`];

        if (
          !prediction ||
          prediction.value == null ||
          !Number.isFinite(Number(prediction.value))
        ) {
          return;
        }

        // Do not invent future timestamps for demo data.
        if (
          !latestTime ||
          Number.isNaN(latestTime.getTime())
        ) {
          return;
        }

        const futureTime = new Date(
          latestTime.getTime() +
          minutes * 60 * 1000
        );

        chart.push({
          time: futureTime.toLocaleTimeString(
            "en-GB",
            {
              timeZone: TIME_ZONE,
              hour: "2-digit",
              minute: "2-digit"
            }
          ),

          actual: null,
          forecast: Number(prediction.value),
          horizon: `${minutes} min`,
          method: prediction.method || "Not specified"
        });
      });

      return chart;
    },
    [readings, forecasts]
  );

  const tdsForecastData = useMemo(
    () => buildForecastChart("tds", "tds"),
    [buildForecastChart]
  );

  const temperatureForecastData = useMemo(
    () => buildForecastChart("temp", "temperature"),
    [buildForecastChart]
  );

  const humidityForecastData = useMemo(
    () => buildForecastChart("humidity", "humidity"),
    [buildForecastChart]
  );

  // --------------------------------------------------
  // 10. SEARCH FILTER
  // --------------------------------------------------

  const filtered = useMemo(() => {
    const query = search.trim().toLowerCase();

    if (!query) return readings;

    return readings.filter((row) => {
      const searchable = [
        row.timestamp,
        row.time,
        row.tds,
        row.temp,
        row.humidity
      ]
        .map((value) => value ?? "")
        .join(" ")
        .toLowerCase();

      return searchable.includes(query);
    });
  }, [readings, search]);
  
  const filteredHistory = useMemo(() => {
  const query = search.trim().toLowerCase();

  if (!query) return historyReadings;

  return historyReadings.filter((row) => {
    const searchable = [
      row.timestamp,
      row.time,
      row.tds,
      row.temp,
      row.humidity
    ]
      .map((value) => value ?? "")
      .join(" ")
      .toLowerCase();

    return searchable.includes(query);
  });
}, [historyReadings, search]);
  // --------------------------------------------------
  // 11. NAVIGATION
  // --------------------------------------------------

  const title =
    active === "Dashboard"
      ? "Water Quality Overview"
      : active;

  const navTo = (label) => {
    setActive(label);
    setMobileNav(false);
  };

  // --------------------------------------------------
  // 12. BOREMAN CHATBOT PREVIEW
  // --------------------------------------------------

  
const sendChat = async (question = chatInput) => {
  const message = question.trim();

  if (!message) return;

  setChatInput("");

  setChatMessages((previous) => [
    ...previous,
    {
      role: "user",
      content: message
    }
  ]);

  try {
    const response = await fetch(`${API_BASE_URL}/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        question: message,
        context: {
          source: sourceLabel,
          latest_readings: {
            tds: tdsValue,
            temperature: tempValue,
            humidity: humidityValue
          },
          forecasts: forecasts,
          loaded_readings: readings.length,
          timezone: TIME_ZONE
        }
      })
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(
        data.detail || `API returned HTTP ${response.status}`
      );
    }

    setChatMessages((previous) => [
      ...previous,
      {
        role: "assistant",
        content: data.answer || "The BOREMAN API returned an empty response."
      }
    ]);
  } catch (err) {
    setChatMessages((previous) => [
      ...previous,
      {
        role: "assistant",
        content: `I could not connect to the BOREMAN assistant service: ${err.message}`
      }
    ]);
  }
};


  // --------------------------------------------------
  // 13. MAIN DASHBOARD JSX STARTS IN PART 3
  // --------------------------------------------------

  return (
    <div className="app-shell">
      {/* SIDEBAR */}
      <aside className={`sidebar ${mobileNav ? "sidebar-open" : ""}`}>
        <div className="brand">
          <div className="brand-mark">
            <Droplets size={25} />
          </div>
          <div>
            <div className="brand-name">BOREMAN</div>
            <div className="brand-sub">Water quality monitoring</div>
          </div>
          <button
            className="mobile-close"
            onClick={() => setMobileNav(false)}
            aria-label="Close navigation"
          >
            <X size={18} />
          </button>
        </div>

        <div className="site-selector">
          <div className="site-avatar">
            <Waves size={19} />
          </div>
          <div className="site-text">
            <span>Monitoring site</span>
            <strong>Borewell 01</strong>
          </div>
          <ChevronDown size={16} />
        </div>

        <div className="nav-caption">WORKSPACE</div>

        <nav>
          {NAV.map(({ label, icon: Icon }) => (
            <button
              key={label}
              className={`nav-item ${active === label ? "active" : ""}`}
              onClick={() => navTo(label)}
            >
              <Icon size={18} />
              <span>{label}</span>
              {label === "Live readings" && (
                <span className="nav-live-dot" />
              )}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="connection-card">
            <div className="connection-heading">
              <span
                className={`connection-indicator ${
                  source === "live" && !error ? "connected" : ""
                }`}
              />
              {error
                ? "Connection issue"
                : source === "live"
                  ? "Live data mode"
                  : "Demo data active"}
            </div>

            <p>
              {sourceLabel}. Live sensor data is supplied by
              the BOREMAN API.
            </p>

            <button onClick={() => navTo("Settings")}>
              <Settings size={14} />
              Data source settings
            </button>
          </div>

          <div className="profile">
            <div className="profile-avatar">B</div>
            <div className="profile-text">
              <strong>Administrator</strong>
              <span>Project workspace</span>
            </div>
            <ChevronDown size={15} />
          </div>
        </div>
      </aside>

      {/* MAIN CONTENT */}
      <main className="main-content">
        <header className="topbar">
          <div className="topbar-left">
            <button
              className="mobile-menu"
              onClick={() => setMobileNav(true)}
              aria-label="Open navigation"
            >
              <Menu size={21} />
            </button>

            <div>
              <div className="breadcrumb">BOREMAN / WORKSPACE</div>
              <h1>{title}</h1>
            </div>
          </div>

          <div className="topbar-actions">
            <div className="last-updated">
              <Clock3 size={15} />
              <span>
                {lastUpdated
                  ? `Updated ${formatTime(lastUpdated)}`
                  : "Waiting for data"}
              </span>
            </div>

            <button
              className="icon-button"
              onClick={fetchLive}
              disabled={loading}
              title="Refresh live data"
            >
              <RefreshCw
                size={17}
                className={loading ? "spin" : ""}
              />
            </button>

            <button
              className="icon-button"
              onClick={() => setChatOpen(true)}
              title="Open assistant"
            >
              <MessageCircle size={18} />
            </button>
          </div>
        </header>

        {/* ERROR MESSAGE */}
        {error && (
          <div className="alert-banner">
            <AlertTriangle size={18} />
            <span>{error}</span>
            <button onClick={fetchLive}>Retry</button>
          </div>
        )}

        {/* DASHBOARD */}
        {active === "Dashboard" && (
          <>
            <section className="section-intro">
              <div>
                <h2>Water quality overview</h2>
                <p>
                  Monitor borewell sensor measurements and
                  review recent environmental conditions.
                </p>
              </div>

              <div className="intro-actions">
                <span className={`source-badge ${source}`}>
                  <Radio size={14} />
                  {sourceLabel}
                </span>

                <button
                  className="primary-button"
                  onClick={fetchLive}
                  disabled={loading}
                >
                  <RefreshCw size={15} />
                  {loading ? "Refreshing..." : "Refresh data"}
                </button>
              </div>
            </section>

            <div className="section-note">
              <CircleHelp size={17} />
              <span>
                This dashboard displays sensor observations and
                experimental forecasts. It is a research
                monitoring tool, not a drinking-water
                certification system.
              </span>
            </div>

            <section className="metrics-grid">
              <MetricCard
                icon={Droplets}
                label="Total dissolved solids"
                value={
                  tdsValue == null
                    ? null
                    : Number(tdsValue).toFixed(1)
                }
                unit="ppm"
                status={tdsStatus}
                color="blue"
                detail="Latest sensor reading"
                tag="TDS"
              />

              <MetricCard
                icon={Thermometer}
                label="Water temperature"
                value={
                  tempValue == null
                    ? null
                    : Number(tempValue).toFixed(1)
                }
                unit="°C"
                status={tempStatus}
                color="orange"
                detail="Latest sensor reading"
                tag="TEMP"
              />

              <MetricCard
                icon={Activity}
                label="Ambient humidity"
                value={
                  humidityValue == null
                    ? null
                    : Number(humidityValue).toFixed(1)
                }
                unit="%"
                status={humidityStatus}
                color="purple"
                detail="Environmental sensor"
                tag="HUMIDITY"
              />

              <MetricCard
                icon={Gauge}
                label="Monitoring status"
                value={readings.length ? "Active" : "Waiting"}
                unit=""
                status={overall}
                color="green"
                detail={`${readings.length} readings loaded`}
                tag="STATUS"
              />
            </section>

            <section className="secondary-grid">
              <article className="panel">
                <PanelHeader
                  icon={TrendingUp}
                  title="Recent TDS readings"
                  subtitle="Recent measured values in ppm"
                />

                <div className="large-chart">
                  {trendData.length ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={trendData}>
                        <defs>
                          <linearGradient
                            id="tdsGradient"
                            x1="0"
                            y1="0"
                            x2="0"
                            y2="1"
                          >
                            <stop
                              offset="0%"
                              stopColor="#2675ef"
                              stopOpacity={0.25}
                            />
                            <stop
                              offset="100%"
                              stopColor="#2675ef"
                              stopOpacity={0.02}
                            />
                          </linearGradient>
                        </defs>

                        <CartesianGrid
                          stroke="#e9eef5"
                          strokeDasharray="3 4"
                          vertical={false}
                        />

                        <XAxis dataKey="time" />
                        <YAxis />
                        <Tooltip />

                        <Area
                          type="monotone"
                          dataKey="tds"
                          name="TDS"
                          stroke="#2675ef"
                          fill="url(#tdsGradient)"
                          connectNulls={false}
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="empty-state">
                      No TDS readings are currently available.
                    </div>
                  )}
                </div>
              </article>

              <article className="panel">
                <PanelHeader
                  icon={Database}
                  title="Reading summary"
                  subtitle="Summary of currently loaded observations"
                />

                <div className="detail-stats">
                  <DetailStat
                    icon={Droplets}
                    label="Average TDS"
                    value={
                      averageTds == null
                        ? "N/A"
                        : `${averageTds.toFixed(1)} ppm`
                    }
                  />

                  <DetailStat
                    icon={Thermometer}
                    label="Average temperature"
                    value={
                      averageTemp == null
                        ? "N/A"
                        : `${averageTemp.toFixed(1)} °C`
                    }
                  />

                  <DetailStat
                    icon={Activity}
                    label="Average ambient humidity"
                    value={
                      averageHumidity == null
                        ? "N/A"
                        : `${averageHumidity.toFixed(1)}%`
                    }
                  />

                  <DetailStat
                    icon={History}
                    label="Loaded observations"
                    value={readings.length}
                  />
                </div>

                <div className="panel-footer">
                  <span>Data source</span>
                  <strong>{sourceLabel}</strong>
                </div>
              </article>
            </section>

            <article className="panel">
              <PanelHeader
                icon={History}
                title="Recent sensor observations"
                subtitle="Latest available records from the selected source"
                action={
                  <button
                    className="text-button"
                    onClick={() => navTo("Live readings")}
                  >
                    View all
                    <ArrowUpRight size={15} />
                  </button>
                }
              />

              <ReadingsTable
                readings={readings.slice(-8).reverse()}
                source={source}
                showTimestamp
              />
            </article>

            <section className="section-note">
              <ShieldCheck size={17} />
              <span>
                TDS and temperature display thresholds are
                dashboard review settings only. They do not
                determine whether water is safe to drink.
              </span>
            </section>
          </>
        )}

        {/* LIVE READINGS */}
        {active === "Live readings" && (
          <>
            <section className="section-intro">
              <div>
                <h2>Sensor readings explorer</h2>
                <p>
                  Search, inspect and compare readings from
                  the selected data source.
                </p>
              </div>

              <div className="intro-actions">
                <button
                  className="primary-button"
                  onClick={exportCsv}
                  disabled={!readings.length}
                >
                  <Download size={15} />
                  Export CSV
                </button>
              </div>
            </section>

            <section className="panel">
              <PanelHeader
                icon={Search}
                title="Search observations"
                subtitle="Search timestamps and measured sensor values"
              />

              <div className="search-row">
                <div className="search-box">
                  <Search size={17} />
                  <input
                    value={search}
                    onChange={(event) =>
                      setSearch(event.target.value)
                    }
                    placeholder="Search TDS, temperature, humidity or timestamp..."
                  />
                  {search && (
                    <button
                      onClick={() => setSearch("")}
                      aria-label="Clear search"
                    >
                      <X size={16} />
                    </button>
                  )}
                </div>

                <span className="result-count">
                  {filteredHistory.length} records
                </span>
              </div>

              <ReadingsTable
                readings={[...filteredHistory].reverse()}
                source="live"
                showTimestamp
              />

              {!filteredHistory.length && (
                <div className="empty-state">
                  No matching readings found.
                </div>
              )}
            </section>

            <section className="metrics-grid">
              <MetricCard
                icon={Droplets}
                label="Average TDS"
                value={
                  averageTds == null
                    ? null
                    : averageTds.toFixed(1)
                }
                unit="ppm"
                status="Based on loaded records"
                color="blue"
                detail="Historical average"
                tag="AVG"
              />

              <MetricCard
                icon={Thermometer}
                label="Average water temperature"
                value={
                  averageTemp == null
                    ? null
                    : averageTemp.toFixed(1)
                }
                unit="°C"
                status="Based on loaded records"
                color="orange"
                detail="Historical average"
                tag="AVG"
              />

              <MetricCard
                icon={Activity}
                label="Average ambient humidity"
                value={
                  averageHumidity == null
                    ? null
                    : averageHumidity.toFixed(1)
                }
                unit="%"
                status="Based on loaded records"
                color="purple"
                detail="Historical average"
                tag="AVG"
              />

              <MetricCard
                icon={Database}
                label="Available observations"
                value={readings.length}
                unit=""
                status="Loaded records"
                color="green"
                detail={sourceLabel}
                tag="COUNT"
              />
            </section>
          </>
        )}

        {/* PREDICTIONS SECTION CONTINUES IN PART 4 */}

        {/* PREDICTIONS */}
        {active === "Predictions" && (
          <>
            <section className="section-intro">
              <div>
                <h2>AI-assisted sensor forecasts</h2>
                <p>
                  Forecasts generated by the BOREMAN backend
                  using the latest sensor readings.
                </p>
              </div>

              <span className="ai-badge">
                {Object.keys(forecasts).length
                  ? "FORECASTS AVAILABLE"
                  : "WAITING FOR API"}
              </span>
            </section>

            <div className="section-note">
              <CircleHelp size={17} />
              <span>
                Forecasts are generated using the latest available sensor observations
and the configured forecasting models. Different forecasting methods
are used depending on the sensor and forecast horizon. Forecasts provide
monitoring insights and are not a substitute for laboratory water-quality testing.
              </span>
            </div>

            <section className="metrics-grid">
              {[
                { key: "tds_5m", name: "TDS", horizon: "5 minutes", unit: "ppm" },
                { key: "tds_15m", name: "TDS", horizon: "15 minutes", unit: "ppm" },
                { key: "tds_30m", name: "TDS", horizon: "30 minutes", unit: "ppm" },
                { key: "tds_60m", name: "TDS", horizon: "60 minutes", unit: "ppm" },

                { key: "temperature_5m", name: "Temperature", horizon: "5 minutes", unit: "°C" },
                { key: "temperature_15m", name: "Temperature", horizon: "15 minutes", unit: "°C" },
                { key: "temperature_30m", name: "Temperature", horizon: "30 minutes", unit: "°C" },
                { key: "temperature_60m", name: "Temperature", horizon: "60 minutes", unit: "°C" },

                { key: "humidity_5m", name: "Humidity", horizon: "5 minutes", unit: "%" },
                { key: "humidity_15m", name: "Humidity", horizon: "15 minutes", unit: "%" },
                { key: "humidity_30m", name: "Humidity", horizon: "30 minutes", unit: "%" },
                { key: "humidity_60m", name: "Humidity", horizon: "60 minutes", unit: "%" }
              ].map((item) => {
  const forecast = forecasts[item.key];
  const value = forecast?.value;
  const isRidge = forecast?.method === "RIDGE_SHADOW";

  const policyTarget =
    item.name === "Temperature"
      ? "Temperature"
      : item.name.toLowerCase();

  const horizon = item.key.split("_")[1].replace("m", "");

  const configuredMethod =
    modelStatus?.deployment_policy?.[policyTarget]?.[horizon] || null;

                return (
                  <article className="metric-card blue" key={item.key}>
                    <div className="metric-icon">
                      <TrendingUp size={21} />
                    </div>

                    <div className="metric-content">
                      <div className="metric-label">
                        {item.name} · {item.horizon}
                      </div>

                      <div className="metric-value">
                        {value == null
                          ? "N/A"
                          : Number(value).toFixed(
                              item.unit === "ppm" ? 1 : 2
                            )}
                        <span>{item.unit}</span>
                      </div>

                      <div className="metric-foot">
  <span className="status-dot" />
  {forecast
    ? isRidge
      ? "Ridge model"
      : "Persistence baseline"
    : "Forecast unavailable"}
</div>

<div className="metric-foot">
  Configured policy: {configuredMethod || "Not available"}
</div>

{forecast?.method && configuredMethod &&
  forecast.method !== configuredMethod && (
    <div className="metric-foot">
      <span className="status-dot" />
      Method differs from configured policy
    </div>
  )}
                    </div>

                    <div className="metric-change">
                      {isRidge
                        ? "RIDGE"
                        : forecast
                          ? "BASELINE"
                          : "N/A"}
                    </div>
                  </article>
                );
              })}
            </section>

            <section className="forecast-charts">
              <article className="panel report-chart-panel">
                <PanelHeader
                  icon={Droplets}
                  title="TDS forecast"
                  subtitle="Historical readings and future estimates (ppm)"
                />

                <div className="large-chart">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={tdsForecastData}>
                      <CartesianGrid
                        stroke="#e9eef5"
                        strokeDasharray="3 4"
                        vertical={false}
                      />
                      <XAxis dataKey="time" />
                      <YAxis />
                      <Tooltip />

                      <Line
                        dataKey="actual"
                        name="Historical TDS"
                        stroke="#2675ef"
                        strokeWidth={2.5}
                        dot={false}
                        connectNulls
                      />

                      <Line
                        dataKey="forecast"
                        name="Forecast TDS"
                        stroke="#16a085"
                        strokeWidth={2.5}
                        strokeDasharray="6 4"
                        connectNulls
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </article>

              <article className="panel report-chart-panel">
                <PanelHeader
                  icon={Thermometer}
                  title="Temperature forecast"
                  subtitle="Historical readings and future estimates (°C)"
                />

                <div className="large-chart">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={temperatureForecastData}>
                      <CartesianGrid
                        stroke="#e9eef5"
                        strokeDasharray="3 4"
                        vertical={false}
                      />
                      <XAxis dataKey="time" />
                      <YAxis />
                      <Tooltip />

                      <Line
                        dataKey="actual"
                        name="Historical temperature"
                        stroke="#f59e42"
                        strokeWidth={2.5}
                        dot={false}
                        connectNulls
                      />

                      <Line
                        dataKey="forecast"
                        name="Forecast temperature"
                        stroke="#16a085"
                        strokeWidth={2.5}
                        strokeDasharray="6 4"
                        connectNulls
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </article>

              <article className="panel report-chart-panel">
                <PanelHeader
                  icon={Activity}
                  title="Environmental humidity forecast"
                  subtitle="Historical readings and future estimates (%)"
                />

                <div className="large-chart">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={humidityForecastData}>
                      <CartesianGrid
                        stroke="#e9eef5"
                        strokeDasharray="3 4"
                        vertical={false}
                      />
                      <XAxis dataKey="time" />
                      <YAxis />
                      <Tooltip />

                      <Line
                        dataKey="actual"
                        name="Historical humidity"
                        stroke="#8a68e8"
                        strokeWidth={2.5}
                        dot={false}
                        connectNulls
                      />

                      <Line
                        dataKey="forecast"
                        name="Forecast humidity"
                        stroke="#16a085"
                        strokeWidth={2.5}
                        strokeDasharray="6 4"
                        connectNulls
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </article>
            </section>

            <section className="forecast-charts">
              <article className="panel report-chart-panel">
                <PanelHeader
                  icon={Activity}
                  title="Selected model vs persistence — MAE"
                  subtitle="Model performance comparison across forecast horizons"
                />

                <div className="large-chart">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={step40ChartData}>
                      <CartesianGrid
                        stroke="#e9eef5"
                        strokeDasharray="3 4"
                        vertical={false}
                      />
                      <XAxis dataKey="horizon" />
                      <YAxis />
                      <Tooltip />

                      <Bar dataKey="tds_selected_mae" name="TDS selected" fill="#2675ef" />
                      <Bar dataKey="tds_persistence_mae" name="TDS persistence" fill="#9bbcf0" />

                      <Bar dataKey="humidity_selected_mae" name="Humidity selected" fill="#8a68e8" />
                      <Bar dataKey="humidity_persistence_mae" name="Humidity persistence" fill="#c3b3f2" />

                      <Bar dataKey="temperature_selected_mae" name="Temperature selected" fill="#f59e42" />
                      <Bar dataKey="temperature_persistence_mae" name="Temperature persistence" fill="#f8d0a0" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </article>

              <article className="panel report-chart-panel">
                <PanelHeader
                  icon={Activity}
                  title="Selected model vs persistence — RMSE"
                  subtitle="Step 40 frozen candidate-versus-persistence comparison"
                />

                <div className="large-chart">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={step40ChartData}>
                      <CartesianGrid
                        stroke="#e9eef5"
                        strokeDasharray="3 4"
                        vertical={false}
                      />
                      <XAxis dataKey="horizon" />
                      <YAxis />
                      <Tooltip />

                      <Bar dataKey="tds_selected_rmse" name="TDS selected" fill="#2675ef" />
                      <Bar dataKey="tds_persistence_rmse" name="TDS persistence" fill="#9bbcf0" />

                      <Bar dataKey="humidity_selected_rmse" name="Humidity selected" fill="#8a68e8" />
                      <Bar dataKey="humidity_persistence_rmse" name="Humidity persistence" fill="#c3b3f2" />

                      <Bar dataKey="temperature_selected_rmse" name="Temperature selected" fill="#f59e42" />
                      <Bar dataKey="temperature_persistence_rmse" name="Temperature persistence" fill="#f8d0a0" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </article>
            </section>

            <div className="section-note">
              <CircleHelp size={17} />
              <span>
                Step 40 frozen candidate-versus-persistence test
                results. These comparisons do not mean every
                candidate is deployed. Only three test origin
                dates were observed.
              </span>
            </div>

            <article className="panel report-chart-panel">
              <PanelHeader
                icon={Activity}
                title="Actual vs predicted values"
                subtitle="Historical frozen test-set evaluation"
              />

              <div className="intro-actions" style={{ marginBottom: 16 }}>
                <label className="field-label">
                  Sensor
                  <select
                    className="text-input"
                    value={selectedEvalTarget}
                    onChange={(event) =>
                      setSelectedEvalTarget(event.target.value)
                    }
                  >
                    <option value="tds">TDS</option>
                    <option value="humidity">Environmental humidity</option>
                    <option value="temperature">Temperature</option>
                  </select>
                </label>

                <label className="field-label">
                  Forecast horizon
                  <select
                    className="text-input"
                    value={selectedEvalHorizon}
                    onChange={(event) =>
                      setSelectedEvalHorizon(event.target.value)
                    }
                  >
                    <option value="5">5 minutes</option>
                    <option value="15">15 minutes</option>
                    <option value="30">30 minutes</option>
                    <option value="60">60 minutes</option>
                  </select>
                </label>
              </div>

              <div className="chart-legend">
                <span>Actual observed</span>
                <span>Frozen model prediction</span>
                <span>Persistence baseline</span>
              </div>

              <div className="large-chart">
                {actualVsPredictedData.length ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={actualVsPredictedData}>
                      <CartesianGrid
                        stroke="#e9eef5"
                        strokeDasharray="3 4"
                        vertical={false}
                      />
                      <XAxis dataKey="time" />
                      <YAxis />
                      <Tooltip />

                      <Line
                        dataKey="actual"
                        name="Actual observed"
                        stroke="#2675ef"
                        strokeWidth={2.5}
                        dot={false}
                      />

                      <Line
                        dataKey="prediction"
                        name="Frozen model prediction"
                        stroke="#16a085"
                        strokeWidth={2}
                        dot={false}
                      />

                      <Line
                        dataKey="persistence"
                        name="Persistence baseline"
                        stroke="#f59e42"
                        strokeWidth={2}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="empty-state">
                    No evaluation records available for this selection.
                  </div>
                )}
              </div>

              <div className="section-note">
                <CircleHelp size={17} />
                <span>
                  These are sampled historical test predictions,
                  not live readings or proof of future accuracy.
                  Test coverage includes only three distinct
                  origin dates. Humidity represents ambient
                  environmental humidity, not water humidity.
                </span>
              </div>
            </article>

            <article className="panel prediction-panel">
  <PanelHeader
    icon={ShieldCheck}
    title="Model information"
    subtitle="Forecast methodology and operational safeguards"
  />

  <div className="formula-card">
    <strong>Ridge regression</strong>
    <p>
      Uses trained machine-learning models to estimate
      future temperature and humidity values at supported
      forecast horizons.
    </p>
  </div>

  <div className="formula-card">
    <strong>Persistence baseline</strong>
    <p>
      Uses the latest available sensor measurement as a
      reference forecast where this method is selected.
    </p>
  </div>

  <div className="formula-card">
    <strong>Data quality monitoring</strong>
    <p>
      Monitors sensor observations, input availability,
      and data quality to support reliable forecasting.
    </p>
  </div>

  <div className="formula-card">
    <strong>Operational safeguards</strong>
    <p>
      Uses fallback forecasting when required inputs are
      unavailable, stale, outside supported ranges,
      or when feature history is incomplete.
    </p>
  </div>

  <div className="formula-card">
    <strong>Water quality guidance</strong>
    <p>
      Forecasts provide environmental monitoring insights.
      They should not be used as a substitute for laboratory
      water-quality testing or as an automatic water-safety
      classification.
    </p>
  </div>

  <div className="model-foot">
    <span className="model-status-dot" />
    Forecast values are generated by the connected BOREMAN backend API.
  </div>
</article>
          </>
        )}

        {/* REPORTS */}
        {active === "Reports" && (
          <>
            <section className="section-intro">
              <div>
                <h2>Reports and historical data</h2>
                <p>
                  Export the currently loaded dataset or review
                  its summary statistics.
                </p>
              </div>

              <button
                className="button-primary"
                onClick={exportCsv}
                disabled={!readings.length}
              >
                <Download size={15} />
                Export CSV
              </button>
            </section>

            <section className="metrics-grid">
              <MetricCard
                icon={Database}
                label="Total observations"
                value={historyReadings.length}
                unit=""
                status="Currently loaded"
                color="blue"
                detail={sourceLabel}
                tag="COUNT"
              />

              <MetricCard
                icon={Droplets}
                label="Average TDS"
                value={
                mean(historyReadings, "tds") == null
                  ? null
                  : mean(historyReadings, "tds").toFixed(1)
              }
                unit="ppm"
                status="Loaded records"
                color="blue"
                detail="Historical average"
                tag="AVG"
              />

              <MetricCard
                icon={Thermometer}
                label="Average temperature"
                value={
                mean(historyReadings, "temp") == null
                  ? null
                  : mean(historyReadings, "temp").toFixed(1)
              }
                unit="°C"
                status="Loaded records"
                color="orange"
                detail="Historical average"
                tag="AVG"
              />

              <MetricCard
                icon={Activity}
                label="Average ambient humidity"
                value={
                mean(historyReadings, "humidity") == null
                  ? null
                  : mean(historyReadings, "humidity").toFixed(1)
              }
                unit="%"
                status="Loaded records"
                color="purple"
                detail="Historical average"
                tag="AVG"
              />
            </section>

            <article className="panel">
              <PanelHeader
                icon={FileSpreadsheet}
                title="Historical readings"
                subtitle="Records currently available in the dashboard"
              />

              <ReadingsTable
                readings={[...historyReadings].reverse()}
                source="live"
                showTimestamp
              />
            </article>

            <div className="section-note">
              <CircleHelp size={17} />
              <span>
                This report exports the records currently loaded
                in the frontend. It does not independently
                retrieve the complete historical database.
              </span>
            </div>
          </>
        )}

        {/* SETTINGS */}
        {active === "Settings" && (
          <>
            <section className="section-intro">
              <div>
                <h2>Data source and dashboard settings</h2>
                <p>
                  Monitor the live BOREMAN API connection and
                  configure dashboard preferences.
                </p>
              </div>

              <span className={`device-badge ${error ? "demo" : "good"}`}>
                <span />
                {error ? "Connection issue" : sourceLabel}
              </span>
            </section>

            <article className="panel">
              <PanelHeader
                icon={CloudDownload}
                title="API connection"
                subtitle="Connection to the BOREMAN backend"
              />

              <div className="settings-row">
                <div>
                  <strong>API base URL</strong>
                  <p>{API_BASE_URL}</p>
                </div>
                <span className="settings-status">
                  {error ? "Check connection" : "Configured"}
                </span>
              </div>

              <div className="settings-row">
                <div>
                  <strong>Current data source</strong>
                  <p>{sourceLabel}</p>
                </div>
                <span className="settings-status">
                  {source === "live" ? "Live mode" : "Demo mode"}
                </span>
              </div>

              <div className="settings-row">
                <div>
                  <strong>Last successful update</strong>
                  <p>
                    {lastUpdated
                      ? formatTime(lastUpdated)
                      : "No successful update recorded"}
                  </p>
                </div>
              </div>

              {error && (
                <div className="alert-banner">
                  <AlertTriangle size={17} />
                  {error}
                </div>
              )}

              <div className="intro-actions">
                <button
                className="button-primary"
                onClick={checkBackendHealth}
                disabled={healthLoading}
              >
                <RefreshCw size={15} />
                {healthLoading ? "Checking..." : "Check backend health"}
              </button>

                <button
                  className="button-secondary"
                  onClick={switchToDemo}
                >
                  Use sample data
                </button>
              </div>

              <div className="settings-row">
  <div>
    <strong>Backend health</strong>
    <p>
      {healthStatus
        ? healthStatus.status
        : healthLoading
          ? "Checking..."
          : healthError || "Not checked"}
    </p>
  </div>
  <span className="settings-status">
    {healthStatus?.status === "healthy"
      ? "Healthy"
      : "Unavailable"}
  </span>
</div>

<div className="settings-row">
  <div>
    <strong>Forecast models loaded</strong>
    <p>
      {healthStatus
        ? `${healthStatus.models_loaded} / ${healthStatus.models_expected}`
        : "Not available"}
    </p>
  </div>
  <span className="settings-status">
    {healthStatus?.models_loaded === healthStatus?.models_expected
      ? "All loaded"
      : "Check models"}
  </span>
</div>

<div className="settings-row">
  <div>
    <strong>Adafruit IO configuration</strong>
    <p>
      {healthStatus?.adafruit_configured
        ? "Configured"
        : "Not configured"}
    </p>
  </div>
</div>

<div className="settings-row">
  <div>
    <strong>Model version</strong>
    <p>
      {modelStatus?.model_version || "Not available"}
    </p>
  </div>
</div>

<div className="settings-row">
  <div>
    <strong>Backend timezone</strong>
    <p>
      {modelStatus?.timezone || "Not available"}
    </p>
  </div>
</div>
            </article>

            <article className="panel">
              <PanelHeader
                icon={SlidersHorizontal}
                title="Dashboard preferences"
                subtitle="Configure refresh interval and display thresholds"
              />

              <div className="settings-grid">
                <label className="field-label">
                  Automatic refresh interval
                  <select
                    className="text-input"
                    value={refreshSeconds}
                    onChange={(event) =>
                      setRefreshSeconds(event.target.value)
                    }
                  >
                    <option value="15">15 seconds</option>
                    <option value="30">30 seconds</option>
                    <option value="60">60 seconds</option>
                    <option value="120">2 minutes</option>
                    <option value="300">5 minutes</option>
                  </select>
                </label>

                <label className="field-label">
                  TDS display threshold (ppm)
                  <input
                    className="text-input"
                    type="number"
                    min="1"
                    value={thresholdTds}
                    onChange={(event) =>
                      setThresholdTds(event.target.value)
                    }
                  />
                </label>

                <label className="field-label">
                  Temperature display threshold (°C)
                  <input
                    className="text-input"
                    type="number"
                    min="1"
                    value={thresholdTemp}
                    onChange={(event) =>
                      setThresholdTemp(event.target.value)
                    }
                  />
                </label>
              </div>

              <div className="section-note">
                <CircleHelp size={17} />
                <span>
                  These thresholds control dashboard display
                  indicators only. They are not regulatory limits
                  and must not be used to certify drinking-water
                  safety.
                </span>
              </div>
            </article>

            <article className="panel">
              <PanelHeader
                icon={Radio}
                title="Sensor configuration"
                subtitle="Available direct measurement feeds"
              />

              <div className="settings-row">
                <div>
                  <strong>TDS</strong>
                  <p>Total dissolved solids sensor reading</p>
                </div>
                <span className="settings-status">Available</span>
              </div>

              <div className="settings-row">
                <div>
                  <strong>Water temperature</strong>
                  <p>Water temperature sensor reading</p>
                </div>
                <span className="settings-status">Available</span>
              </div>

              <div className="settings-row">
                <div>
                  <strong>Ambient humidity</strong>
                  <p>Environmental humidity sensor reading</p>
                </div>
                <span className="settings-status">Available</span>
              </div>

              <div className="settings-row">
                <div>
                  <strong>Electrical conductivity</strong>
                  <p>No separate conductivity feed is configured.</p>
                </div>
                <span className="settings-status">Not connected</span>
              </div>
            </article>
          </>
        )}

        <footer className="footer">
          <span>BOREMAN Water Quality Monitoring System</span>
          <span>
            Research dashboard · Experimental forecasts
          </span>
        </footer>
      </main>

      {mobileNav && (
        <button
          className="mobile-backdrop"
          aria-label="Close menu"
          onClick={() => setMobileNav(false)}
        />
      )}

      {/* CHATBOT */}
      {chatOpen && (
        <section className="chat-panel" aria-label="BOREMAN chatbot preview">
          <header className="chat-header">
            <div className="chat-avatar">
              <Bot size={20} />
            </div>

            <div className="chat-heading">
              <strong>BOREMAN Assistant</strong>
              <span>
                <i />
                Chatbot preview
              </span>
            </div>

            <button
              className="icon-button"
              onClick={() => setChatOpen(false)}
              aria-label="Close assistant"
            >
              <X size={18} />
            </button>
          </header>

          <div className="chat-context">
            <span className="chat-context-dot" />
            Using {sourceLabel.toLowerCase()}
          </div>

          <div className="chat-messages" aria-live="polite">
            {chatMessages.map((message, index) => (
              <div
                className={`chat-message-row ${message.role}`}
                key={index}
              >
                <div className="chat-message-icon">
                  {message.role === "assistant"
                    ? <Bot size={15} />
                    : "You"}
                </div>

                <div className={`chat-bubble ${message.role}`}>
                  {message.content}
                </div>
              </div>
            ))}
          </div>

          {chatMessages.length === 1 && (
            <div className="chat-suggestions">
              <span>Try asking</span>

              <button onClick={() => sendChat("What does TDS mean?")}>
                What does TDS mean?
              </button>

              <button onClick={() => sendChat("What is the latest temperature?")}>
                Latest temperature?
              </button>

              <button onClick={() => sendChat("Explain the forecasts")}>
                Explain the forecasts
              </button>
            </div>
          )}

          <form
            className="chat-composer"
            onSubmit={(event) => {
              event.preventDefault();
              sendChat();
            }}
          >
            <input
              value={chatInput}
              onChange={(event) => setChatInput(event.target.value)}
              placeholder="Ask about BOREMAN readings..."
            />

            <button
              type="submit"
              disabled={!chatInput.trim()}
              aria-label="Send message"
            >
              <Send size={17} />
            </button>
          </form>

          <p className="chat-disclaimer">
            Informational guidance only. Sensor readings do
            not confirm drinking-water safety.
          </p>
        </section>
      )}

      {!chatOpen && (
        <button
          className="chat-launcher"
          onClick={() => setChatOpen(true)}
          aria-label="Open BOREMAN assistant"
        >
          <MessageCircle size={22} />
          <span>Ask BOREMAN</span>
          <i />
        </button>
      )}

      {toast && (
        <div className="toast">
          <CheckCircle2 size={17} />
          {toast}
        </div>
      )}
    </div>
  );
}

// --------------------------------------------------
// REUSABLE DETAIL STAT COMPONENT
// --------------------------------------------------

function DetailStat({ label, value, icon: Icon }) {
  return (
    <article className="detail-stat">
      <div className="detail-stat-icon">
        <Icon size={19} />
      </div>

      <div>
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
    </article>
  );
}

// --------------------------------------------------
// REUSABLE READINGS TABLE
// --------------------------------------------------

function ReadingsTable({
  readings,
  source,
  showTimestamp = false
}) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Timestamp</th>
            <th>TDS (ppm)</th>
            <th>Humidity (%)</th>
            <th>Water temperature (°C)</th>
            <th>Source</th>
          </tr>
        </thead>

        <tbody>
          {readings.map((row, index) => (
            <tr key={`${row.timestamp || row.time}-${index}`}>
              <td>
                {showTimestamp
                  ? formatTime(row.timestamp || row.time)
                  : row.time || formatTime(row.timestamp)}
              </td>

              <td>
                {row.tds == null
                  ? "N/A"
                  : Number(row.tds).toFixed(1)}
              </td>

              <td>
                {row.humidity == null
                  ? "N/A"
                  : Number(row.humidity).toFixed(1)}
              </td>

              <td>
                {row.temp == null
                  ? "N/A"
                  : Number(row.temp).toFixed(1)}
              </td>

              <td>
                <span className="table-source">
                  {source === "live"
                    ? "Adafruit IO"
                    : "Sample data"}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {!readings.length && (
        <div className="empty-state">
          No readings available.
        </div>
      )}
    </div>
  );
}

export default App;
