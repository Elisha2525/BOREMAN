import React, { useEffect, useMemo, useState } from "react";
//import LocationSettings from "./components/LocationSettings";
import {
  Activity, AlertTriangle, Bell, CheckCircle2, ChevronDown, CircleHelp,
  Droplets, Gauge, LayoutDashboard, Menu, Radio, Settings, ShieldCheck,
  Thermometer, Waves, Zap, Clock3, BrainCircuit, Download, Wifi,
  RefreshCw, MapPin, SlidersHorizontal, ArrowUpRight, FlaskConical,
  FileSpreadsheet, Database, CloudDownload, TrendingUp, History, Save,
  ExternalLink, X, Search, CalendarDays, CircleGauge, MessageCircle, Send, Bot, Minimize2
} from "lucide-react";
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Line, LineChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis
} 

from "recharts";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:9000";
const SHEET_CSV_URL = "";const DEMO_READINGS = [
  { time: "00:00", tds: 332, ec: 518, temp: 24.1 },
  { time: "02:00", tds: 326, ec: 509, temp: 24.0 },
  { time: "04:00", tds: 329, ec: 514, temp: 23.8 },
  { time: "06:00", tds: 336, ec: 525, temp: 24.2 },
  { time: "08:00", tds: 341, ec: 533, temp: 24.5 },
  { time: "10:00", tds: 338, ec: 529, temp: 24.7 },
  { time: "12:00", tds: 344, ec: 538, temp: 24.8 },
  { time: "14:00", tds: 347, ec: 542, temp: 24.6 },
  { time: "16:00", tds: 343, ec: 536, temp: 24.5 },
  { time: "18:00", tds: 339, ec: 530, temp: 24.3 },
  { time: "20:00", tds: 341, ec: 534, temp: 24.2 },
  { time: "22:00", tds: 344, ec: 538, temp: 24.4 }
];

const NAV = [
  { label: "Dashboard", icon: LayoutDashboard, desc: "Overview of the latest borewell readings." },
  { label: "Live readings", icon: Activity, desc: "Inspect the latest sensor values and recent history." },
  { label: "Predictions", icon: BrainCircuit, desc: "View trend-based estimates and derived indicators." },
  { label: "Reports", icon: FlaskConical, desc: "Review and export historical sensor records." },
  { label: "Settings", icon: Settings, desc: "Configure the Google Sheet data source and thresholds." }
];

function parseCsv(text) {
  const rows = text.trim().split(/\r?\n/).map(line => line.split(",").map(v => v.trim().replace(/^"|"$/g, "")));
  if (rows.length < 2) throw new Error("The sheet has no data rows.");
  const headers = rows[0].map(h => h.toLowerCase().replace(/[^a-z0-9]/g, ""));
  const findCol = (names) => headers.findIndex(h => names.some(n => h.includes(n)));
  const timeCol = findCol(["timestamp", "datetime", "date", "time"]);
  const tdsCol = findCol(["tds", "totaldissolvedsolids"]);
  const ecCol = findCol(["conductivity", "ec", "electricalconductivity"]);
  const tempCol = findCol(["temperature", "temp"]);
  if (tdsCol < 0 && ecCol < 0) throw new Error("Could not find TDS or Conductivity columns. Check the sheet's header row.");
  const result = rows.slice(1).map((row, i) => {
    const numberAt = idx => idx < 0 ? NaN : Number(String(row[idx] ?? "").replace(/[^\d.-]/g, ""));
    const ec = numberAt(ecCol);
    const tds = numberAt(tdsCol);
    const temp = numberAt(tempCol);
    const time = timeCol >= 0 ? (row[timeCol] || `Row ${i + 1}`) : `Row ${i + 1}`;
    return {
      time: time.length > 16 ? time.slice(-8) : time,
      timestamp: time,
      tds: Number.isFinite(tds) ? tds : (Number.isFinite(ec) ? Math.round(ec * 0.64) : null),
      ec: Number.isFinite(ec) ? ec : (Number.isFinite(tds) ? Math.round(tds / 0.64) : null),
      temp: Number.isFinite(temp) ? temp : null
    };
  }).filter(r => r.tds !== null || r.ec !== null || r.temp !== null);
  if (!result.length) throw new Error("No numeric TDS, conductivity, or temperature readings found.");
  return result.slice(-100);
}

function formatTime(value) {
  if (!value) return "â€”";
  const d = new Date(value);
  if (!Number.isNaN(d.getTime()) && String(value).includes("-")) return d.toLocaleString([], { month:"short", day:"2-digit", hour:"2-digit", minute:"2-digit" });
  return value;
}
function mean(arr, key) {
  const values = arr.map(r => r[key]).filter(v => Number.isFinite(v));
  return values.length ? values.reduce((a,b) => a+b,0)/values.length : null;
}
function MetricCard({ icon: Icon, label, value, unit, status, color, detail, tag }) {
  return <article className={`metric-card ${color}`}>
    <div className="metric-icon"><Icon size={22}/></div>
    <div className="metric-content"><div className="metric-label">{label}</div>
      <div className="metric-value">{value ?? "â€”"}<span>{unit}</span></div>
      <div className="metric-foot"><span className="status-dot"/>{status}<span className="metric-detail">{detail}</span></div>
    </div><div className="metric-change">{tag}</div>
  </article>;
}
function PanelHeader({ icon: Icon, title, subtitle, action }) {
  return <div className="panel-header"><div><h3><Icon size={18}/>{title}</h3><p>{subtitle}</p></div>{action}</div>;
}
function App() {
  const [active, setActive] = useState("Dashboard");
  const [readings, setReadings] = useState([]);
  const [liveData, setLiveData] = useState(null);
  const [forecasts, setForecasts] = useState({});
  const [source, setSource] = useState("live");
  const [sheetUrl, setSheetUrl] = useState(SHEET_CSV_URL);
  const [refreshSeconds, setRefreshSeconds] = useState("30");
  const [loading, setLoading] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(new Date());
  const [error, setError] = useState("");
  const [toast, setToast] = useState("");
  const [mobileNav, setMobileNav] = useState(false);
  const [search, setSearch] = useState("");
  const [thresholdTds, setThresholdTds] = useState("500");
  const [thresholdEc, setThresholdEc] = useState("800");
  const [thresholdTemp, setThresholdTemp] = useState("35");
  const [chatOpen, setChatOpen] = useState(false);
  const [chatInput, setChatInput] = useState("");
  const [chatBusy, setChatBusy] = useState(false);
  const [chatMessages, setChatMessages] = useState([{ role: "assistant", content: "Hi! Iâ€™m AquaSense Assistant. Ask me about TDS, conductivity, temperature, or the readings currently shown on this dashboard. I can explain indicators, but I canâ€™t certify that water is safe to drink." }]);

  const latest = readings[readings.length - 1] || {};
  const prev = readings[readings.length - 2] || latest;
  const tdsValue = latest.tds;
  const ecValue = latest.ec;
  const tempValue = latest.temp;
  const tdsLimit = Number(thresholdTds) || 500;
  const ecLimit = Number(thresholdEc) || 800;
  const tempLimit = Number(thresholdTemp) || 35;
  const tdsStatus = tdsValue == null ? "No data" : tdsValue <= tdsLimit ? "Within demo limit" : "Above demo limit";
  const ecStatus = ecValue == null ? "No data" : ecValue <= ecLimit ? "Within demo limit" : "Above demo limit";
  const tempStatus = tempValue == null ? "No data" : tempValue <= tempLimit ? "Within demo limit" : "Above demo limit";
  const estimatedSalinity = ecValue == null ? null : +(ecValue * 0.00064).toFixed(2);
  const averageTds = mean(readings, "tds");
  const averageEc = mean(readings, "ec");
  const averageTemp = mean(readings, "temp");
  const overall = [tdsValue != null && tdsValue > tdsLimit, ecValue != null && ecValue > ecLimit, tempValue != null && tempValue > tempLimit].some(Boolean) ? "Review readings" : "Within demo limits";
  const sourceLabel = source === "live" ? "Adafruit IO Live" : "Simulated data";

  const notify = (message) => {
    setToast(message);
    setTimeout(() => setToast(""), 3000);
  };
  const fetchLive = async () => {
    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_BASE_URL}/api/live`);
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(detail.detail || `API returned HTTP ${response.status}`);
      }

      const data = await response.json();

      const parsed = (data.readings || []).map((item) => ({
        timestamp: item.timestamp,
        time: new Date(item.timestamp).toLocaleTimeString("en-GB", {
          timeZone: "Africa/Windhoek",
          hour: "2-digit",
          minute: "2-digit"
        }),
        tds: item.tds,
        ec: null,
        temp: item.temperature,
        humidity: item.humidity
      }));

      setReadings(parsed);
      setLiveData(data);
      setForecasts(data.forecasts || {});
      setSource("live");
      setLastUpdated(new Date(data.updated_at_utc));
      setError("");
    } catch (e) {
      setError(`Live API connection failed: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchLive();
    const seconds = Math.max(15, Number(refreshSeconds) || 30);
    const timer = setInterval(fetchLive, seconds * 1000);
    return () => clearInterval(timer);
  }, [refreshSeconds]);

  const exportCsv = () => {
    const csv = ["Timestamp,TDS (ppm),Conductivity (uS/cm),Temperature (C)", ...readings.map(r => `"${r.timestamp || r.time}",${r.tds ?? ""},${r.ec ?? ""},${r.temp ?? ""}`)].join("\n");
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8;" }));
    const link = document.createElement("a"); link.href = url; link.download = "water-quality-readings.csv"; link.click(); URL.revokeObjectURL(url);
    notify("CSV report downloaded");
  };
  const switchToDemo = () => { setSource("demo"); setReadings(DEMO_READINGS); setLiveData(null); setForecasts({}); setError(""); notify("Switched to sample data"); };
  const trendData = useMemo(() => readings.slice(-24), [readings]);
  const filtered = useMemo(() => readings.filter(r => `${r.timestamp || r.time} ${r.tds} ${r.ec} ${r.temp}`.toLowerCase().includes(search.toLowerCase())), [readings, search]);
  const title = active === "Dashboard" ? "Water Quality Overview" : active;
  const navTo = (label) => { setActive(label); setMobileNav(false); };
  const sendChat = (question = chatInput) => {
    const message = question.trim();
    if (!message || chatBusy) return;
    setChatInput("");
    setChatMessages(prev => [...prev, { role: "user", content: message }]);
    const q = message.toLowerCase();
    const tdsText = tdsValue == null ? "not available" : `${Math.round(tdsValue)} ppm`;
    const ecText = ecValue == null ? "not available" : `${Math.round(ecValue)} ÂµS/cm`;
    const tempText = tempValue == null ? "not available" : `${Number(tempValue).toFixed(1)} Â°C`;
    let answer;
    if (/\b(hi|hello|hey|good morning|good afternoon)\b/.test(q)) {
      answer = "Hello! Iâ€™m the AquaSense chatbot preview. You can ask me about TDS, conductivity, temperature, or the readings displayed on this dashboard. This version uses prepared answers only; AI is not connected yet.";
    } else if (q.includes("tds") || q.includes("dissolved solids")) {
      answer = `TDS means total dissolved solidsâ€”the amount of dissolved substances in water, commonly displayed in ppm or mg/L. The latest dashboard TDS is ${tdsText}. TDS alone does not identify which substances are present or prove that water is safe to drink.`;
    } else if (q.includes("conductivity") || /\bec\b/.test(q)) {
      answer = `Electrical conductivity (EC) indicates how easily water conducts electricity, which is influenced by dissolved ions. The latest dashboard conductivity is ${ecText}. EC is an indicator, not a complete water-quality test.`;
    } else if (q.includes("temperature") || q.includes("hot") || q.includes("cold")) {
      answer = `The latest water temperature shown on the dashboard is ${tempText}. Temperature can affect measurements and may vary over time. This sensor reading does not determine drinking-water safety.`;
    } else if (q.includes("safe") || q.includes("drink") || q.includes("potable")) {
      answer = `The dashboard alone cannot confirm whether this water is safe to drink. Current readings: TDS ${tdsText}, conductivity ${ecText}, and temperature ${tempText}. These measurements do not detect pathogens or many specific contaminants, such as arsenic, fluoride, or nitrate. Arrange an appropriate laboratory test against local drinking-water standards before deciding whether it is potable.`;
    } else if (q.includes("latest") || q.includes("current") || q.includes("readings") || q.includes("values")) {
      answer = `The latest readings shown are: TDS ${tdsText}, conductivity ${ecText}, and temperature ${tempText}. Data source: ${sourceLabel}; records loaded: ${readings.length}. ${source === "demo" ? "These are simulated demo readings, not live sensor data." : "These values come from the connected Google Sheet."}`;
    } else if (q.includes("salinity")) {
      answer = `The dashboard may show a rough salinity proxy derived from conductivity. It is an estimate, not a direct salinity measurement or laboratory result. Current conductivity is ${ecText}.`;
    } else if (q.includes("model") || q.includes("prediction") || q.includes("ai")) {
      answer = "The chatbot is currently a UI preview with prepared responses only. No AI model or trained prediction model is connected yet. We can connect the existing local Phi-3 model in a later step.";
    } else if (q.includes("source") || q.includes("google sheet") || q.includes("data")) {
      answer = `The dashboard is currently using ${sourceLabel.toLowerCase()} and has ${readings.length} records loaded. ${source === "demo" ? "Demo values are simulated. To use project data, open Settings and configure the Google Sheets CSV source." : "The displayed data was loaded from Google Sheets."}`;
    } else {
      answer = "Iâ€™m currently a chatbot UI preview and can answer a limited set of common questions using prepared responses. Try asking â€œWhat does TDS mean?â€, â€œExplain conductivityâ€, â€œShow my latest readingsâ€, or â€œCan I drink this water?â€. An AI model will be connected later.";
    }
    setChatMessages(prev => [...prev, { role: "assistant", content: answer }]);
  };

  return <div className="app-shell">
    <aside className={`sidebar ${mobileNav ? "sidebar-open" : ""}`}>
      <div className="brand"><div className="brand-mark"><Droplets size={25} fill="currentColor"/></div><div><div className="brand-name">AquaSense</div><div className="brand-sub">Water quality intelligence</div></div><button className="mobile-close" onClick={() => setMobileNav(false)}>Ã—</button></div>
      <div className="site-selector"><div className="site-avatar"><Waves size={19}/></div><div className="site-text"><span>Monitoring site</span><strong>Borewell 01</strong></div><ChevronDown size={16}/></div>
      <div className="nav-caption">WORKSPACE</div>
      <nav>{NAV.map(({label, icon: Icon}) => <button key={label} className={`nav-item ${active === label ? "active" : ""}`} onClick={() => navTo(label)}><Icon size={18}/><span>{label}</span>{label === "Live readings" && <span className="nav-live-dot"/>}</button>)}</nav>
      <div className="sidebar-bottom"><div className="connection-card"><div className="connection-heading"><span className={`connection-indicator ${source === "live" ? "connected" : ""}`}/>{source === "live" ? "Sheet connected" : "Demo data active"}</div><p>{sourceLabel} Â· one borewell</p><button onClick={() => navTo("Settings")}><Settings size={14}/> Data source settings</button></div>
        <div className="profile"><div className="profile-avatar">A</div><div className="profile-text"><strong>Administrator</strong><span>Project workspace</span></div><ChevronDown size={15}/></div>
      </div>
    </aside>

    <main className="main-content">
      <header className="topbar"><div className="mobile-menu-wrap"><button className="mobile-menu" onClick={() => setMobileNav(true)}><Menu/></button></div><div className="page-heading"><div className="eyebrow"><span className="eyebrow-dot"/> SINGLE BOREWELL MONITORING</div><h1>{title}</h1><p>Sensor measurements, derived indicators, and trend-based insights.</p></div><div className="top-actions"><div className={`live-pill ${source === "live" ? "sheet-pill" : ""}`}><span className="pulse-dot"/>{source === "live" ? "ADAFRUIT IO LIVE" : "DEMO MODE"}</div><button className="icon-button" onClick={() => notify("No new alerts")} aria-label="Notifications"><Bell size={19}/><i/></button><div className="date-block"><span>LAST UPDATED</span><strong>{lastUpdated.toLocaleTimeString([], {hour:"2-digit",minute:"2-digit"})}</strong></div></div></header>

      {active === "Dashboard" && <>
        <section className="section-intro"><div><h2>Live sensor measurements</h2><p>Only TDS, conductivity, and temperature are treated as direct sensor inputs.</p></div><div className="intro-actions"><span className="site-tag"><MapPin size={14}/> Borewell 01</span><button className="button-secondary" onClick={() => source === "live" ? fetchLive() : (setLastUpdated(new Date()), notify("Demo dashboard refreshed"))}><RefreshCw size={15}/>{loading ? "Loadingâ€¦" : "Refresh"}</button><button className="button-primary" onClick={exportCsv}><Download size={15}/> Export CSV</button></div></section>
        <section className="metrics-grid">
          <MetricCard icon={Droplets} label="Total dissolved solids (TDS)" value={tdsValue == null ? "â€”" : Math.round(tdsValue)} unit="ppm" status={tdsStatus} detail={`Demo threshold ${tdsLimit} ppm`} color="blue" tag="SENSOR"/>
          <MetricCard icon={Zap} label="Electrical conductivity" value={ecValue == null ? "â€”" : Math.round(ecValue)} unit="ÂµS/cm" status={ecStatus} detail={`Demo threshold ${ecLimit}`} color="violet" tag="SENSOR"/>
          <MetricCard icon={Thermometer} label="Water temperature" value={tempValue == null ? "â€”" : Number(tempValue).toFixed(1)} unit="Â°C" status={tempStatus} detail={`Demo threshold ${tempLimit} Â°C`} color="orange" tag="SENSOR"/>
          <article className="metric-card status-card"><div className="metric-icon"><ShieldCheck size={23}/></div><div className="metric-content"><div className="metric-label">Derived status</div><div className="metric-value status-value">{overall}</div><div className="metric-foot"><span className="status-dot"/>{sourceLabel}<span className="metric-detail">Threshold check only</span></div></div><div className="status-seal">{overall === "Within demo limits" ? <CheckCircle2 size={22}/> : <AlertTriangle size={22}/>}</div></article>
        </section>
        <section className="primary-grid">
          <article className="panel trend-panel"><PanelHeader icon={Activity} title="Sensor trends" subtitle="Recent readings Â· values from current data source" action={<button className="select-control" onClick={() => navTo("Live readings")}>View details <ArrowUpRight size={14}/></button>}/><div className="chart-legend"><span><i className="legend-blue"/> TDS (ppm)</span><span><i className="legend-orange"/> Temperature (Â°C)</span><span><i className="legend-violet"/> Conductivity</span></div><div className="chart-area"><ResponsiveContainer width="100%" height="100%"><LineChart data={trendData} margin={{top:10,right:12,left:-15,bottom:0}}><CartesianGrid stroke="#e9eef5" strokeDasharray="3 4" vertical={false}/><XAxis dataKey="time" tick={{fontSize:11,fill:"#8793a8"}} axisLine={false} tickLine={false} minTickGap={24}/><YAxis yAxisId="left" tick={{fontSize:11,fill:"#8793a8"}} axisLine={false} tickLine={false}/><YAxis yAxisId="right" orientation="right" hide/><Tooltip contentStyle={{border:"1px solid #e5eaf2",borderRadius:12,fontSize:12}}/><Line yAxisId="left" type="monotone" dataKey="tds" name="TDS (ppm)" stroke="#2675ef" strokeWidth={2.5} dot={false} connectNulls/><Line yAxisId="right" type="monotone" dataKey="temp" name="Temperature (Â°C)" stroke="#f59e42" strokeWidth={2.5} dot={false} connectNulls/><Line yAxisId="right" type="monotone" dataKey="ec" name="Conductivity (ÂµS/cm)" stroke="#8a68e8" strokeWidth={2.5} dot={false} connectNulls/></LineChart></ResponsiveContainer></div><div className="chart-foot"><span><span className="foot-dot blue"/> Historical sensor values</span><span>Mixed units shown for trend comparison</span></div></article>
          <article className="panel device-panel"><PanelHeader icon={Radio} title="Sensor system" subtitle="ESP32 Â· Borewell 01" action={<span className={`device-badge ${source === "live" ? "good" : "demo"}`}><span/>{source === "live" ? "Sheet feed" : "Demo"}</span>}/><div className="device-illustration"><div className="device-orbit orbit-one"/><div className="device-orbit orbit-two"/><div className="device-chip"><Radio size={29}/><span>ESP32</span></div><div className="sensor-node node-one"><Droplets size={15}/></div><div className="sensor-node node-two"><Thermometer size={15}/></div><div className="sensor-node node-three"><Zap size={15}/></div></div><div className="device-info-row"><span>Device ID</span><strong>ESP32-BW-001</strong></div><div className="device-info-row"><span>Data source</span><strong>{sourceLabel}</strong></div><div className="device-info-row"><span>Records loaded</span><strong>{readings.length}</strong></div><button className="device-link" onClick={() => navTo("Settings")}>Configure data source <ArrowUpRight size={15}/></button></article>
        </section>
        <section className="secondary-grid">
          <article className="panel prediction-panel"><PanelHeader icon={BrainCircuit} title="Derived indicators & insights" subtitle="Calculated from available sensor data" action={<span className="ai-badge">ESTIMATES</span>}/><div className="prediction-highlight"><div className="prediction-icon"><Gauge size={20}/></div><div><span className="prediction-label">Conductivity-based estimate</span><strong>{estimatedSalinity == null ? "Not enough conductivity data" : `Approx. salinity proxy: ${estimatedSalinity} ppt`}</strong><p>Rough conversion only. Actual salinity depends on ionic composition and sensor calibration.</p></div></div><div className="insight-list"><div><span className="insight-icon green"><CheckCircle2 size={16}/></span><span>Average TDS in loaded data: {averageTds == null ? "â€”" : `${Math.round(averageTds)} ppm`}.</span></div><div><span className="insight-icon blue"><Activity size={16}/></span><span>Average conductivity: {averageEc == null ? "â€”" : `${Math.round(averageEc)} ÂµS/cm`}.</span></div><div><span className="insight-icon amber"><Thermometer size={16}/></span><span>Average temperature: {averageTemp == null ? "â€”" : `${averageTemp.toFixed(1)} Â°C`}.</span></div></div><div className="model-foot"><span className="model-status-dot"/>Derived statistics only Â· no trained ML model connected</div></article>
          <article className="panel quality-panel"><PanelHeader icon={CircleGauge} title="Threshold overview" subtitle="Editable demo thresholds" action={<button className="small-icon-button" onClick={() => navTo("Settings")}><SlidersHorizontal size={16}/></button>}/><div className="threshold-item"><div className="threshold-top"><span>TDS</span><strong>{tdsValue == null ? "â€”" : Math.round(tdsValue)} <small>/ {tdsLimit} ppm</small></strong></div><div className="threshold-track"><div className="threshold-fill blue-fill" style={{width:`${Math.min(100,(tdsValue||0)/tdsLimit*100)}%`}}/></div><div className="threshold-caption"><span>0</span><span>Demo limit</span><span>High</span></div></div><div className="threshold-item"><div className="threshold-top"><span>Conductivity</span><strong>{ecValue == null ? "â€”" : Math.round(ecValue)} <small>/ {ecLimit} ÂµS/cm</small></strong></div><div className="threshold-track"><div className="threshold-fill violet-fill" style={{width:`${Math.min(100,(ecValue||0)/ecLimit*100)}%`}}/></div><div className="threshold-caption"><span>0</span><span>Demo limit</span><span>High</span></div></div><div className="threshold-note"><CircleHelp size={15}/> These are configurable display thresholds, not a drinking-water safety determination.</div></article>
        </section>
        <section className="panel readings-panel"><PanelHeader icon={Clock3} title="Recent sensor readings" subtitle={`Latest records Â· ${sourceLabel}`} action={<button className="text-button" onClick={() => navTo("Live readings")}>View all <ArrowUpRight size={14}/></button>}/><ReadingsTable readings={readings.slice(-5).reverse()} source={source}/><div className="table-bottom"><span>Showing 5 of {readings.length} records</span><span><span className="table-live-dot"/>{source === "live" ? `Sheet refresh every ${refreshSeconds}s` : "Sample data Â· switch to Google Sheets in Settings"}</span></div></section>
      </>}

      {active === "Live readings" && <><section className="section-intro"><div><h2>Sensor readings explorer</h2><p>Search, inspect, and compare readings from the selected data source.</p></div><div className="intro-actions"><button className="button-secondary" onClick={() => source === "live" ? fetchLive() : notify("Showing sample readings")}><RefreshCw size={15}/> Refresh data</button><button className="button-primary" onClick={exportCsv}><Download size={15}/> Export CSV</button></div></section><div className="detail-stat-grid"><DetailStat label="Latest TDS" value={tdsValue == null ? "â€”" : `${Math.round(tdsValue)} ppm`} icon={Droplets}/><DetailStat label="Latest conductivity" value={ecValue == null ? "â€”" : `${Math.round(ecValue)} ÂµS/cm`} icon={Zap}/><DetailStat label="Latest temperature" value={tempValue == null ? "â€”" : `${Number(tempValue).toFixed(1)} Â°C`} icon={Thermometer}/><DetailStat label="Records loaded" value={readings.length} icon={Database}/></div><article className="panel readings-panel page-panel"><PanelHeader icon={History} title="Reading history" subtitle="Use search to filter by time or value" action={<div className="search-box"><Search size={15}/><input value={search} onChange={e=>setSearch(e.target.value)} placeholder="Search readingsâ€¦"/></div>}/><ReadingsTable readings={[...filtered].reverse()} source={source} showTimestamp/><div className="table-bottom"><span>{filtered.length} matching records</span><span>Source: {sourceLabel}</span></div></article></>}

      {active === "Predictions" && <><section className="section-intro"><div><h2>Predictions & derived values</h2><p>Transparent calculations based on available TDS, conductivity, and temperature measurements.</p></div><span className="ai-badge">MODEL NOT CONNECTED</span></section><div className="section-note"><CircleHelp size={17}/><span>These are descriptive estimates, not a trained machine-learning forecast. A reliable predictive model needs historical data, labels, validation, and a defined prediction target.</span></div><div className="detail-stat-grid"><DetailStat label="Average TDS" value={averageTds == null ? "â€”" : `${Math.round(averageTds)} ppm`} icon={Droplets}/><DetailStat label="Average conductivity" value={averageEc == null ? "â€”" : `${Math.round(averageEc)} ÂµS/cm`} icon={Zap}/><DetailStat label="Average temperature" value={averageTemp == null ? "â€”" : `${averageTemp.toFixed(1)} Â°C`} icon={Thermometer}/><DetailStat label="Salinity proxy" value={estimatedSalinity == null ? "â€”" : `${estimatedSalinity} ppt`} icon={Waves}/></div><div className="secondary-grid"><article className="panel prediction-panel"><PanelHeader icon={TrendingUp} title="Recent trend" subtitle="TDS and conductivity over loaded records"/><div className="large-chart"><ResponsiveContainer width="100%" height="100%"><AreaChart data={trendData}><defs><linearGradient id="tdsGradient" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#2675ef" stopOpacity={.22}/><stop offset="95%" stopColor="#2675ef" stopOpacity={0}/></linearGradient></defs><CartesianGrid stroke="#e9eef5" strokeDasharray="3 4" vertical={false}/><XAxis dataKey="time" tick={{fontSize:10,fill:"#8793a8"}} axisLine={false} tickLine={false}/><YAxis tick={{fontSize:10,fill:"#8793a8"}} axisLine={false} tickLine={false}/><Tooltip/><Area type="monotone" dataKey="tds" name="TDS (ppm)" stroke="#2675ef" fill="url(#tdsGradient)" strokeWidth={2.5} connectNulls/></AreaChart></ResponsiveContainer></div><div className="model-foot"><span className="model-status-dot"/> Trend visualization only; it does not predict future values.</div></article><article className="panel quality-panel"><PanelHeader icon={ShieldCheck} title="How derived values work" subtitle="Calculation notes"/><div className="formula-card"><strong>TDS estimate (if missing)</strong><code>TDS â‰ˆ conductivity Ã— 0.64</code><p>Approximate conversion factor only; it varies with water chemistry and device calibration.</p></div><div className="formula-card"><strong>Salinity proxy</strong><code>EC (ÂµS/cm) Ã— 0.00064</code><p>A rough indicator, not a laboratory salinity measurement.</p></div><div className="formula-card"><strong>Status</strong><code>Measured value vs configured limit</code><p>Threshold status is not a complete water-quality assessment.</p></div></article></div></>}

      {active === "Reports" && <><section className="section-intro"><div><h2>Reports & historical data</h2><p>Export the current dataset or review its summary statistics.</p></div><div className="intro-actions"><button className="button-primary" onClick={exportCsv}><Download size={15}/> Download CSV report</button></div></section><div className="detail-stat-grid"><DetailStat label="Records available" value={readings.length} icon={Database}/><DetailStat label="Average TDS" value={averageTds == null ? "â€”" : `${Math.round(averageTds)} ppm`} icon={Droplets}/><DetailStat label="Average conductivity" value={averageEc == null ? "â€”" : `${Math.round(averageEc)} ÂµS/cm`} icon={Zap}/><DetailStat label="Average temperature" value={averageTemp == null ? "â€”" : `${averageTemp.toFixed(1)} Â°C`} icon={Thermometer}/></div><article className="panel report-chart-panel"><PanelHeader icon={Activity} title="Dataset overview" subtitle="TDS readings in the loaded dataset" action={<span className="site-tag">{sourceLabel}</span>}/><div className="large-chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={readings.slice(-30)}><CartesianGrid stroke="#e9eef5" strokeDasharray="3 4" vertical={false}/><XAxis dataKey="time" tick={{fontSize:10,fill:"#8793a8"}} axisLine={false} tickLine={false}/><YAxis tick={{fontSize:10,fill:"#8793a8"}} axisLine={false} tickLine={false}/><Tooltip/><Bar dataKey="tds" name="TDS (ppm)" fill="#4389ed" radius={[4,4,0,0]}/></BarChart></ResponsiveContainer></div></article><article className="panel readings-panel page-panel"><PanelHeader icon={History} title="All records" subtitle="The current dataset, newest first"/><ReadingsTable readings={[...readings].reverse()} source={source} showTimestamp/></article></>}

      {active === "Settings" && <>

  <section className="section-intro"><div><h2>Data source & dashboard settings</h2><p>Connect the published Google Sheet that receives your ESP sensor readings.</p></div><span className={`device-badge ${source === "live" ? "good" : "demo"}`}><span/>{source === "live" ? "Sheet connected" : "Demo mode"}</span></section><div className="settings-grid"><article className="panel settings-panel"><PanelHeader icon={FileSpreadsheet} title="Google Sheets connection" subtitle="Read a published CSV export"/><div className="settings-callout"><FileSpreadsheet size={20}/><div><strong>Conductivity spreadsheet</strong><p>Paste the spreadsheet link. The first row should contain column headings for timestamp, TDS, conductivity/EC, and temperature.</p></div></div><label className="field-label" htmlFor="sheetUrl">Google Sheets URL or CSV export URL</label><input id="sheetUrl" className="text-input" value={sheetUrl} onChange={e=>setSheetUrl(e.target.value)} placeholder="https://docs.google.com/spreadsheets/d/â€¦"/><div className="settings-actions"><button className="button-primary" disabled={loading} onClick={()=>fetchLive()}>{loading ? <RefreshCw size={15} className="spin"/> : <CloudDownload size={15}/>} {loading ? "Connectingâ€¦" : "Connect & load data"}</button><button className="button-secondary" onClick={switchToDemo}>Use sample data</button></div>{error && <div className="error-box"><AlertTriangle size={17}/><span>{error}</span></div>}<div className="helper-box"><strong>Make the sheet accessible</strong><ol><li>Open the Google Sheet.</li><li>For testing, use Share â†’ General access â†’ Anyone with the link â†’ Viewer, or publish it via File â†’ Share â†’ Publish to web.</li><li>Make sure the selected tab is the one containing sensor readings and headings are in row 1.</li><li>Click â€œConnect & load dataâ€.</li></ol><p>Only make the sheet public if it contains no private or sensitive information.</p></div><div className="connection-summary"><span>Current source</span><strong>{sourceLabel}</strong><span>Rows loaded</span><strong>{readings.length}</strong><span>Last successful load</span><strong>{lastUpdated.toLocaleString()}</strong></div></article><article className="panel settings-panel"><PanelHeader icon={SlidersHorizontal} title="Refresh & thresholds" subtitle="Local dashboard preferences"/><label className="field-label" htmlFor="refreshSeconds">Refresh interval (seconds)</label><select id="refreshSeconds" className="text-input" value={refreshSeconds} onChange={e=>setRefreshSeconds(e.target.value)}><option value="15">15 seconds</option><option value="30">30 seconds</option><option value="60">60 seconds</option><option value="300">5 minutes</option></select><p className="field-help">Auto-refresh applies while Google Sheets is the selected source. The sheet may update less frequently depending on how ESP data is written.</p><div className="settings-divider"/><div className="threshold-setting"><label className="field-label" htmlFor="tdsLimit">TDS display threshold (ppm)</label><input id="tdsLimit" type="number" className="text-input" value={thresholdTds} onChange={e=>setThresholdTds(e.target.value)}/></div><div className="threshold-setting"><label className="field-label" htmlFor="ecLimit">Conductivity display threshold (ÂµS/cm)</label><input id="ecLimit" type="number" className="text-input" value={thresholdEc} onChange={e=>setThresholdEc(e.target.value)}/></div><div className="threshold-setting"><label className="field-label" htmlFor="tempLimit">Temperature display threshold (Â°C)</label><input id="tempLimit" type="number" className="text-input" value={thresholdTemp} onChange={e=>setThresholdTemp(e.target.value)}/></div><button className="button-primary full-button" onClick={()=>notify("Settings applied to the current dashboard session")}><Save size={15}/> Apply settings</button><div className="threshold-note"><CircleHelp size={15}/> These thresholds are for dashboard indicators only. Confirm appropriate standards and sensor calibration before operational use.</div></article></div></>}
      <footer className="footer"><span>Water Quality Monitoring System <span className="footer-separator">Â·</span> Single borewell</span><span>{sourceLabel} <span className="footer-separator">Â·</span> TDS, conductivity & temperature sensor inputs</span></footer>
    </main>
    {mobileNav && <button className="mobile-backdrop" aria-label="Close menu" onClick={() => setMobileNav(false)}/>}
    {chatOpen && <section className="chat-panel" aria-label="AquaSense chatbot preview">
      <header className="chat-header"><div className="chat-avatar"><Bot size={20}/></div><div className="chat-heading"><strong>AquaSense Assistant</strong><span><i/> Chatbot preview Â· no AI connected</span></div><button className="chat-close" onClick={() => setChatOpen(false)} aria-label="Minimize chat"><Minimize2 size={17}/></button></header>
      <div className="chat-context"><span className="chat-context-dot"/> Using {sourceLabel.toLowerCase()} Â· {readings.length} records loaded</div>
      <div className="chat-messages" aria-live="polite">{chatMessages.map((m, i) => <div className={`chat-message-row ${m.role}`} key={i}><div className="chat-message-icon">{m.role === "assistant" ? <Bot size={15}/> : "You"}</div><div className={`chat-bubble ${m.error ? "chat-error" : ""}`}>{m.content}</div></div>)}{chatBusy && <div className="chat-message-row assistant"><div className="chat-message-icon"><Bot size={15}/></div><div className="chat-bubble chat-typing"><span/><span/><span/> Thinkingâ€¦</div></div>}</div>
      {chatMessages.length === 1 && <div className="chat-suggestions"><span>Try asking</span><button onClick={() => sendChat("What does TDS mean?")}>What does TDS mean?</button><button onClick={() => sendChat("Can I safely drink this water based on these readings?")}>Is this water safe to drink?</button><button onClick={() => sendChat("Explain the latest readings in simple terms.")}>Explain my latest readings</button></div>}
      <form className="chat-composer" onSubmit={e => { e.preventDefault(); sendChat(); }}><input value={chatInput} onChange={e => setChatInput(e.target.value)} placeholder="Ask about your water qualityâ€¦" aria-label="Your question" disabled={chatBusy}/><button type="submit" disabled={chatBusy || !chatInput.trim()} aria-label="Send message"><Send size={17}/></button></form>
      <p className="chat-disclaimer">Informational guidance only. Sensor readings do not confirm drinking-water safety.</p>
    </section>}
    {!chatOpen && <button className="chat-launcher" onClick={() => setChatOpen(true)} aria-label="Open AquaSense chatbot"><MessageCircle size={22}/><span>Ask AquaSense</span><i/></button>}
    {toast && <div className="toast"><CheckCircle2 size={17}/>{toast}</div>}
  </div>;
}

function DetailStat({label,value,icon:Icon}) {
  return <article className="detail-stat"><div className="detail-stat-icon"><Icon size={19}/></div><div><span>{label}</span><strong>{value}</strong></div></article>;
}
function ReadingsTable({readings,source,showTimestamp=false}) {
  return <div className="table-wrap"><table><thead><tr><th>Timestamp</th><th>TDS (ppm)</th><th>Conductivity (ÂµS/cm)</th><th>Temperature (Â°C)</th><th>Source</th></tr></thead><tbody>{readings.map((r,i)=><tr key={`${r.timestamp||r.time}-${i}`}><td><span className="table-time">{showTimestamp ? formatTime(r.timestamp || r.time) : r.time}</span>{i===0&&<small>Latest in view</small>}</td><td><strong>{r.tds == null ? "â€”" : Number(r.tds).toFixed(0)}</strong></td><td>{r.ec == null ? "â€”" : Number(r.ec).toFixed(0)}</td><td>{r.temp == null ? "â€”" : Number(r.temp).toFixed(1)}</td><td><span className="source-tag"><Radio size={12}/>{source === "live" ? "Google Sheet" : "Demo"}</span></td></tr>)}</tbody></table>{readings.length===0&&<div className="empty-state">No readings match your search.</div>}</div>;
}
export default App;

