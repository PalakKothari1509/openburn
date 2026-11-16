const API_BASE = window.location.origin;

// Tab switching
let familiesLoaded = false;
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll("section.card").forEach((s) => s.classList.add("hidden"));
    btn.classList.add("active");
    document.getElementById(`tab-${btn.dataset.tab}`).classList.remove("hidden");
    hideError();
    if (btn.dataset.tab === "families" && !familiesLoaded) {
      document.getElementById("btn-load-families").click();
    }
  });
});

// Toggle a submit button into/out of a "working" state so slow requests
// (and accidental double-submits) give visible feedback.
function setBusy(btn, busyText, idleText) {
  btn.disabled = true;
  btn.dataset.idleText = idleText;
  btn.textContent = busyText;
}
function clearBusy(btn) {
  btn.disabled = false;
  btn.textContent = btn.dataset.idleText;
}

function setConfidencePill(el, confidence) {
  el.textContent = confidence || "-";
  el.classList.remove("confidence-high", "confidence-low");
  if (confidence === "high") el.classList.add("confidence-high");
  else if (confidence === "low") el.classList.add("confidence-low");
}

function renderEmissionsBreakdown(container, emissions) {
  const max = Math.max(...Object.values(emissions).map((v) => Number(v) || 0));
  container.innerHTML = Object.entries(emissions)
    .map(([k, v]) => {
      const pct = max > 0 ? Math.max((Number(v) / max) * 100, 2) : 0;
      return `<div class="emission-row">
        <span class="emission-name">${k.toUpperCase()}</span>
        <span class="emission-bar-track"><span class="emission-bar-fill" style="width:${pct}%"></span></span>
        <span class="emission-value">${formatNum(v)} kg</span>
      </div>`;
    })
    .join("");
}

function showError(msg) {
  const box = document.getElementById("error-box");
  box.textContent = msg;
  box.classList.remove("hidden");
}

function hideError() {
  document.getElementById("error-box").classList.add("hidden");
}

async function api(endpoint, options = {}) {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "Unknown error");
  return data;
}

function formatNum(n) {
  return typeof n === "number" ? n.toLocaleString() : n;
}

// Check model status on load
async function checkStatus() {
  try {
    const data = await api("/health");
    const badge = document.getElementById("model-status");
    if (data.model_trained) {
      const fromNames = Array.isArray(data.trained_from) ? data.trained_from.join(", ") : data.trained_from;
      const trainedNote = data.trained_at
        ? ` — trained ${new Date(data.trained_at).toLocaleString()} from ${fromNames}`
        : "";
      badge.textContent = `Model trained — ${data.families_available.length} aircraft types${trainedNote}`;
      badge.className = "label label-primary";
    } else {
      badge.textContent = "Model not trained — go to the \"Train the Model\" tab";
      badge.className = "label label-warning";
    }
    populateAircraftSelect(data.model_trained ? data.families_available : []);
  } catch {
    document.getElementById("model-status").textContent = "API offline — start the server with: python api/app.py";
    document.getElementById("model-status").className = "label label-danger";
    populateAircraftSelect([]);
  }
}

// Keep the Route tab's aircraft dropdown in sync with whatever the model
// is actually trained on, instead of a hardcoded list that can drift.
function populateAircraftSelect(families) {
  const select = document.getElementById("aircraft-select");
  const form = document.getElementById("form-route");
  const wanted = form.dataset.restoreAircraft || select.value;
  if (families.length === 0) {
    select.innerHTML = `<option value="">Train a model first...</option>`;
    return;
  }
  select.innerHTML =
    `<option value="">Select...</option>` +
    families.map((f) => `<option value="${f}">${f}</option>`).join("");
  if (families.includes(wanted)) {
    select.value = wanted;
    delete form.dataset.restoreAircraft;
  }
}

// Populate the shared airport-code datalist used by the Departure/Arrival
// inputs, so typos get caught by autocomplete instead of a failed request.
async function loadAirportCodes() {
  try {
    const data = await api("/airports");
    document.getElementById("airport-codes").innerHTML = data.airports
      .map((code) => `<option value="${code}"></option>`)
      .join("");
  } catch {
    // Non-critical — inputs still work without autocomplete.
  }
}

// Remember the last-used route inputs so returning visitors don't have to
// re-type them. Browser storage only; never required for the app to work.
function rememberRouteInputs(body) {
  try {
    localStorage.setItem("openburn.lastRoute", JSON.stringify(body));
  } catch {}
}
function restoreRouteInputs() {
  try {
    const saved = JSON.parse(localStorage.getItem("openburn.lastRoute") || "null");
    if (!saved) return;
    const form = document.getElementById("form-route");
    if (saved.departure) form.querySelector('[name="departure"]').value = saved.departure;
    if (saved.arrival) form.querySelector('[name="arrival"]').value = saved.arrival;
    if (saved.passengers) form.querySelector('[name="passengers"]').value = saved.passengers;
    if (saved.aircraft_type) form.dataset.restoreAircraft = saved.aircraft_type;
  } catch {}
}

checkStatus();
loadAirportCodes();
restoreRouteInputs();

// ============================================================
// Route-based estimate (primary)
// ============================================================
document.getElementById("form-route").addEventListener("submit", async (e) => {
  e.preventDefault();
  hideError();
  const fd = new FormData(e.target);
  const body = {
    departure: fd.get("departure").toUpperCase().trim(),
    arrival: fd.get("arrival").toUpperCase().trim(),
    aircraft_type: fd.get("aircraft_type"),
  };
  if (fd.get("passengers")) body.passengers = fd.get("passengers");

  const btn = e.target.querySelector("button[type=submit]");
  setBusy(btn, "Calculating...", "Calculate Fuel Burn");
  try {
    const data = await api("/estimate/route", { method: "POST", body: JSON.stringify(body) });

    document.getElementById("route-header").textContent =
      `${data.route.departure} → ${data.route.arrival} · ${data.aircraft_family}`;

    document.getElementById("route-fuel").textContent = formatNum(data.fuel_burn_kg);
    document.getElementById("route-co2").textContent = formatNum(data.emissions.co2);
    document.getElementById("route-distance").textContent = formatNum(data.route.distance_km);
    document.getElementById("route-duration").textContent = formatNum(Math.round(data.route.estimated_duration_min));
    setConfidencePill(document.getElementById("route-confidence"), data.confidence);
    document.getElementById("route-rmse").textContent = data.model_rmse_kg ? formatNum(data.model_rmse_kg) : "N/A";

    const routeWarning = document.getElementById("route-warning");
    if (data.warning) {
      routeWarning.textContent = data.warning;
      routeWarning.hidden = false;
    } else {
      routeWarning.hidden = true;
    }

    renderEmissionsBreakdown(document.getElementById("route-emissions-breakdown"), data.emissions);

    document.getElementById("result-route").hidden = false;
    rememberRouteInputs(body);
  } catch (err) {
    showError(err.message);
  } finally {
    clearBusy(btn);
  }
});

// Auto-uppercase airport inputs
document.querySelectorAll('input[name="departure"], input[name="arrival"]').forEach((input) => {
  input.addEventListener("input", () => { input.value = input.value.toUpperCase(); });
});

// ============================================================
// Advanced estimate
// ============================================================
document.getElementById("form-estimate").addEventListener("submit", async (e) => {
  e.preventDefault();
  hideError();
  const fd = new FormData(e.target);
  const body = { aircraft_type: fd.get("aircraft_type").toUpperCase().trim(), distance_km: fd.get("distance_km") };
  if (fd.get("flight_duration_min")) body.flight_duration_min = fd.get("flight_duration_min");
  if (fd.get("cruise_altitude_ft")) body.cruise_altitude_ft = fd.get("cruise_altitude_ft");

  const btn = e.target.querySelector("button[type=submit]");
  setBusy(btn, "Estimating...", "Estimate Fuel Burn");
  try {
    const data = await api("/estimate", { method: "POST", body: JSON.stringify(body) });
    document.getElementById("est-fuel").textContent = formatNum(data.fuel_burn_kg);
    document.getElementById("est-co2").textContent = formatNum(data.emissions.co2);
    document.getElementById("est-method").textContent = data.method;
    setConfidencePill(document.getElementById("est-confidence"), data.confidence);

    const estWarning = document.getElementById("est-warning");
    if (data.warning) {
      estWarning.textContent = data.warning;
      estWarning.hidden = false;
    } else {
      estWarning.hidden = true;
    }

    renderEmissionsBreakdown(document.getElementById("emissions-breakdown"), data.emissions);

    document.getElementById("result-estimate").hidden = false;
  } catch (err) {
    showError(err.message);
  } finally {
    clearBusy(btn);
  }
});

// ============================================================
// Batch estimate
// ============================================================
document.getElementById("form-batch").addEventListener("submit", async (e) => {
  e.preventDefault();
  hideError();
  const fd = new FormData(e.target);
  const btn = e.target.querySelector("button[type=submit]");
  setBusy(btn, "Estimating...", "Estimate Batch");
  try {
    const flights = JSON.parse(fd.get("flights"));
    const data = await api("/estimate/batch", { method: "POST", body: JSON.stringify({ flights }) });

    const table = document.getElementById("batch-table");
    const thead = table.querySelector("thead tr");
    const tbody = table.querySelector("tbody");

    if (data.length === 0) { showError("No results"); return; }

    const cols = ["aircraft_family", "distance_km", "fuel_burn_estimated_kg", "co2_kg", "estimation_method", "estimation_confidence", "warning"];
    const available = cols.filter((c) => c in data[0]);

    thead.innerHTML = available.map((c) => `<th>${c}</th>`).join("");
    tbody.innerHTML = data
      .map((row) => `<tr>${available.map((c) => `<td>${formatNum(row[c]) ?? ""}</td>`).join("")}</tr>`)
      .join("");

    document.getElementById("result-batch").hidden = false;
  } catch (err) {
    showError(err.message);
  } finally {
    clearBusy(btn);
  }
});

document.getElementById("btn-load-example").addEventListener("click", () => {
  const textarea = document.querySelector('#form-batch [name="flights"]');
  textarea.value = JSON.stringify(
    [
      { aircraft_type: "A320", distance_km: 1500 },
      { aircraft_type: "E190", distance_km: 800 },
    ],
    null,
    2
  );
});

// ============================================================
// Train model
// ============================================================
function renderTrainOutput(data) {
  document.getElementById("train-total").textContent = formatNum(data.total_flights);
  document.getElementById("train-families").textContent = (data.families_trained || []).join(", ") || "-";
  const overall = data.holdout_validation && data.holdout_validation.overall;
  document.getElementById("train-mape").textContent = overall ? `${overall.mape_pct}%` : "N/A";
  document.getElementById("train-r2").textContent = overall ? overall.r_squared : "N/A";
  document.getElementById("train-output").textContent = JSON.stringify(data, null, 2);
  document.getElementById("result-train").hidden = false;
}

document.getElementById("form-train").addEventListener("submit", async (e) => {
  e.preventDefault();
  hideError();
  const fd = new FormData(e.target);
  const btn = e.target.querySelector("button");
  setBusy(btn, "Training...", "Train from Path");
  try {
    const csvPaths = fd.get("csv_path").split(",").map((p) => p.trim()).filter(Boolean);
    const data = await api("/train", {
      method: "POST",
      body: JSON.stringify({ csv_paths: csvPaths, replace: fd.get("replace") === "on" }),
    });
    renderTrainOutput(data);
    checkStatus();
    familiesLoaded = false;
  } catch (err) {
    showError(err.message);
  } finally {
    clearBusy(btn);
  }
});

// ============================================================
// Train model (upload a CSV file directly)
// ============================================================
document.getElementById("form-train-upload").addEventListener("submit", async (e) => {
  e.preventDefault();
  hideError();
  const fd = new FormData(e.target);
  const btn = e.target.querySelector("button");
  setBusy(btn, "Uploading & Training...", "Upload & Train");
  try {
    // Don't use the api() helper here — it forces a JSON content-type,
    // which breaks multipart/form-data file uploads. Let fetch set its own.
    const res = await fetch(`${API_BASE}/train/upload`, { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Unknown error");
    renderTrainOutput(data);
    checkStatus();
    familiesLoaded = false;
  } catch (err) {
    showError(err.message);
  } finally {
    clearBusy(btn);
  }
});

// ============================================================
// Load families
// ============================================================
document.getElementById("btn-load-families").addEventListener("click", async () => {
  hideError();
  const btn = document.getElementById("btn-load-families");
  setBusy(btn, "Loading...", "Refresh");
  try {
    const [families, summary] = await Promise.all([api("/families"), api("/model/summary")]);

    // Training summary (persists across restarts, unlike the one-time /train response)
    document.getElementById("sum-trained-at").textContent = summary.trained_at
      ? new Date(summary.trained_at).toLocaleString()
      : "-";
    const summaryFromNames = Array.isArray(summary.trained_from) ? summary.trained_from.join(", ") : summary.trained_from;
    document.getElementById("sum-trained-from").textContent = summaryFromNames || "-";
    document.getElementById("sum-total").textContent = formatNum(summary.total_flights);
    const overall = summary.holdout_validation && summary.holdout_validation.overall;
    document.getElementById("sum-mape").textContent = overall ? `${overall.mape_pct}%` : "N/A";
    document.getElementById("sum-r2").textContent = overall ? overall.r_squared : "N/A";
    document.getElementById("sum-rmse").textContent = overall ? `${formatNum(overall.rmse_kg)} kg` : "N/A";
    document.getElementById("result-summary").hidden = false;

    // Per-family table rows, with an inline bar (relative to the worst
    // family's RMSE) so accuracy differences are visible at a glance,
    // plus the top 3 most important factors for each aircraft.
    const maxRmse = Math.max(...Object.values(families).map((info) => info.rmse_kg || 0));
    document.getElementById("family-grid").innerHTML = Object.entries(families)
      .map(([fam, info]) => {
        const topFactors = Object.entries(info.feature_importance || {})
          .sort((a, b) => b[1] - a[1])
          .slice(0, 3)
          .map(([name]) => name.replace(/_/g, " "))
          .join(", ");
        const pct = maxRmse > 0 ? Math.max((info.rmse_kg / maxRmse) * 100, 2) : 0;
        const rmseCell = `<span class="bar-cell">
          <span class="bar-track"><span class="bar-fill" style="width:${pct}%"></span></span>
          <span class="bar-number">${formatNum(info.rmse_kg)}</span>
        </span>`;
        return `<tr><td><strong>${fam}</strong></td><td>${rmseCell}</td><td>${formatNum(info.n_training_flights)}</td><td>${topFactors}</td></tr>`;
      })
      .join("");
    document.getElementById("result-families").hidden = false;
    familiesLoaded = true;
  } catch (err) {
    showError(err.message);
  } finally {
    clearBusy(btn);
  }
});
