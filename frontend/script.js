const API_BASE = window.location.origin;

// Tab switching
let familiesLoaded = false;
let modelTrained = false;
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll("section.card").forEach((s) => s.classList.add("hidden"));
    btn.classList.add("active");
    document.getElementById(`tab-${btn.dataset.tab}`).classList.remove("hidden");
    hideError();
    if (btn.dataset.tab === "families" && !familiesLoaded) {
      if (modelTrained) {
        document.getElementById("btn-load-families").click();
      } else {
        document.getElementById("families-untrained").hidden = false;
      }
    }
  });
});

// Showing "X training files" instead of a raw, often long/ugly filename
// (or filenames) when there's more than one; the full list is still
// available on hover for anyone who wants it.
function formatTrainedFrom(names) {
  const arr = Array.isArray(names) ? names : names ? [names] : [];
  if (arr.length === 0) return { text: "-", title: "" };
  if (arr.length === 1) return { text: arr[0], title: "" };
  return { text: `${arr.length} training files`, title: arr.join(", ") };
}

// Shows how many rows each training file actually contributed after
// cleaning, so a file that's silently mostly dropped (bad encoding, wrong
// columns, a truncated upload, ...) is visible instead of just looking
// like a smaller-than-expected "Flights Trained On" total.
function renderSourceBreakdown(breakdown) {
  const container = document.getElementById("sum-source-breakdown");
  container.innerHTML = "";
  if (!breakdown || breakdown.length < 2) return; // not useful for a single file

  const heading = document.createElement("p");
  heading.className = "text-muted small";
  heading.style.marginTop = "0.5rem";
  heading.style.marginBottom = "0.2rem";
  const strong = document.createElement("strong");
  strong.textContent = "Per-file breakdown:";
  heading.appendChild(strong);
  container.appendChild(heading);

  const list = document.createElement("ul");
  list.className = "source-breakdown-list";
  breakdown.forEach((b) => {
    const li = document.createElement("li");
    const pct = b.raw_rows > 0 ? Math.round((b.rows_used / b.raw_rows) * 100) : 0;
    li.textContent = `${b.file} — ${formatNum(b.rows_used)} of ${formatNum(b.raw_rows)} rows used (${pct}%)`;
    list.appendChild(li);
  });
  container.appendChild(list);
}

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
  el.classList.remove("confidence-pill", "confidence-high", "confidence-low");
  if (confidence === "high" || confidence === "low") {
    el.classList.add("confidence-pill", `confidence-${confidence}`);
  }
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
    modelTrained = data.model_trained;
    if (data.model_trained) {
      const from = formatTrainedFrom(data.trained_from);
      const trainedNote = data.trained_at
        ? ` — trained ${new Date(data.trained_at).toLocaleString()} from ${from.text}`
        : "";
      badge.textContent = `Model trained — ${data.families_available.length} aircraft types${trainedNote}`;
      badge.title = from.title;
      badge.className = "label label-primary";
      document.getElementById("families-untrained").hidden = true;
    } else {
      badge.textContent = "Model not trained — go to the \"Train the Model\" tab";
      badge.className = "label label-warning";
    }
    populateAircraftSelect(data.model_trained ? data.families_available : []);
  } catch {
    modelTrained = false;
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
  document.getElementById("train-r2").textContent = overall ? `${(overall.r_squared * 100).toFixed(1)}%` : "N/A";
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
// Model accuracy
// ============================================================

// Plain-language labels for the raw feature names the model trains on.
const FACTOR_LABELS = {
  distance_km: "Flight distance",
  flight_duration_min: "Flight duration",
  airborne_duration_min: "Time in the air",
  taxi_time_min: "Taxi time",
  cruise_altitude_ft: "Cruising altitude",
  average_speed_kts: "Average speed",
  payload_kg: "Passenger & cargo weight",
  headwind_kts: "Wind",
  temperature_c: "Outside temperature",
  takeoff_weight_kg: "Takeoff weight",
  passengers: "Passenger count",
  cargo_kg: "Cargo weight",
};
function friendlyFactor(name) {
  return FACTOR_LABELS[name] || name.replace(/_/g, " ");
}

// Translate MAPE (average % error on unseen flights) into a plain rating,
// since "MAPE" on its own means nothing to most users.
function accuracyRating(mapePct) {
  if (mapePct == null || isNaN(mapePct)) return { label: "Not yet measured", cls: "unknown" };
  if (mapePct <= 3) return { label: "Excellent", cls: "excellent" };
  if (mapePct <= 6) return { label: "Good", cls: "good" };
  if (mapePct <= 10) return { label: "Fair", cls: "fair" };
  return { label: "Needs more data", cls: "poor" };
}

// Worst-accuracy-first, so the aircraft that actually need attention
// surface before the ones that are already doing fine.
const RATING_SEVERITY = { poor: 0, unknown: 1, fair: 2, good: 3, excellent: 4 };

// ------------------------------------------------------------
// Predicted vs actual scatter chart (holdout flights)
// ------------------------------------------------------------
let scatterPoints = [];

function renderScatterChart(points) {
  scatterPoints = points || [];
  const noData = document.getElementById("scatter-no-data");
  const wrap = document.getElementById("scatter-chart-wrap");

  if (scatterPoints.length === 0) {
    noData.hidden = false;
    wrap.hidden = true;
    return;
  }
  noData.hidden = true;
  wrap.hidden = false;

  const families = [...new Set(scatterPoints.map((p) => p.family))].sort();
  const select = document.getElementById("scatter-family-filter");
  const prevValue = select.value;
  select.innerHTML =
    `<option value="">All aircraft</option>` +
    families.map((f) => `<option value="${f}">${f}</option>`).join("");
  select.value = families.includes(prevValue) ? prevValue : "";

  drawScatter(select.value);
}

function drawScatter(familyFilter) {
  const container = document.getElementById("scatter-chart");
  const wrap = document.getElementById("scatter-chart-wrap");
  const points = familyFilter ? scatterPoints.filter((p) => p.family === familyFilter) : scatterPoints;

  if (points.length === 0) {
    container.innerHTML = "";
    return;
  }

  const W = wrap.clientWidth || 600;
  const H = 360;
  const PAD = { top: 16, right: 16, bottom: 46, left: 60 };
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;

  const allVals = points.flatMap((p) => [p.actual_kg, p.predicted_kg]);
  const rawMin = Math.min(...allVals);
  const rawMax = Math.max(...allVals);
  const span = rawMax - rawMin || Math.max(rawMax, 1);
  const min = Math.max(0, rawMin - span * 0.05);
  const max = rawMax + span * 0.05;

  const xPx = (v) => PAD.left + ((v - min) / (max - min)) * plotW;
  const yPx = (v) => PAD.top + plotH - ((v - min) / (max - min)) * plotH;

  const tickCount = 5;
  const tickVals = Array.from({ length: tickCount + 1 }, (_, i) => min + (i * (max - min)) / tickCount);
  const fmtTick = (v) => Math.round(v).toLocaleString();

  let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="Predicted versus actual fuel burn for ${points.length} holdout flights">`;

  tickVals.forEach((v) => {
    const gy = yPx(v);
    svg += `<line x1="${PAD.left}" y1="${gy}" x2="${W - PAD.right}" y2="${gy}" class="scatter-grid" />`;
    svg += `<text x="${PAD.left - 8}" y="${gy}" class="scatter-tick scatter-tick-y">${fmtTick(v)}</text>`;
    const gx = xPx(v);
    svg += `<text x="${gx}" y="${H - PAD.bottom + 20}" class="scatter-tick scatter-tick-x">${fmtTick(v)}</text>`;
  });

  svg += `<line x1="${PAD.left}" y1="${PAD.top}" x2="${PAD.left}" y2="${H - PAD.bottom}" class="scatter-axis" />`;
  svg += `<line x1="${PAD.left}" y1="${H - PAD.bottom}" x2="${W - PAD.right}" y2="${H - PAD.bottom}" class="scatter-axis" />`;
  svg += `<line x1="${xPx(min)}" y1="${yPx(min)}" x2="${xPx(max)}" y2="${yPx(max)}" class="scatter-diagonal" />`;

  svg += `<text x="${PAD.left + plotW / 2}" y="${H - 6}" class="scatter-axis-label">Actual fuel burn (kg)</text>`;
  svg += `<text x="16" y="${PAD.top + plotH / 2}" class="scatter-axis-label" transform="rotate(-90 16 ${PAD.top + plotH / 2})">Predicted fuel burn (kg)</text>`;

  points.forEach((p, i) => {
    const px = xPx(p.actual_kg);
    const py = yPx(p.predicted_kg);
    svg += `<circle cx="${px}" cy="${py}" r="4" class="scatter-dot" data-i="${i}" />`;
    svg += `<circle cx="${px}" cy="${py}" r="12" class="scatter-hit" data-i="${i}" tabindex="0" role="button" aria-label="${p.family}: actual ${formatNum(p.actual_kg)} kilograms, predicted ${formatNum(p.predicted_kg)} kilograms" />`;
  });

  svg += `</svg>`;
  container.innerHTML = svg;

  const tooltip = document.getElementById("scatter-tooltip");
  container.querySelectorAll(".scatter-hit").forEach((hit) => {
    const p = points[Number(hit.dataset.i)];
    const dot = container.querySelector(`.scatter-dot[data-i="${hit.dataset.i}"]`);

    const show = () => {
      dot.classList.add("scatter-dot-active");
      tooltip.innerHTML = "";
      const valueLine = document.createElement("div");
      valueLine.className = "scatter-tooltip-value";
      valueLine.textContent = `${formatNum(p.predicted_kg)} kg predicted vs ${formatNum(p.actual_kg)} kg actual`;
      const nameLine = document.createElement("div");
      nameLine.className = "scatter-tooltip-name";
      nameLine.textContent = p.family;
      tooltip.appendChild(valueLine);
      tooltip.appendChild(nameLine);

      const wrapRect = wrap.getBoundingClientRect();
      const hitRect = hit.getBoundingClientRect();
      tooltip.style.left = `${hitRect.left + hitRect.width / 2 - wrapRect.left}px`;
      tooltip.style.top = `${hitRect.top - wrapRect.top}px`;
      tooltip.hidden = false;
    };
    const hide = () => {
      dot.classList.remove("scatter-dot-active");
      tooltip.hidden = true;
    };

    hit.addEventListener("pointerenter", show);
    hit.addEventListener("pointerleave", hide);
    hit.addEventListener("focus", show);
    hit.addEventListener("blur", hide);
  });
}

document.getElementById("scatter-family-filter").addEventListener("change", (e) => {
  drawScatter(e.target.value);
});

let scatterResizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(scatterResizeTimer);
  scatterResizeTimer = setTimeout(() => {
    if (!document.getElementById("scatter-chart-wrap").hidden) {
      drawScatter(document.getElementById("scatter-family-filter").value);
    }
  }, 150);
});

document.getElementById("btn-load-families").addEventListener("click", async () => {
  hideError();
  if (!modelTrained) {
    document.getElementById("families-untrained").hidden = false;
    document.getElementById("result-summary").hidden = true;
    document.getElementById("result-families").hidden = true;
    document.getElementById("result-scatter").hidden = true;
    return;
  }
  document.getElementById("families-untrained").hidden = true;
  const btn = document.getElementById("btn-load-families");
  setBusy(btn, "Loading...", "Refresh");
  try {
    const [families, summary] = await Promise.all([api("/families"), api("/model/summary")]);
    const perFamily = (summary.holdout_validation && summary.holdout_validation.per_family) || {};

    // Training summary (persists across restarts, unlike the one-time /train response)
    document.getElementById("sum-trained-at").textContent = summary.trained_at
      ? new Date(summary.trained_at).toLocaleString()
      : "-";
    const from = formatTrainedFrom(summary.trained_from);
    const fromEl = document.getElementById("sum-trained-from");
    fromEl.textContent = from.text;
    fromEl.title = from.title;
    document.getElementById("sum-total").textContent = formatNum(summary.total_flights);
    const overall = summary.holdout_validation && summary.holdout_validation.overall;
    document.getElementById("sum-mape").textContent = overall ? `${overall.mape_pct}%` : "N/A";
    document.getElementById("sum-r2").textContent = overall ? `${(overall.r_squared * 100).toFixed(1)}%` : "N/A";
    document.getElementById("sum-rmse").textContent = overall ? `${formatNum(overall.rmse_kg)} kg` : "N/A";
    renderSourceBreakdown(summary.source_breakdown);
    document.getElementById("result-summary").hidden = false;

    // One card per aircraft type: a plain-language accuracy rating instead
    // of a bare MAPE number, flight count, and the top factors driving its
    // fuel burn as readable tags instead of raw snake_case feature names.
    // Sorted so aircraft needing attention show up first.
    document.getElementById("family-grid").innerHTML = Object.entries(families)
      .map(([fam, info]) => {
        const famStats = perFamily[fam];
        const rating = accuracyRating(famStats ? famStats.mape_pct : null);
        return { fam, info, famStats, rating };
      })
      .sort((a, b) => {
        const severity = RATING_SEVERITY[a.rating.cls] - RATING_SEVERITY[b.rating.cls];
        if (severity !== 0) return severity;
        const aMape = a.famStats ? a.famStats.mape_pct : -Infinity;
        const bMape = b.famStats ? b.famStats.mape_pct : -Infinity;
        return bMape - aMape || a.fam.localeCompare(b.fam);
      })
      .map(({ fam, info, famStats, rating }) => {
        const errorText = famStats
          ? `~${famStats.mape_pct}% typical error`
          : `~${formatNum(info.rmse_kg)} kg typical error`;

        const topFactors = Object.entries(info.feature_importance || {})
          .sort((a, b) => b[1] - a[1])
          .slice(0, 3)
          .map(([name]) => `<span class="factor-tag">${friendlyFactor(name)}</span>`)
          .join("");

        return `<div class="accuracy-card ${rating.cls}">
          <div class="accuracy-card-header">
            <span class="accuracy-card-title">${fam}</span>
            <span class="accuracy-badge ${rating.cls}">${rating.label}</span>
          </div>
          <div class="accuracy-card-metrics">
            <span>${formatNum(info.n_training_flights)} flights trained on</span>
            <span>${errorText}</span>
          </div>
          <div class="accuracy-card-factors">
            <span class="factor-label">What matters most for this aircraft:</span>
            <div class="factor-tags">${topFactors || '<span class="text-muted small">Not enough data yet</span>'}</div>
          </div>
        </div>`;
      })
      .join("");
    document.getElementById("result-families").hidden = false;

    document.getElementById("result-scatter").hidden = false;
    renderScatterChart((summary.holdout_validation && summary.holdout_validation.scatter) || []);

    familiesLoaded = true;
  } catch (err) {
    showError(err.message);
  } finally {
    clearBusy(btn);
  }
});
