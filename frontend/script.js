const API_BASE = window.location.origin;

// Tab switching
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll("section.card").forEach((s) => s.classList.add("hidden"));
    btn.classList.add("active");
    document.getElementById(`tab-${btn.dataset.tab}`).classList.remove("hidden");
    hideError();
  });
});

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
      badge.textContent = `Model trained (${data.families_available.length} families: ${data.families_available.join(", ")})${trainedNote}`;
      badge.className = "status-badge status-trained";
    } else {
      badge.textContent = "Model not trained — go to Train Model tab";
      badge.className = "status-badge status-untrained";
    }
  } catch {
    document.getElementById("model-status").textContent = "API offline — start the server with: python api/app.py";
    document.getElementById("model-status").className = "status-badge status-untrained";
  }
}
checkStatus();

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

  try {
    const data = await api("/estimate/route", { method: "POST", body: JSON.stringify(body) });

    // Route header
    document.getElementById("route-header").innerHTML =
      `<span class="route-label">${data.route.departure}</span>` +
      `<span class="route-arrow">→</span>` +
      `<span class="route-label">${data.route.arrival}</span>` +
      `<span class="route-aircraft">${data.aircraft_family}</span>`;

    document.getElementById("route-fuel").textContent = formatNum(data.fuel_burn_kg);
    document.getElementById("route-co2").textContent = formatNum(data.emissions.co2);
    document.getElementById("route-distance").textContent = formatNum(data.route.distance_km);
    document.getElementById("route-duration").textContent = formatNum(Math.round(data.route.estimated_duration_min));
    document.getElementById("route-confidence").textContent = data.confidence;
    document.getElementById("route-rmse").textContent = data.model_rmse_kg || "N/A";

    const routeWarning = document.getElementById("route-warning");
    if (data.warning) {
      routeWarning.textContent = `⚠ ${data.warning}`;
      routeWarning.hidden = false;
    } else {
      routeWarning.hidden = true;
    }

    // Emissions breakdown
    const breakdown = document.getElementById("route-emissions-breakdown");
    breakdown.innerHTML = Object.entries(data.emissions)
      .map(([k, v]) => `<div class="emission-row"><span>${k.toUpperCase()}</span><span>${formatNum(v)} kg</span></div>`)
      .join("");

    document.getElementById("result-route").hidden = false;
  } catch (err) {
    showError(err.message);
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

  try {
    const data = await api("/estimate", { method: "POST", body: JSON.stringify(body) });
    document.getElementById("est-fuel").textContent = formatNum(data.fuel_burn_kg);
    document.getElementById("est-co2").textContent = formatNum(data.emissions.co2);
    document.getElementById("est-method").textContent = data.method;
    document.getElementById("est-confidence").textContent = data.confidence;

    const estWarning = document.getElementById("est-warning");
    if (data.warning) {
      estWarning.textContent = `⚠ ${data.warning}`;
      estWarning.hidden = false;
    } else {
      estWarning.hidden = true;
    }

    const breakdown = document.getElementById("emissions-breakdown");
    breakdown.innerHTML = Object.entries(data.emissions)
      .map(([k, v]) => `<div class="emission-row"><span>${k.toUpperCase()}</span><span>${formatNum(v)} kg</span></div>`)
      .join("");

    document.getElementById("result-estimate").hidden = false;
  } catch (err) {
    showError(err.message);
  }
});

// ============================================================
// Batch estimate
// ============================================================
document.getElementById("form-batch").addEventListener("submit", async (e) => {
  e.preventDefault();
  hideError();
  const fd = new FormData(e.target);
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
  }
});

// ============================================================
// Train model
// ============================================================
document.getElementById("form-train").addEventListener("submit", async (e) => {
  e.preventDefault();
  hideError();
  const fd = new FormData(e.target);
  const btn = e.target.querySelector("button");
  btn.textContent = "Training...";
  btn.disabled = true;
  try {
    const csvPaths = fd.get("csv_path").split(",").map((p) => p.trim()).filter(Boolean);
    const data = await api("/train", {
      method: "POST",
      body: JSON.stringify({ csv_paths: csvPaths }),
    });
    document.getElementById("train-output").textContent = JSON.stringify(data, null, 2);
    document.getElementById("result-train").hidden = false;
    checkStatus();
  } catch (err) {
    showError(err.message);
  } finally {
    btn.textContent = "Train Model";
    btn.disabled = false;
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
  btn.textContent = "Uploading & Training...";
  btn.disabled = true;
  try {
    // Don't use the api() helper here — it forces a JSON content-type,
    // which breaks multipart/form-data file uploads. Let fetch set its own.
    const res = await fetch(`${API_BASE}/train/upload`, { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Unknown error");
    document.getElementById("train-output").textContent = JSON.stringify(data, null, 2);
    document.getElementById("result-train").hidden = false;
    checkStatus();
  } catch (err) {
    showError(err.message);
  } finally {
    btn.textContent = "Upload & Train";
    btn.disabled = false;
  }
});

// ============================================================
// Load families
// ============================================================
document.getElementById("btn-load-families").addEventListener("click", async () => {
  hideError();
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
    document.getElementById("sum-mape").textContent = overall ? overall.mape_pct : "N/A";
    document.getElementById("sum-r2").textContent = overall ? overall.r_squared : "N/A";
    document.getElementById("sum-rmse").textContent = overall ? formatNum(overall.rmse_kg) : "N/A";
    document.getElementById("result-summary").hidden = false;

    // Per-family table, with top 3 most important factors for each aircraft
    const tbody = document.getElementById("families-table").querySelector("tbody");
    tbody.innerHTML = Object.entries(families)
      .map(([fam, info]) => {
        const topFactors = Object.entries(info.feature_importance || {})
          .sort((a, b) => b[1] - a[1])
          .slice(0, 3)
          .map(([name]) => name)
          .join(", ");
        return `<tr><td>${fam}</td><td>${formatNum(info.rmse_kg)}</td><td>${formatNum(info.n_training_flights)}</td><td>${topFactors}</td></tr>`;
      })
      .join("");
    document.getElementById("result-families").hidden = false;
  } catch (err) {
    showError(err.message);
  }
});
