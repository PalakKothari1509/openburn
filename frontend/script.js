const API_BASE = "http://localhost:5000";

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

// Check model status on load
async function checkStatus() {
  try {
    const data = await api("/health");
    const badge = document.getElementById("model-status");
    if (data.model_trained) {
      badge.textContent = `Model trained (${data.families_available.length} families)`;
      badge.className = "status-badge status-trained";
    } else {
      badge.textContent = "Model not trained";
      badge.className = "status-badge status-untrained";
    }
  } catch {
    document.getElementById("model-status").textContent = "API offline";
  }
}
checkStatus();

// Single estimate
document.getElementById("form-estimate").addEventListener("submit", async (e) => {
  e.preventDefault();
  hideError();
  const fd = new FormData(e.target);
  const body = { aircraft_type: fd.get("aircraft_type"), distance_km: fd.get("distance_km") };
  if (fd.get("flight_duration_min")) body.flight_duration_min = fd.get("flight_duration_min");
  if (fd.get("cruise_altitude_ft")) body.cruise_altitude_ft = fd.get("cruise_altitude_ft");
  if (fd.get("average_speed_kts")) body.average_speed_kts = fd.get("average_speed_kts");

  try {
    const data = await api("/estimate", { method: "POST", body: JSON.stringify(body) });
    document.getElementById("est-fuel").textContent = data.fuel_burn_kg.toLocaleString();
    document.getElementById("est-co2").textContent = data.emissions.co2.toLocaleString();
    document.getElementById("est-method").textContent = data.method;
    document.getElementById("est-confidence").textContent = data.confidence;
    document.getElementById("est-rmse").textContent = data.model_rmse_kg || "N/A";

    // Emissions breakdown
    const breakdown = document.getElementById("emissions-breakdown");
    breakdown.innerHTML = Object.entries(data.emissions)
      .map(([k, v]) => `<div class="emission-row"><span>${k.toUpperCase()}</span><span>${v.toLocaleString()} kg</span></div>`)
      .join("");

    document.getElementById("result-estimate").hidden = false;
  } catch (err) {
    showError(err.message);
  }
});

// Batch estimate
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

    const cols = ["aircraft_family", "distance_km", "fuel_burn_estimated_kg", "co2_kg", "estimation_method", "estimation_confidence"];
    const available = cols.filter((c) => c in data[0]);

    thead.innerHTML = available.map((c) => `<th>${c}</th>`).join("");
    tbody.innerHTML = data
      .map((row) => `<tr>${available.map((c) => `<td>${row[c] ?? ""}</td>`).join("")}</tr>`)
      .join("");

    document.getElementById("result-batch").hidden = false;
  } catch (err) {
    showError(err.message);
  }
});

// Train model
document.getElementById("form-train").addEventListener("submit", async (e) => {
  e.preventDefault();
  hideError();
  const fd = new FormData(e.target);
  try {
    const data = await api("/train", {
      method: "POST",
      body: JSON.stringify({ csv_path: fd.get("csv_path") }),
    });
    document.getElementById("train-output").textContent = JSON.stringify(data, null, 2);
    document.getElementById("result-train").hidden = false;
    checkStatus();
  } catch (err) {
    showError(err.message);
  }
});

// Load families
document.getElementById("btn-load-families").addEventListener("click", async () => {
  hideError();
  try {
    const data = await api("/families");
    const tbody = document.getElementById("families-table").querySelector("tbody");
    tbody.innerHTML = Object.entries(data)
      .map(([fam, info]) => {
        const topFeature = info.feature_importance
          ? Object.entries(info.feature_importance).sort((a, b) => b[1] - a[1])[0]?.[0] || "-"
          : "-";
        return `<tr><td>${fam}</td><td>${info.rmse_kg}</td><td>${info.n_training_flights}</td><td>${topFeature}</td></tr>`;
      })
      .join("");
    document.getElementById("result-families").hidden = false;
  } catch (err) {
    showError(err.message);
  }
});
