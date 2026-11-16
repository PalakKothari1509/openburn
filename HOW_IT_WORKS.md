# OpenBurn — How the Code Works

A complete walkthrough of every part of the OpenBurn codebase, explaining
what each file does, how data flows through the system, and why things are
built the way they are.

---

## Table of Contents

1. [Big Picture — What OpenBurn Does](#1-big-picture)
2. [Project Structure](#2-project-structure)
3. [Data Flow (End to End)](#3-data-flow)
4. [Feature Extraction (`openburn/fuel/features.py`)](#4-feature-extraction)
5. [ML Model — The BADA Replacement (`openburn/aircraft/performance.py`)](#5-ml-model)
6. [High-Level Model API (`openburn/fuel/model.py`)](#6-high-level-model-api)
7. [Fallback Estimator (`openburn/fuel/fallback.py`)](#7-fallback-estimator)
8. [Emissions Reporting (`openburn/reporting/emissions.py`)](#8-emissions-reporting)
9. [Validation (`openburn/validation/compare.py`)](#9-validation)
10. [Flask Server (`api/app.py`)](#10-flask-server)
11. [Frontend (`frontend/`)](#11-frontend)
12. [Step-by-Step: What Happens When You Click "Calculate Fuel Burn"](#12-step-by-step)

---

## 1. Big Picture

OpenBurn estimates **how much jet fuel** an aircraft will burn on a flight.

Traditional systems (BADA, PyContrails) use physics equations — drag, thrust,
lift coefficients provided by EUROCONTROL. OpenBurn **replaces all of that**
with machine learning trained on **actual airline operational data** (real fuel
records from real flights).

**Input:** Departure airport, arrival airport, aircraft type (e.g. A320)
**Output:** Fuel burn in kg, CO2 emissions, and other pollutants

---

## 2. Project Structure

```
openburn/
├── openburn/                    # Core Python library
│   ├── fuel/
│   │   ├── features.py          # Reads CSV, extracts/computes features
│   │   ├── model.py             # Main API — train, estimate, estimate_route
│   │   └── fallback.py          # Backup estimator for unknown aircraft
│   ├── aircraft/
│   │   ├── registry.py          # ICAO type → real-world range/MTOW/speed lookup
│   │   └── performance.py       # ML models (one per aircraft family)
│   ├── reporting/
│   │   └── emissions.py         # Converts fuel kg → CO2, NOx, etc.
│   ├── validation/
│   │   └── compare.py           # Compares estimated vs actual fuel burn
│   └── trajectory/              # ADS-B processing (unused scaffolding, not
│       ├── adsb.py              # wired into the app yet — reserved for a
│       ├── segmentation.py      # future ADS-B-based ingestion feature)
│       └── interpolation.py
├── api/
│   ├── app.py                   # Flask API — main entry point, serves API + frontend
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── index.html               # Web dashboard
│   ├── style.css                # Dark theme styling
│   ├── script.js                # Frontend logic (API calls, rendering)
│   └── Dockerfile
├── data/
│   ├── airport_coords.csv       # 5,459 IATA airports with lat/lon
│   ├── sample_aircraft_types.csv
│   ├── sample_fuel_data.csv
│   └── training_history/        # accumulated CSVs from /train (gitignored)
├── tests/                       # pytest test suite
├── run_test.py                  # Standalone smoke-test/demo script
├── docker-compose.yml           # backend (Flask/gunicorn) + frontend (nginx)
└── pyproject.toml               # Python package configuration
```

---

## 3. Data Flow (End to End)

Here's what happens from raw CSV to fuel estimate, step by step:

```
┌─────────────────────────────────────────────────────────────┐
│                     TRAINING PHASE                          │
│                                                             │
│  CSV file (airline operational data)                        │
│       │                                                     │
│       ▼                                                     │
│  FeatureExtractor.load_fuel_csv()                           │
│    - Reads CSV                                              │
│    - Maps column names (OFFBLOCK_FUEL → offblock_fuel)      │
│    - Computes: fuel_burn_kg, duration, distance, payload    │
│       │                                                     │
│       ▼                                                     │
│  FeatureExtractor.prepare_training_data()                   │
│    - Drops rows with missing fuel/distance                  │
│    - Removes outliers (> 3 std deviations per family)       │
│       │                                                     │
│       ▼                                                     │
│  train_test_split (80% train, 20% test)                     │
│       │                                                     │
│       ▼                                                     │
│  PerformanceModel.train()                                   │
│    - Groups data by aircraft_family (A320, A339, E190...)   │
│    - Trains one HistGradientBoostingRegressor per family     │
│    - Computes cross-validated RMSE for each                 │
│       │                                                     │
│       ▼                                                     │
│  FallbackEstimator.fit()                                    │
│    - Learns fuel-per-km rates by category                   │
│    - Used when aircraft type has no trained model            │
│       │                                                     │
│       ▼                                                     │
│  _learn_speeds()                                            │
│    - Learns avg speed (km/min) per family from training data│
│    - Used to estimate flight duration from distance          │
│       │                                                     │
│       ▼                                                     │
│  Holdout validation on the 20% test set                     │
│    - MAPE, RMSE, R², bias per family                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   PREDICTION PHASE                          │
│                                                             │
│  User enters: departure=MAD, arrival=LHR, aircraft=A320    │
│       │                                                     │
│       ▼                                                     │
│  Look up airport coordinates from airport_coords.csv        │
│  MAD = (40.47, -3.56), LHR = (51.47, -0.46)               │
│       │                                                     │
│       ▼                                                     │
│  _haversine() → computes great-circle distance              │
│  MAD → LHR = ~1,264 km                                     │
│       │                                                     │
│       ▼                                                     │
│  _estimate_duration() → distance / learned_speed + overhead │
│  ~127 minutes for A320 at 1,264 km                         │
│       │                                                     │
│       ▼                                                     │
│  PerformanceModel.predict("A320", {distance, duration, …}) │
│    - Uses the trained A320 model                            │
│    - Fills missing features with NaN (model handles it)     │
│       │                                                     │
│       ▼                                                     │
│  EmissionsReport.from_fuel_kg(fuel_kg)                      │
│    - fuel × 3.16 = CO2                                     │
│    - fuel × 1.24 = H2O                                     │
│    - fuel × 0.014 = NOx     ...etc                         │
│       │                                                     │
│       ▼                                                     │
│  JSON response → frontend displays results                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Feature Extraction (`openburn/fuel/features.py`)

**Purpose:** Read raw airline CSV data and turn it into clean features for ML.

### The Problem It Solves

Airline CSVs don't come in a standard format. One airline might call it
`OFFBLOCK_FUEL`, another `fuel_burn_kg`, another `total_fuel_kg`. This
module handles all those variations.

### How It Works

**Column Mapping** (lines 52-119):
```python
COLUMN_ALIASES = {
    "fuel_burn_kg": ["fuel_burn_kg", "fuel_kg", "fuel_burn", ...],
    "aircraft_family": ["aircraft_family", "aircraft_type", "ac_subtype", ...],
    "origin": ["origin", "departure", "dep_ap_actual", ...],
    ...
}
```
When loading a CSV, it checks every column name against these aliases and
renames them to the standard name. So whether your CSV says `AC_SUBTYPE` or
`aircraft_type`, it becomes `aircraft_family` internally.

**Derived Fields** (lines 222-281):
If the CSV has raw operational data instead of pre-computed values, this
module computes:

| Derived Field | How It's Computed |
|---|---|
| `fuel_burn_kg` | `OFFBLOCK_FUEL - ONBLOCK_FUEL` (fuel at gate departure minus fuel at gate arrival) |
| `flight_duration_min` | `ONBLOCK time - OFFBLOCK time` (total gate-to-gate time) |
| `airborne_duration_min` | `LANDING time - AIRBORNE time` (time in the air only) |
| `taxi_time_min` | `flight_duration - airborne_duration` |
| `payload_kg` | `(passengers × 100kg) + cargo_kg + mail_kg` |
| `distance_km` | Haversine great-circle distance between departure and arrival airports |

**Outlier Removal** (lines 295-303):
For each aircraft family, it removes flights where fuel-per-km is more than
3 standard deviations from the mean. This catches data entry errors (e.g.
a typo making fuel 10x too high).

**Airport Coordinates** (lines 17-39):
`_AIRPORT_COORDS` is a global dictionary `{IATA_code: (lat, lon)}` loaded
from `data/airport_coords.csv`. The `_haversine()` function computes the
straight-line distance between two lat/lon points on Earth's surface.

---

## 5. ML Model — The BADA Replacement (`openburn/aircraft/performance.py`)

**Purpose:** This is the core of OpenBurn. Instead of physics equations,
it learns fuel burn patterns from historical data.

### Why One Model Per Aircraft Family?

An A320 burns fuel very differently from an A330. A short-haul narrowbody
vs a long-haul widebody have different weight, drag, and engine
characteristics. Training one model per family lets each model learn the
specific fuel burn curve for that aircraft type.

### The Algorithm: HistGradientBoostingRegressor

```python
model = HistGradientBoostingRegressor(
    max_iter=200,      # Up to 200 boosting rounds
    max_depth=5,       # Each tree can be 5 levels deep
    learning_rate=0.1, # How much each tree corrects the previous
    random_state=42,   # Reproducible results
)
```

**Why this algorithm?**
- It's a gradient boosting model (like XGBoost/LightGBM) built into scikit-learn
- **Key advantage:** It handles `NaN` values natively. This is critical because
  during prediction we often only know distance + duration, but not altitude,
  payload, etc. Other models would crash on missing values.
- It builds an ensemble of decision trees, where each new tree corrects the
  errors of the previous ones

### Features Used

The model uses whatever features are available in the training data:

| Feature | Description |
|---|---|
| `distance_km` | Great-circle distance (always available) |
| `flight_duration_min` | Gate-to-gate time |
| `airborne_duration_min` | Time in the air (excludes taxi) |
| `taxi_time_min` | Time taxiing on ground |
| `payload_kg` | Total weight of passengers + cargo + mail |
| `passengers` | Passenger count |
| `cargo_kg` | Cargo weight |
| `cruise_altitude_ft` | Cruising altitude |
| `average_speed_kts` | Average ground speed |
| `headwind_kts` | Wind component |
| `temperature_c` | Outside air temperature |
| `takeoff_weight_kg` | Gross takeoff weight |

### Training Process (lines 39-101)

```
For each aircraft family (A320, A339, E190, etc.):
  1. Filter data to just this family
  2. Skip if fewer than 10 flights (not enough data)
  3. Run 5-fold cross-validation to get RMSE estimate
  4. Train final model on all data for this family
  5. Store as PerformanceProfile (model + feature names + metrics)
```

### Prediction (lines 103-134)

```python
def predict(self, aircraft_family, features):
    profile = self._profiles.get(aircraft_family)  # Get the trained model
    if profile is None:
        return None  # No model for this aircraft → use fallback

    # Build feature row, filling missing features with NaN
    row = {f: features.get(f, np.nan) for f in profile.feature_names}
    X = pd.DataFrame([row])[profile.feature_names]
    prediction = profile.model.predict(X)[0]

    return {"fuel_burn_kg": round(max(0, prediction), 2), ...}
```

The key trick: `features.get(f, np.nan)` — if we don't know a feature
(e.g. no payload info), we pass NaN and the model still works because
HistGradientBoosting handles missing values internally.

---

## 6. High-Level Model API (`openburn/fuel/model.py`)

**Purpose:** The main class that ties everything together. This is what the
API server and tests interact with.

### Class: `FuelBurnModel`

Combines:
- `PerformanceModel` (ML models per aircraft family)
- `FeatureExtractor` (CSV reading + feature engineering)
- `FallbackEstimator` (backup for unknown aircraft)
- Airport coordinates (for distance calculation)

### Key Methods

#### `train_from_csv(csv_path, test_size=0.2)`
The training pipeline:
```
1. Load CSV → FeatureExtractor.load_fuel_csv()
2. Clean data → FeatureExtractor.prepare_training_data()
3. Split 80/20 → train_test_split (stratified by aircraft family)
4. Train ML models → PerformanceModel.train()
5. Train fallback → FallbackEstimator.fit()
6. Learn speeds → _learn_speeds()  (for route-based estimation)
7. Run holdout validation on the 20% test set
8. Return training summary with metrics
```

#### `estimate(aircraft_type, distance_km, ...optional features...)`
Single flight estimation:
```
1. Check model is trained
2. Build feature dict from provided parameters
3. Try PerformanceModel.predict() → returns "high confidence" result
4. If no model for this aircraft type → use FallbackEstimator → "low confidence"
```

#### `estimate_route(departure, arrival, aircraft_type, passengers=None)`
**This is the primary user-facing method.** Route-based estimation:
```
1. Look up airport coordinates: MAD → (40.47, -3.56)
2. Compute great-circle distance: MAD → LHR = 1,264 km
3. Estimate flight duration from learned speeds:
   - A320 flies at ~10 km/min (learned from training data)
   - duration = 1264/10 + 25min overhead = ~151 min
4. Estimate airborne time (subtract ~20min taxi)
5. Build features: {distance, duration, airborne_time, taxi_time, passengers, payload}
6. Call PerformanceModel.predict() or FallbackEstimator.estimate()
7. Calculate emissions: fuel × emission factors
8. Return: {fuel_burn_kg, route info, emissions, confidence}
```

#### `_learn_speeds(df)`
After training, this method learns the average cruising speed for each
aircraft family:
```python
# Filter out very short flights (< 100km or < 30min)
# For each family, compute: median(distance_km / flight_duration_min)
# Store as km/min → used later to estimate duration from distance
```

#### `batch_estimate(df)`
Bulk prediction for a DataFrame of flights. Uses **vectorized prediction**
(predicts all flights of the same family at once) for speed:
```python
for family in df["aircraft_family"].unique():
    mask = df["aircraft_family"] == family
    X = build_feature_matrix_for_family(...)
    predictions = profile.model.predict(X)  # All at once, not one by one
    df.loc[mask, "fuel_burn_estimated_kg"] = predictions
```

---

## 7. Fallback Estimator (`openburn/fuel/fallback.py`)

**Purpose:** When someone asks about an aircraft type we haven't trained a
model for (e.g. B747), this provides a rough estimate.

### How It Works

1. **Aircraft Classification** (lines 16-46):
   Classifies any ICAO type code into one of three categories:
   - **Widebody:** A330, A340, A350, A380, B747, B767, B777, B787
   - **Narrowbody:** A319, A320, A321, B737, B757, E170-195
   - **Turboprop:** ATR42/72, Dash 8

2. **Learning Rates** (lines 60-79):
   During training, it computes the median fuel-per-km for each category
   from the training data.

3. **Estimation** (lines 81-103):
   ```
   fuel_burn_kg = fuel_per_km_rate × distance_km
   ```
   Falls back to industry averages if no training data exists:
   - Widebody: 8.0 kg/km
   - Narrowbody: 3.5 kg/km
   - Turboprop: 1.5 kg/km

**This is marked "low confidence"** in the output — it's a rough estimate,
not the precise per-family ML model.

---

## 8. Emissions Reporting (`openburn/reporting/emissions.py`)

**Purpose:** Convert fuel burn (kg) into greenhouse gas emissions.

### Emission Factors (ICAO/IPCC standard)

For every 1 kg of jet fuel burned:

| Pollutant | Factor | Example: 5,000 kg fuel |
|---|---|---|
| CO2 | 3.16 kg | 15,800 kg |
| H2O (water vapour) | 1.24 kg | 6,200 kg |
| NOx | 0.014 kg | 70 kg |
| SO2 | 0.0012 kg | 6 kg |
| CO | 0.0028 kg | 14 kg |
| Hydrocarbons | 0.0004 kg | 2 kg |
| Particulate matter | 0.00004 kg | 0.2 kg |

### Methods

- `from_fuel_kg(fuel_kg)` — Single flight: returns dict of all emissions
- `from_fuel_litres(fuel_litres)` — Converts litres to kg first (× 0.804 density)
- `batch_report(df, fuel_col)` — Adds emission columns to a DataFrame
- `aggregate(df, group_by)` — Sums emissions grouped by airline/route/period

---

## 9. Validation (`openburn/validation/compare.py`)

**Purpose:** Measure how accurate our estimates are by comparing them to
actual fuel burn values.

### Metrics Computed

| Metric | What It Means |
|---|---|
| **MAE** (Mean Absolute Error) | Average difference in kg between estimate and actual |
| **RMSE** (Root Mean Squared Error) | Like MAE but penalizes large errors more |
| **MAPE** (Mean Absolute Percentage Error) | Average % difference — **the key metric** |
| **R²** (R-squared) | How much variance the model explains (1.0 = perfect) |
| **Bias** | Positive = we overestimate, negative = we underestimate |

### How Validation Works

During training, 20% of flights are held out (never seen by the model).
After training, we predict fuel for those held-out flights and compare
against the actual fuel values.

```
Holdout validation results (January 2025 data, ~2,000 test flights):
  Overall MAPE: ~3.3%  ← Our estimates are within 3.3% of actual fuel burn
  R²: > 0.99           ← Model explains 99%+ of fuel burn variance
```

`validate_by_family()` breaks this down per aircraft type so you can see
which models are strong (e.g. A320: 2.5% MAPE) vs weak (e.g. A332: 3.7%
MAPE due to fewer training flights).

---

## 10. Flask Server (`api/app.py`)

**Purpose:** Serves both the REST API and the web frontend on port 5000.
This is the project's only server entry point (an earlier `run_server.py`
was removed as dead code once `api/app.py` took over this role).

### How It Starts

```python
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests — open to any origin, no auth

# Model is created and a saved copy is loaded from disk immediately at
# import time (NOT lazily on first request) so /health reflects real
# state right away:
model = FuelBurnModel()
emissions = EmissionsReport()

if model.load_model(MODEL_PATH):
    print(f"Loaded saved model from {MODEL_PATH} — no need to retrain.")
```

Run directly with `python api/app.py` (dev, `debug=False`) or via
`gunicorn app:app` from the `api/` directory (production, see
`api/Dockerfile`).

### API Endpoints

| Method | Endpoint | What It Does |
|---|---|---|
| GET | `/` | Serves the frontend (`index.html`) |
| GET | `/style.css`, `/script.js` | Serves the frontend's static assets |
| GET | `/health` | Returns model status (trained/not, which families, last trained time) |
| POST | `/train` | Trains model from one or more server-side CSV paths. Body: `{"csv_path": "..."}` or `{"csv_paths": [...]}`, optional `"replace": true` |
| POST | `/train/upload` | Same as `/train` but the CSV(s) are uploaded as `multipart/form-data` instead of referenced by path |
| POST | `/estimate` | Single flight estimate. Body: `{"aircraft_type": "A320", "distance_km": 1500}` |
| POST | `/estimate/route` | Route estimate. Body: `{"departure": "MAD", "arrival": "LHR", "aircraft_type": "A320"}` |
| POST | `/estimate/batch` | Batch estimate. Body: `{"flights": [{...}, {...}]}` |
| GET | `/airports` | List all known airport codes |
| GET | `/model/summary` | Full training summary (holdout validation, flight counts) — persists across restarts |
| POST | `/data/summary` | Summarize a CSV's columns/stats without training on it. Body: `{"csv_path": "..."}` |
| GET | `/families` | List trained aircraft families with metrics |

Every `/train*`, `/estimate*`, and `/data/summary` handler parses the
request body with `request.get_json(silent=True) or {}`, so a missing or
malformed JSON body falls through to the existing "missing required
field" `400` response instead of crashing with a raw `500`.

**Note:** `/train` and `/data/summary` accept a `csv_path` taken directly
from the request body with no path restriction — fine for a trusted,
single-user local tool, but something to lock down (e.g. an allow-listed
directory) before exposing this API beyond localhost.

### Helper Functions

- `_clean_numpy(obj)` — Converts numpy types (`np.int64`, `np.float64`)
  to native Python types so Flask can serialize them to JSON
- `_float_or_none(val)` — Safely converts a value to float, returning
  `None` if it can't (e.g. an empty string from a form field)

---

## 11. Frontend (`frontend/`)

### `index.html`
A single-page app with 5 tabs:
- **Route Estimate** (primary) — Enter departure + arrival + aircraft type
- **Advanced** — Manually enter distance, duration, altitude
- **Batch** — Paste JSON array of flights
- **Train Model** — Enter CSV file path to train the model
- **Families** — View trained aircraft families and their accuracy

### `script.js`

**API helper** (lines 24-32):
```javascript
async function api(endpoint, options = {}) {
    const res = await fetch(`${API_BASE}${endpoint}`, {
        headers: {"Content-Type": "application/json"},
        ...options,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Unknown error");
    return data;
}
```
All API calls go through this function. `API_BASE` is `window.location.origin`
(same server), so no CORS issues.

**Status check on page load** (lines 39-55):
When the page loads, it calls `/health` to check if the model is trained
and shows a status badge ("Model trained" or "Model not trained").

**Route estimate form** (lines 60-98):
1. Collects departure, arrival, aircraft_type, passengers from the form
2. Sends POST to `/estimate/route`
3. Displays: fuel burn, CO2, distance, duration, confidence, RMSE
4. Shows full emissions breakdown in a collapsible section

### `style.css`
Dark theme using CSS variables. The main design elements:
- `.card` — Each tab section is a card with rounded corners
- `.result-grid` — CSS grid layout for displaying results
- `.route-header` — Shows "MAD → LHR (A320)" after calculation
- `.status-badge` — Green when trained, yellow when not

---

## 12. Step-by-Step: What Happens When You Click "Calculate Fuel Burn"

Let's trace a complete request: **MAD → LHR on an A320 with 150 passengers.**

### Frontend (script.js)

```
1. User fills form: departure=MAD, arrival=LHR, aircraft=A320, passengers=150
2. Form submit event fires
3. JavaScript builds JSON body:
   {"departure": "MAD", "arrival": "LHR", "aircraft_type": "A320", "passengers": "150"}
4. Sends POST to http://localhost:5000/estimate/route
```

### Server (api/app.py)

```
5. Flask routes request to estimate_route() function
6. data = request.get_json(silent=True) or {} — the model was already
   loaded/trained at server startup, so there's no lazy-init step here
7. Extracts fields from JSON body
8. Calls model.estimate_route("MAD", "LHR", "A320", passengers=150)
```

### Model (openburn/fuel/model.py → estimate_route)

```
9.  Look up MAD coordinates: (40.4722, -3.5611)
10. Look up LHR coordinates: (51.4700, -0.4543)
11. Compute haversine distance: ~1,264 km
12. Look up A320's learned speed: ~10.2 km/min (from training data)
13. Estimate duration: 1264 / 10.2 + 26.3 overhead = ~150 min
14. Estimate airborne time: 150 - 20 = ~130 min (subtract taxi)
15. Build features dict:
    {
        distance_km: 1264,
        flight_duration_min: 150,
        airborne_duration_min: 130,
        taxi_time_min: 20,
        passengers: 150,
        payload_kg: 15000   (150 pax × 100 kg each)
    }
```

### Performance Model (openburn/aircraft/performance.py → predict)

```
16. Look up trained A320 model (PerformanceProfile)
17. Build feature row matching model's expected features
    (fills any missing features with NaN)
18. model.predict(X) → e.g. 4,850 kg
19. Return: {fuel_burn_kg: 4850, aircraft_family: "A320", model_rmse_kg: 95, ...}
```

### Back in model.py

```
20. Add route info: {departure: "MAD", arrival: "LHR", distance_km: 1264, ...}
21. Calculate emissions:
    EmissionsReport.from_fuel_kg(4850) →
    {co2: 15,326 kg, h2o: 6,014 kg, nox: 67.9 kg, ...}
22. Return complete result dict
```

### Back in api/app.py

```
23. _clean_numpy() converts any numpy types to native Python
24. Flask serializes to JSON and sends HTTP response
```

### Frontend (script.js)

```
25. Receives JSON response
26. Updates DOM elements:
    - route-fuel → "4,850"
    - route-co2 → "15,326"
    - route-distance → "1,264"
    - route-duration → "150"
    - route-confidence → "high"
    - route-rmse → "95"
27. Shows emissions breakdown in collapsible section
28. Unhides the result div
```

**The user sees:** A card showing MAD → LHR, 4,850 kg fuel, 15,326 kg CO2,
1,264 km distance, ~150 min duration, high confidence, 95 kg RMSE.

---

## Key Design Decisions — Why Things Are Built This Way

| Decision | Why |
|---|---|
| **One ML model per aircraft family** | Different aircraft have very different fuel curves. A single model would be inaccurate. |
| **HistGradientBoostingRegressor** | Handles NaN natively — critical because prediction often has missing features. |
| **Haversine for distance** | Great-circle distance is a good approximation. Real routes are ~5-10% longer due to airways, but the model learns to compensate. |
| **Learned speeds** | Instead of hardcoding aircraft speeds, we learn them from training data. More accurate for the specific airline's operations. |
| **Lazy model loading** | Flask starts instantly. Heavy imports (sklearn, pandas) only load when first API call comes in. |
| **Fallback estimator** | If someone asks about a B747 but we only trained on A320/A330/etc., we still give a rough estimate rather than failing. |
| **80/20 holdout split** | Trains on 80% of data, validates on 20% the model never saw. This gives honest accuracy metrics. |
| **Column aliases** | Airlines use different CSV formats. The alias system auto-detects columns instead of requiring a specific format. |
