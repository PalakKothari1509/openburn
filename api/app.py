"""OpenBurn REST API — aviation fuel burn estimation service."""

import os
import shutil
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from openburn.fuel.model import FuelBurnModel
from openburn.fuel.features import FeatureExtractor
from openburn.reporting.emissions import EmissionsReport
from openburn.validation.compare import ValidationReport

FRONTEND_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
)
MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "trained_model.joblib")
)
# Every CSV ever trained on gets kept here, so a new upload ADDS to the
# model instead of silently replacing everything that came before it.
TRAINING_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "training_history")
)
os.makedirs(TRAINING_DATA_DIR, exist_ok=True)


def _accumulated_csv_paths() -> list[str]:
    """All CSVs trained on so far (old + new), used so every /train call
    retrains on the full accumulated history, not just the latest file."""
    return sorted(
        os.path.join(TRAINING_DATA_DIR, f)
        for f in os.listdir(TRAINING_DATA_DIR)
        if f.lower().endswith(".csv")
    )


def _clear_training_history() -> None:
    for f in os.listdir(TRAINING_DATA_DIR):
        if f.lower().endswith(".csv"):
            os.remove(os.path.join(TRAINING_DATA_DIR, f))

app = Flask(__name__)
CORS(app)

# Global model instance — loaded from disk if a saved model exists, otherwise
# trained via /train or /train/upload
model = FuelBurnModel()
emissions = EmissionsReport()

if model.load_model(MODEL_PATH):
    print(f"Loaded saved model from {MODEL_PATH} — no need to retrain.")


# --- Frontend serving ---
@app.route("/")
def serve_frontend():
    with open(os.path.join(FRONTEND_DIR, "index.html"), encoding="utf-8") as f:
        return f.read(), 200, {"Content-Type": "text/html"}


@app.route("/style.css")
def serve_css():
    with open(os.path.join(FRONTEND_DIR, "style.css"), encoding="utf-8") as f:
        return f.read(), 200, {"Content-Type": "text/css"}


@app.route("/script.js")
def serve_js():
    with open(os.path.join(FRONTEND_DIR, "script.js"), encoding="utf-8") as f:
        return f.read(), 200, {"Content-Type": "application/javascript"}


# --- API endpoints ---
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_trained": model._is_trained,
        "families_available": model.performance.families if model._is_trained else [],
        "trained_at": model._trained_at,
        "trained_from": model._trained_from,
    })


@app.route("/train", methods=["POST"])
def train():
    """Train model from one or more CSV file paths.

    By default this ADDS to everything previously trained on — the new
    file(s) are kept alongside every past one, and the model retrains on
    the full combined history. Pass "replace": true to discard previous
    data and start fresh with only the file(s) given here.

    POST body: {"csv_path": "/path/to/fuel_data.csv"}
    or:        {"csv_paths": ["/path/jan.csv", "/path/feb.csv"]} to combine several
    optional:  "replace": true
    """
    data = request.get_json()
    csv_paths = data.get("csv_paths")
    if not csv_paths and data.get("csv_path"):
        csv_paths = [data["csv_path"]]
    if not csv_paths:
        return jsonify({"error": "csv_path or csv_paths is required"}), 400

    test_size = float(data.get("test_size", 0.2))
    try:
        if data.get("replace"):
            _clear_training_history()
        for p in csv_paths:
            shutil.copy(p, os.path.join(TRAINING_DATA_DIR, os.path.basename(p)))

        result = model.train_from_csv(_accumulated_csv_paths(), test_size=test_size)
        model.save_model(MODEL_PATH)
        return jsonify({"status": "trained", **_clean_numpy(result)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/train/upload", methods=["POST"])
def train_upload():
    """Train model from one or more uploaded CSV files.

    By default this ADDS to everything previously trained on — uploaded
    files are kept alongside every past one, and the model retrains on
    the full combined history. Pass "replace": "true" (form field) to
    discard previous data and start fresh with only the file(s) given here.

    multipart/form-data: one or more "file" fields (the CSVs), optional
    "test_size" and "replace" fields.
    """
    files = [f for f in request.files.getlist("file") if f.filename]
    if not files:
        return jsonify({"error": "No file uploaded"}), 400
    for f in files:
        if not f.filename.lower().endswith(".csv"):
            return jsonify({"error": f"File must be a .csv: {f.filename}"}), 400

    test_size = float(request.form.get("test_size", 0.2))
    replace = request.form.get("replace", "").lower() in ("true", "1", "yes", "on")

    try:
        if replace:
            _clear_training_history()
        for f in files:
            f.save(os.path.join(TRAINING_DATA_DIR, secure_filename(f.filename)))

        result = model.train_from_csv(_accumulated_csv_paths(), test_size=test_size)
        model.save_model(MODEL_PATH)
        return jsonify({"status": "trained", **_clean_numpy(result)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/estimate", methods=["POST"])
def estimate():
    """Estimate fuel burn for a single flight.

    POST body: {
        "aircraft_type": "B738",
        "distance_km": 1500,
        "flight_duration_min": 180,     // optional
        "cruise_altitude_ft": 35000,    // optional
        "average_speed_kts": 450        // optional
    }
    """
    data = request.get_json()

    required = ["aircraft_type", "distance_km"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        result = model.estimate(
            aircraft_type=data["aircraft_type"],
            distance_km=float(data["distance_km"]),
            flight_duration_min=_float_or_none(data.get("flight_duration_min")),
            cruise_altitude_ft=_float_or_none(data.get("cruise_altitude_ft")),
            average_speed_kts=_float_or_none(data.get("average_speed_kts")),
            payload_kg=_float_or_none(data.get("payload_kg")),
        )

        # Add emissions
        result["emissions"] = emissions.from_fuel_kg(result["fuel_burn_kg"])
        return jsonify(result)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/estimate/batch", methods=["POST"])
def estimate_batch():
    """Estimate fuel burn for multiple flights.

    POST body: {"flights": [{"aircraft_type": "B738", "distance_km": 1500}, ...]}
    """
    import pandas as pd

    data = request.get_json()
    flights = data.get("flights", [])
    if not flights:
        return jsonify({"error": "No flights provided"}), 400

    try:
        df = pd.DataFrame(flights)
        if "aircraft_type" in df.columns and "aircraft_family" not in df.columns:
            df.rename(columns={"aircraft_type": "aircraft_family"}, inplace=True)
        result_df = model.batch_estimate(df)

        # Add emissions
        result_df = emissions.batch_report(result_df, fuel_col="fuel_burn_estimated_kg")

        return jsonify(result_df.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/estimate/route", methods=["POST"])
def estimate_route():
    """Estimate fuel burn from departure, arrival, and aircraft type.

    POST body: {
        "departure": "MAD",
        "arrival": "LHR",
        "aircraft_type": "A320",
        "passengers": 150           // optional
    }
    """
    data = request.get_json()

    required = ["departure", "arrival", "aircraft_type"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        result = model.estimate_route(
            departure=data["departure"],
            arrival=data["arrival"],
            aircraft_type=data["aircraft_type"],
            passengers=int(data["passengers"]) if data.get("passengers") else None,
        )
        return jsonify(_clean_numpy(result))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/airports", methods=["GET"])
def list_airports():
    """List all known airport codes."""
    return jsonify({"airports": model.get_airports()})


@app.route("/model/summary", methods=["GET"])
def model_summary():
    """Full training summary (holdout validation, flight counts, etc).

    Unlike the one-time response from /train, this persists across
    server restarts since it's saved alongside the model.
    """
    if not model._is_trained:
        return jsonify({"error": "Model not trained yet"}), 400
    return jsonify(_clean_numpy({
        "trained_at": model._trained_at,
        "trained_from": model._trained_from,
        **model._training_summary,
    }))


@app.route("/data/summary", methods=["POST"])
def data_summary():
    """Get summary of a fuel data CSV without training.

    POST body: {"csv_path": "/path/to/fuel_data.csv"}
    """
    data = request.get_json()
    csv_path = data.get("csv_path")
    if not csv_path:
        return jsonify({"error": "csv_path is required"}), 400

    try:
        extractor = FeatureExtractor()
        df = extractor.load_fuel_csv(csv_path)
        summary = extractor.summary(df)
        return jsonify(summary)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/families", methods=["GET"])
def list_families():
    """List trained aircraft families and their metrics."""
    if not model._is_trained:
        return jsonify({"error": "Model not trained yet"}), 400

    families = {}
    for fam in model.performance.families:
        profile = model.performance.get_profile(fam)
        if profile:
            families[fam] = {
                "rmse_kg": profile.rmse_cv,
                "n_training_flights": profile.n_training_flights,
                "features": profile.feature_names,
                "feature_importance": model.performance.feature_importance(fam),
            }
    return jsonify(families)


def _clean_numpy(obj):
    """Convert numpy types to native Python for JSON serialization."""
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _clean_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_numpy(v) for v in obj]
    return obj


def _float_or_none(val):
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
