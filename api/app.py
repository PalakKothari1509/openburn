"""OpenBurn REST API — aviation fuel burn estimation service."""

from flask import Flask, request, jsonify
from flask_cors import CORS

from openburn.fuel.model import FuelBurnModel
from openburn.fuel.features import FeatureExtractor
from openburn.reporting.emissions import EmissionsReport
from openburn.validation.compare import ValidationReport

app = Flask(__name__)
CORS(app)

# Global model instance — trained on startup or via /train endpoint
model = FuelBurnModel()
emissions = EmissionsReport()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_trained": model._is_trained,
        "families_available": model.performance.families if model._is_trained else [],
    })


@app.route("/train", methods=["POST"])
def train():
    """Train model from a CSV file path.

    POST body: {"csv_path": "/path/to/fuel_data.csv"}
    """
    data = request.get_json()
    csv_path = data.get("csv_path")
    if not csv_path:
        return jsonify({"error": "csv_path is required"}), 400

    test_size = float(data.get("test_size", 0.2))
    try:
        result = model.train_from_csv(csv_path, test_size=test_size)
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
