"""Core fuel burn estimation model."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from openburn.aircraft.performance import PerformanceModel
from openburn.aircraft.registry import AircraftRegistry
from openburn.fuel.features import (
    FeatureExtractor, load_airport_coords, _AIRPORT_COORDS, _haversine,
)
from openburn.fuel.fallback import FallbackEstimator

# Default airport coords file (relative to package)
_DEFAULT_AIRPORT_COORDS = Path(__file__).resolve().parent.parent.parent / "data" / "airport_coords.csv"
# Default aircraft type data (real-world range, used to flag unrealistic distances)
_DEFAULT_AIRCRAFT_TYPES = Path(__file__).resolve().parent.parent.parent / "data" / "sample_aircraft_types.csv"


class FuelBurnModel:
    """High-level fuel burn estimation interface.

    Combines the performance model (per-aircraft-family ML models),
    feature extraction, and fallback estimation into a single API.
    """

    def __init__(self, airport_coords_path: str | Path | None = None) -> None:
        self.performance = PerformanceModel()
        self.features = FeatureExtractor()
        self.fallback = FallbackEstimator()
        self._is_trained = False
        self._training_summary: dict = {}
        self._duration_models: dict[str, tuple[float, float]] = {}  # family -> (overhead_min, min_per_km)
        self._trained_at: Optional[str] = None  # ISO timestamp of last training
        self._trained_from: Optional[list[str]] = None  # source CSV filename(s)

        # Load airport coordinates for distance calculation
        coords_path = Path(airport_coords_path) if airport_coords_path else _DEFAULT_AIRPORT_COORDS
        if coords_path.exists():
            load_airport_coords(str(coords_path))

        # Load real-world aircraft range data, used to flag distances that
        # aren't realistic for the given aircraft (see _range_warning)
        self.aircraft_registry = AircraftRegistry()
        if _DEFAULT_AIRCRAFT_TYPES.exists():
            self.aircraft_registry.load_csv(_DEFAULT_AIRCRAFT_TYPES)

    def train_from_csv(
        self,
        fuel_csv_path: str | Path | list[str | Path],
        test_size: float = 0.2,
        random_state: int = 42,
        source_names: Optional[list[str]] = None,
        progress_callback: Optional[Callable[[str, str], None]] = None,
        **kwargs,
    ) -> dict:
        """Train the model from one or more CSVs of historical fuel burn data.

        Multiple files are concatenated into a single dataset before
        cleaning/training, so e.g. several months of data can be combined
        into one model instead of each call only seeing one file.

        Args:
            fuel_csv_path: Path (or list of paths) to CSV(s) with flight
                records and fuel burn.
            test_size: Fraction of data to hold out for validation (0-1).
                Set to 0 to train on all data (no validation).
            random_state: Random seed for reproducibility.
            source_names: Optional display names for the source file(s),
                used instead of the actual path (e.g. when the path is a
                temp file from an upload but the original filename is
                more useful to show).

        Returns:
            Training summary with per-family metrics and validation results.
        """
        paths = [fuel_csv_path] if isinstance(fuel_csv_path, (str, Path)) else list(fuel_csv_path)

        def report(stage: str, detail: str) -> None:
            if progress_callback:
                progress_callback(stage, detail)

        report("loading", f"Reading {len(paths)} file(s)...")

        # Tag each row with its source file before combining, so we can
        # report how many rows each file actually contributed after
        # cleaning — otherwise a file that's silently mostly dropped (bad
        # encoding, wrong columns, a truncated upload, ...) is invisible,
        # and a lower-than-expected total looks identical to "working as
        # intended" from the outside.
        raw_counts: dict[str, int] = {}
        frames = []
        for p in paths:
            frame = self.features.load_fuel_csv(str(p), **kwargs)
            name = Path(p).name
            frame["_source_file"] = name
            raw_counts[name] = raw_counts.get(name, 0) + len(frame)
            frames.append(frame)
        df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        df = self.features.prepare_training_data(df)

        cleaned_counts = df["_source_file"].value_counts().to_dict() if "_source_file" in df.columns else {}
        source_breakdown = [
            {"file": name, "raw_rows": raw_counts[name], "rows_used": int(cleaned_counts.get(name, 0))}
            for name in raw_counts
        ]
        if "_source_file" in df.columns:
            df = df.drop(columns=["_source_file"])

        report("cleaning", f"Cleaned to {len(df)} usable flights")

        # Start from a clean slate — otherwise aircraft families from a
        # previous, unrelated training run could silently stick around
        # even though they're not part of this training data at all.
        self.performance.reset()
        self.fallback.reset()
        self._duration_models = {}

        validation_results = None
        if test_size > 0 and len(df) > 50:
            train_df, test_df = train_test_split(
                df, test_size=test_size, random_state=random_state,
                stratify=df["aircraft_family"] if "aircraft_family" in df.columns else None,
            )
        else:
            train_df = df
            test_df = None

        # Train per-family performance models
        def family_progress(i: int, total: int, family: str) -> None:
            report("training", f"Training {family} ({i} of {total})")

        report("training", "Starting per-aircraft-type training...")
        profiles = self.performance.train(train_df, progress_callback=family_progress)

        # Build fallback estimator from the same data
        self.fallback.fit(train_df)

        # Learn average speed per family for route-based estimation
        self._learn_speeds(train_df)

        report("validating", "Validating against holdout flights...")

        self._is_trained = True
        self._trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._trained_from = source_names if source_names else [Path(p).name for p in paths]

        summary = {
            "families_trained": list(profiles.keys()),
            "total_flights": len(df),
            "training_flights": len(train_df),
            "test_flights": len(test_df) if test_df is not None else 0,
            "source_breakdown": source_breakdown,
            "per_family": {
                name: {
                    "rmse_cv_kg": p.rmse_cv,
                    "n_training_flights": p.n_training_flights,
                }
                for name, p in profiles.items()
            },
            "data_summary": self.features.summary(train_df),
        }

        # Run holdout validation
        if test_df is not None and len(test_df) > 0:
            from openburn.validation.compare import ValidationReport
            estimated = self.batch_estimate(test_df)
            vr = ValidationReport()
            try:
                overall = vr.validate(estimated)
                by_family = vr.validate_by_family(estimated)
                summary["holdout_validation"] = {
                    "overall": {
                        "mae_kg": overall.mae_kg,
                        "rmse_kg": overall.rmse_kg,
                        "mape_pct": overall.mape_pct,
                        "r_squared": overall.r_squared,
                        "bias_kg": overall.bias_kg,
                        "n_flights": overall.n_flights,
                    },
                    "per_family": {
                        fam: {
                            "mae_kg": m.mae_kg,
                            "rmse_kg": m.rmse_kg,
                            "mape_pct": m.mape_pct,
                            "r_squared": m.r_squared,
                            "n_flights": m.n_flights,
                        }
                        for fam, m in by_family.items()
                    },
                    "scatter": _scatter_points(estimated),
                }
            except ValueError:
                pass

        self._training_summary = summary
        return summary

    def estimate(
        self,
        aircraft_type: str,
        distance_km: float,
        flight_duration_min: Optional[float] = None,
        cruise_altitude_ft: Optional[float] = None,
        average_speed_kts: Optional[float] = None,
        payload_kg: Optional[float] = None,
        **extra_features,
    ) -> dict:
        """Estimate fuel burn for a flight.

        Args:
            aircraft_type: ICAO type code (e.g. "B738")
            distance_km: Great circle distance in km
            flight_duration_min: Optional flight duration
            cruise_altitude_ft: Optional cruise altitude
            average_speed_kts: Optional average ground speed
            payload_kg: Optional payload weight
            **extra_features: Any additional features the model was trained on

        Returns:
            Dict with fuel_burn_kg, confidence level, and model info.
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train_from_csv() first.")

        features = {"distance_km": distance_km}
        if flight_duration_min is not None:
            features["flight_duration_min"] = flight_duration_min
        if cruise_altitude_ft is not None:
            features["cruise_altitude_ft"] = cruise_altitude_ft
        if average_speed_kts is not None:
            features["average_speed_kts"] = average_speed_kts
        if payload_kg is not None:
            features["payload_kg"] = payload_kg
        features.update(extra_features)

        warning = self._range_warning(aircraft_type, distance_km)

        # Try the trained per-family model first
        result = self.performance.predict(aircraft_type, features)
        if result is not None:
            result["method"] = "trained_model"
            result["confidence"] = "low" if warning else "high"
            if warning:
                result["warning"] = warning
            return result

        # Fall back to general estimator
        fallback_result = self.fallback.estimate(aircraft_type, distance_km)
        fallback_result["method"] = "fallback"
        fallback_result["confidence"] = "low"
        if warning:
            fallback_result["warning"] = warning
        return fallback_result

    def batch_estimate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estimate fuel burn for a batch of flights.

        Uses vectorized prediction per aircraft family for speed.
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train_from_csv() first.")

        df = df.copy()
        df["fuel_burn_estimated_kg"] = np.nan
        df["estimation_method"] = ""
        df["estimation_confidence"] = ""
        df["warning"] = ""

        family_col = "aircraft_family" if "aircraft_family" in df.columns else "aircraft_type"

        for family in df[family_col].unique():
            mask = df[family_col] == family
            family_df = df.loc[mask]
            profile = self.performance.get_profile(str(family))

            if profile is not None:
                # Vectorized prediction for this family
                feature_data = {}
                for feat in profile.feature_names:
                    if feat in family_df.columns:
                        feature_data[feat] = family_df[feat].values
                    else:
                        feature_data[feat] = np.full(len(family_df), np.nan)

                X = pd.DataFrame(feature_data)[profile.feature_names]
                predictions = profile.model.predict(X)
                predictions = np.maximum(0, predictions)

                df.loc[mask, "fuel_burn_estimated_kg"] = np.round(predictions, 2)
                df.loc[mask, "estimation_method"] = "trained_model"
                df.loc[mask, "estimation_confidence"] = "high"
            else:
                # Fallback for unknown families
                for idx in family_df.index:
                    dist = family_df.loc[idx, "distance_km"] if "distance_km" in family_df.columns else 0
                    fb = self.fallback.estimate(str(family), float(dist))
                    df.loc[idx, "fuel_burn_estimated_kg"] = fb["fuel_burn_kg"]
                    df.loc[idx, "estimation_method"] = "fallback"
                    df.loc[idx, "estimation_confidence"] = "low"

            # Flag rows whose distance isn't realistic for this aircraft's
            # known range — the model still predicted something, but it's
            # extrapolating far outside anything it has actually seen, so
            # this runs after (and overrides) the confidence set above.
            ac = self.aircraft_registry.get(str(family))
            if ac is not None and ac.range_km is not None and "distance_km" in family_df.columns:
                exceeds = family_df["distance_km"] > ac.range_km
                if exceeds.any():
                    flagged_idx = family_df.index[exceeds]
                    df.loc[flagged_idx, "estimation_confidence"] = "low"
                    df.loc[flagged_idx, "warning"] = (
                        f"Distance exceeds {family}'s typical range (~{round(ac.range_km)} km)"
                    )

        return df

    def estimate_route(
        self,
        departure: str,
        arrival: str,
        aircraft_type: str,
        passengers: Optional[int] = None,
    ) -> dict:
        """Estimate fuel burn from departure/arrival airports and aircraft type.

        Args:
            departure: IATA airport code (e.g. "MAD", "LHR", "JFK")
            arrival: IATA airport code
            aircraft_type: Aircraft family code (e.g. "A320", "A339")
            passengers: Optional passenger count

        Returns:
            Dict with fuel_burn_kg, distance, duration, emissions, route info.
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train_from_csv() first.")

        dep = departure.strip().upper()
        arr = arrival.strip().upper()

        # Look up airport coordinates
        dep_coords = _AIRPORT_COORDS.get(dep)
        arr_coords = _AIRPORT_COORDS.get(arr)

        if dep_coords is None:
            raise ValueError(f"Unknown airport: {dep}")
        if arr_coords is None:
            raise ValueError(f"Unknown airport: {arr}")

        # Compute great-circle distance
        distance_km = _haversine(dep_coords[0], dep_coords[1], arr_coords[0], arr_coords[1])

        # Estimate flight duration from learned average speeds
        flight_duration_min = self._estimate_duration(aircraft_type, distance_km)

        # Estimate airborne duration (subtract ~20min taxi)
        airborne_duration_min = max(flight_duration_min - 20, flight_duration_min * 0.9)

        # Build features
        features = {
            "distance_km": distance_km,
            "flight_duration_min": flight_duration_min,
            "airborne_duration_min": airborne_duration_min,
            "taxi_time_min": flight_duration_min - airborne_duration_min,
        }
        if passengers is not None:
            features["passengers"] = float(passengers)
            features["payload_kg"] = float(passengers) * 100  # ~100kg per pax

        warning = self._range_warning(aircraft_type, distance_km)

        # Get fuel estimate
        result = self.performance.predict(aircraft_type, features)
        if result is not None:
            result["method"] = "trained_model"
            result["confidence"] = "low" if warning else "high"
        else:
            result = self.fallback.estimate(aircraft_type, distance_km)
            result["method"] = "fallback"
            result["confidence"] = "low"
        if warning:
            result["warning"] = warning

        # Add route info
        from openburn.reporting.emissions import EmissionsReport
        em = EmissionsReport()

        result["route"] = {
            "departure": dep,
            "arrival": arr,
            "distance_km": round(distance_km, 1),
            "estimated_duration_min": round(flight_duration_min, 0),
        }
        result["emissions"] = em.from_fuel_kg(result["fuel_burn_kg"])

        return result

    def _range_warning(self, aircraft_type: str, distance_km: float) -> Optional[str]:
        """Return a warning if distance_km exceeds this aircraft's known
        real-world range (e.g. a narrowbody being asked to fly a route only
        a widebody could manage non-stop) — the trained model will still
        produce a number, but it's extrapolating far outside anything it
        has actually seen, so callers should treat it as unreliable."""
        ac = self.aircraft_registry.get(aircraft_type)
        if ac is None or ac.range_km is None:
            return None
        if distance_km > ac.range_km:
            return (
                f"Distance ({round(distance_km)} km) exceeds {aircraft_type}'s "
                f"typical range (~{round(ac.range_km)} km) — not realistic "
                f"non-stop, so this estimate may be unreliable."
            )
        return None

    def get_airports(self) -> list[str]:
        """Return list of known airport codes."""
        return sorted(_AIRPORT_COORDS.keys())

    def _learn_speeds(self, df: pd.DataFrame) -> None:
        """Learn a per-family duration model: fixed ground/climb/descent
        overhead + a per-km cruise rate, fit from (distance, duration)
        pairs in the training data.

        A single flat average speed badly distorts any family that flies
        both short-haul and long-haul routes: the fixed overhead is a much
        bigger share of a short flight's total time, so mixing the two
        together pulls the "average speed" away from what either regime
        actually looks like. Fitting overhead and cruise rate separately
        (duration = overhead + rate * distance) fixes that.
        """
        if "aircraft_family" not in df.columns:
            return
        if "distance_km" not in df.columns or "flight_duration_min" not in df.columns:
            return

        valid = df[(df["distance_km"] > 100) & (df["flight_duration_min"] > 30)]

        for family, group in valid.groupby("aircraft_family"):
            self._duration_models[str(family)] = _fit_duration_model(group)

        self._duration_models["_global"] = _fit_duration_model(valid)

    def _estimate_duration(self, aircraft_type: str, distance_km: float) -> float:
        """Estimate flight duration in minutes from distance and aircraft type."""
        overhead_min, min_per_km = self._duration_models.get(
            aircraft_type, self._duration_models.get("_global", (25.0, 1.0 / 8.0))
        )
        return overhead_min + min_per_km * distance_km

    def save_model(self, path: str | Path) -> None:
        """Save the full trained model (profiles, fallback estimator, learned
        speeds) to disk so the server doesn't need to retrain on restart."""
        if not self._is_trained:
            raise RuntimeError("Model not trained — nothing to save.")
        state = {
            "performance": self.performance.get_state(),
            "fallback": self.fallback.get_state(),
            "duration_models": self._duration_models,
            "training_summary": self._training_summary,
            "trained_at": self._trained_at,
            "trained_from": self._trained_from,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(state, path)

    def load_model(self, path: str | Path) -> bool:
        """Load a previously saved model from disk. Returns True if a saved
        model was found and loaded, False if the path doesn't exist yet or
        the saved file is missing/corrupted/from an incompatible version
        (in which case the server just starts untrained instead of crashing)."""
        path = Path(path)
        if not path.exists():
            return False
        try:
            state = joblib.load(path)
            self.performance.set_state(state["performance"])
            self.fallback.set_state(state["fallback"])
            self._duration_models = state.get("duration_models", {})
            self._training_summary = state.get("training_summary", {})
            self._trained_at = state.get("trained_at")
            self._trained_from = state.get("trained_from")
        except Exception as e:
            print(f"Could not load saved model at {path} ({e}) — starting untrained.")
            return False
        self._is_trained = True
        return True

    def save_summary(self, path: str) -> None:
        """Save model summary to JSON."""
        summary = {
            "families": self.performance.families,
            "profiles": {},
        }
        for family in self.performance.families:
            profile = self.performance.get_profile(family)
            if profile:
                summary["profiles"][family] = {
                    "rmse_cv_kg": profile.rmse_cv,
                    "n_training_flights": profile.n_training_flights,
                    "features": profile.feature_names,
                }
        if self._training_summary.get("holdout_validation"):
            summary["holdout_validation"] = self._training_summary["holdout_validation"]
        Path(path).write_text(json.dumps(summary, indent=2))


def _fit_duration_model(group: pd.DataFrame) -> tuple[float, float]:
    """Fit duration_min = overhead + rate * distance_km via linear
    regression, so short-haul and long-haul flights in the same family
    don't drag a single flat speed away from both.

    Falls back to a flat median speed (with a fixed overhead guess) if
    there's too little data to fit, or the fit comes out physically
    implausible (overhead should be non-negative; rate should correspond
    to a believable average speed of roughly 400-3000 km/h).
    """
    if len(group) >= 10:
        rate, overhead = np.polyfit(group["distance_km"], group["flight_duration_min"], 1)
        if 0.02 <= rate <= 0.15 and 0 <= overhead <= 120:
            return (float(overhead), float(rate))

    speed = (group["distance_km"] / group["flight_duration_min"]).median()
    return (25.0, 1.0 / speed) if speed > 0 else (25.0, 1.0 / 8.0)


def _scatter_points(df: pd.DataFrame, max_points: int = 2500, random_state: int = 42) -> list[dict]:
    """Sampled (actual, predicted) pairs from a validated holdout DataFrame,
    for the predicted-vs-actual chart on the Model Accuracy tab. Capped so
    the JSON response stays small even for a large holdout set."""
    cols = ["aircraft_family", "fuel_burn_kg", "fuel_burn_estimated_kg"]
    valid = df[cols].dropna()
    if len(valid) > max_points:
        valid = valid.sample(n=max_points, random_state=random_state)
    return [
        {
            "family": str(row.aircraft_family),
            "actual_kg": round(float(row.fuel_burn_kg), 1),
            "predicted_kg": round(float(row.fuel_burn_estimated_kg), 1),
        }
        for row in valid.itertuples()
    ]
