"""Core fuel burn estimation model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from openburn.aircraft.performance import PerformanceModel
from openburn.fuel.features import FeatureExtractor, load_airport_coords
from openburn.fuel.fallback import FallbackEstimator

# Default airport coords file (relative to package)
_DEFAULT_AIRPORT_COORDS = Path(__file__).resolve().parent.parent.parent / "data" / "airport_coords.csv"


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

        # Load airport coordinates for distance calculation
        coords_path = Path(airport_coords_path) if airport_coords_path else _DEFAULT_AIRPORT_COORDS
        if coords_path.exists():
            load_airport_coords(str(coords_path))

    def train_from_csv(self, fuel_csv_path: str, **kwargs) -> dict:
        """Train the model from a CSV of historical fuel burn data.

        Args:
            fuel_csv_path: Path to CSV with flight records and fuel burn.

        Returns:
            Training summary with per-family metrics.
        """
        df = self.features.load_fuel_csv(fuel_csv_path, **kwargs)
        df = self.features.prepare_training_data(df)

        # Train per-family performance models
        profiles = self.performance.train(df)

        # Build fallback estimator from the same data
        self.fallback.fit(df)

        self._is_trained = True

        return {
            "families_trained": list(profiles.keys()),
            "total_flights_used": len(df),
            "per_family": {
                name: {
                    "rmse_kg": p.rmse_cv,
                    "n_flights": p.n_training_flights,
                }
                for name, p in profiles.items()
            },
            "data_summary": self.features.summary(df),
        }

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

        # Try the trained per-family model first
        result = self.performance.predict(aircraft_type, features)
        if result is not None:
            result["method"] = "trained_model"
            result["confidence"] = "high"
            return result

        # Fall back to general estimator
        fallback_result = self.fallback.estimate(aircraft_type, distance_km)
        fallback_result["method"] = "fallback"
        fallback_result["confidence"] = "low"
        return fallback_result

    def batch_estimate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estimate fuel burn for a batch of flights.

        Args:
            df: DataFrame with at least 'aircraft_family' and 'distance_km' columns.

        Returns:
            Input dataframe with added fuel_burn_estimated_kg and estimation_method columns.
        """
        results = []
        for _, row in df.iterrows():
            features = {
                k: row[k] for k in row.index
                if k not in ("aircraft_family", "fuel_burn_kg") and pd.notna(row[k])
            }
            ac_type = row.get("aircraft_family", row.get("aircraft_type", "UNKNOWN"))
            est = self.estimate(aircraft_type=str(ac_type), **features)
            results.append(est)

        result_df = pd.DataFrame(results)
        df = df.copy()
        df["fuel_burn_estimated_kg"] = result_df["fuel_burn_kg"].values
        df["estimation_method"] = result_df["method"].values
        df["estimation_confidence"] = result_df["confidence"].values
        return df

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
        Path(path).write_text(json.dumps(summary, indent=2))
