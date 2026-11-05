"""Core fuel burn estimation model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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
        self._training_summary: dict = {}

        # Load airport coordinates for distance calculation
        coords_path = Path(airport_coords_path) if airport_coords_path else _DEFAULT_AIRPORT_COORDS
        if coords_path.exists():
            load_airport_coords(str(coords_path))

    def train_from_csv(
        self,
        fuel_csv_path: str,
        test_size: float = 0.2,
        random_state: int = 42,
        **kwargs,
    ) -> dict:
        """Train the model from a CSV of historical fuel burn data.

        Args:
            fuel_csv_path: Path to CSV with flight records and fuel burn.
            test_size: Fraction of data to hold out for validation (0-1).
                Set to 0 to train on all data (no validation).
            random_state: Random seed for reproducibility.

        Returns:
            Training summary with per-family metrics and validation results.
        """
        df = self.features.load_fuel_csv(fuel_csv_path, **kwargs)
        df = self.features.prepare_training_data(df)

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
        profiles = self.performance.train(train_df)

        # Build fallback estimator from the same data
        self.fallback.fit(train_df)

        self._is_trained = True

        summary = {
            "families_trained": list(profiles.keys()),
            "total_flights": len(df),
            "training_flights": len(train_df),
            "test_flights": len(test_df) if test_df is not None else 0,
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

        Uses vectorized prediction per aircraft family for speed.
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train_from_csv() first.")

        df = df.copy()
        df["fuel_burn_estimated_kg"] = np.nan
        df["estimation_method"] = ""
        df["estimation_confidence"] = ""

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
        if self._training_summary.get("holdout_validation"):
            summary["holdout_validation"] = self._training_summary["holdout_validation"]
        Path(path).write_text(json.dumps(summary, indent=2))
