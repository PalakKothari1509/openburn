"""Data-driven aircraft performance model.

Instead of BADA's physics-based coefficients, we learn performance
profiles from historical fuel burn data grouped by aircraft family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance


@dataclass
class PerformanceProfile:
    """Learned performance profile for an aircraft family."""
    aircraft_family: str
    model: HistGradientBoostingRegressor
    feature_names: list[str]
    rmse_cv: float  # Cross-validated RMSE (kg)
    n_training_flights: int
    feature_importance: dict[str, float]


class PerformanceModel:
    """Learns fuel burn profiles per aircraft family from historical data.

    This is OpenBurn's core alternative to BADA. Rather than using
    manufacturer-provided aerodynamic coefficients, we train models
    on actual operational data.
    """

    def __init__(self) -> None:
        self._profiles: dict[str, PerformanceProfile] = {}

    def train(self, df: pd.DataFrame, aircraft_family_col: str = "aircraft_family") -> dict:
        """Train fuel burn models per aircraft family.

        Args:
            df: Training data with columns for features and a 'fuel_burn_kg' target.
                Required columns: fuel_burn_kg, distance_km, aircraft_family
                Optional columns: flight_duration_min, altitude_ft, speed_kts,
                    payload_kg, headwind_kts, temperature_c
            aircraft_family_col: Column containing aircraft family grouping.

        Returns:
            Dict of {family: PerformanceProfile} for all trained models.
        """
        required = {"fuel_burn_kg", "distance_km", aircraft_family_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        feature_cols = _select_feature_cols(df)
        results = {}

        for family, group in df.groupby(aircraft_family_col):
            if len(group) < 10:
                continue  # Need minimum data to train

            X = group[feature_cols].copy()
            y = group["fuel_burn_kg"].copy()

            # Drop rows with NaN in features or target
            valid = X.notna().all(axis=1) & y.notna()
            X = X[valid]
            y = y[valid]

            if len(X) < 10:
                continue

            model = HistGradientBoostingRegressor(
                max_iter=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            )

            # Cross-validated RMSE
            scores = cross_val_score(
                model, X, y, cv=min(5, len(X)), scoring="neg_root_mean_squared_error"
            )
            rmse = -scores.mean()

            # Train final model on all data
            model.fit(X, y)

            # HistGradientBoostingRegressor has no feature_importances_ attribute,
            # so use permutation importance instead (works on any fitted model).
            perm = permutation_importance(
                model, X, y, n_repeats=5, random_state=42,
                scoring="neg_root_mean_squared_error",
            )
            importance = dict(zip(feature_cols, [round(float(v), 4) for v in perm.importances_mean]))

            profile = PerformanceProfile(
                aircraft_family=str(family),
                model=model,
                feature_names=feature_cols,
                rmse_cv=round(rmse, 2),
                n_training_flights=len(X),
                feature_importance=importance,
            )
            self._profiles[str(family)] = profile
            results[str(family)] = profile

        return results

    def predict(
        self, aircraft_family: str, features: dict[str, float]
    ) -> Optional[dict]:
        """Predict fuel burn for a single flight.

        Args:
            aircraft_family: e.g. "B738", "A320"
            features: Dict of feature values (must include at least distance_km)

        Returns:
            Dict with fuel_burn_kg estimate and model confidence, or None if
            no model exists for this aircraft family.
        """
        profile = self._profiles.get(aircraft_family)
        if profile is None:
            return None

        # Only use features that are both available and expected by the model
        available_features = [f for f in profile.feature_names if f in features]
        if not available_features:
            return None

        row = {f: features.get(f, np.nan) for f in profile.feature_names}
        X = pd.DataFrame([row])[profile.feature_names]
        prediction = profile.model.predict(X)[0]

        return {
            "fuel_burn_kg": round(max(0, prediction), 2),
            "aircraft_family": aircraft_family,
            "model_rmse_kg": profile.rmse_cv,
            "n_training_flights": profile.n_training_flights,
        }

    def get_profile(self, aircraft_family: str) -> Optional[PerformanceProfile]:
        return self._profiles.get(aircraft_family)

    @property
    def families(self) -> list[str]:
        return sorted(self._profiles.keys())

    def feature_importance(self, aircraft_family: str) -> Optional[dict[str, float]]:
        """Return feature importances for a given aircraft family model."""
        profile = self._profiles.get(aircraft_family)
        if profile is None:
            return None
        return profile.feature_importance

    def get_state(self) -> dict:
        """Return picklable state for model persistence."""
        return {"profiles": self._profiles}

    def set_state(self, state: dict) -> None:
        """Restore state saved by get_state()."""
        self._profiles = state["profiles"]


def _select_feature_cols(df: pd.DataFrame) -> list[str]:
    """Select available feature columns from the dataframe."""
    potential_features = [
        "distance_km",
        "flight_duration_min",
        "airborne_duration_min",
        "taxi_time_min",
        "cruise_altitude_ft",
        "average_speed_kts",
        "payload_kg",
        "headwind_kts",
        "temperature_c",
        "takeoff_weight_kg",
        "passengers",
        "cargo_kg",
    ]
    return [c for c in potential_features if c in df.columns]
