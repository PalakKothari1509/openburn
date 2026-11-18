"""Data-driven aircraft performance model.

Instead of BADA's physics-based coefficients, we learn performance
profiles from historical fuel burn data grouped by aircraft family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.inspection import permutation_importance

# Features that aren't recorded by every airline/file (e.g. fp_trip_fuel,
# the airline's own flight-planned fuel figure) — kept as a feature only
# for families that have at least some coverage, and allowed to be NaN
# per-row elsewhere; HistGradientBoostingRegressor handles missing values
# in a column natively, so this needs no imputation.
OPTIONAL_FEATURES = {"fp_trip_fuel"}


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

    def train(
        self,
        df: pd.DataFrame,
        aircraft_family_col: str = "aircraft_family",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> dict:
        """Train fuel burn models per aircraft family.

        Args:
            df: Training data with columns for features and a 'fuel_burn_kg' target.
                Required columns: fuel_burn_kg, distance_km, aircraft_family
                Optional columns: flight_duration_min, altitude_ft, speed_kts,
                    payload_kg, headwind_kts, temperature_c
            aircraft_family_col: Column containing aircraft family grouping.
            progress_callback: Optional callback(index, total, family_name),
                called before training each family — lets a caller (the API)
                report "training X (3 of 20)" instead of an opaque wait, since
                each family can take a while (cross-validation + permutation
                importance on top of the fit itself).

        Returns:
            Dict of {family: PerformanceProfile} for all trained models.
        """
        required = {"fuel_burn_kg", "distance_km", aircraft_family_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        feature_cols = _select_feature_cols(df)
        results = {}

        groups = list(df.groupby(aircraft_family_col))
        for i, (family, group) in enumerate(groups, start=1):
            if progress_callback:
                progress_callback(i, len(groups), str(family))
            if len(group) < 10:
                continue  # Need minimum data to train

            # Optional features (e.g. fp_trip_fuel) aren't recorded by every
            # airline/file, so a family with zero coverage for one would
            # otherwise crash HistGradientBoostingRegressor's binning step
            # on an all-missing column. Drop it for this family only; keep
            # it for families that do have it, since HGB handles per-row
            # NaNs in a kept column natively (no imputation needed).
            family_feature_cols = [
                c for c in feature_cols
                if c not in OPTIONAL_FEATURES or group[c].notna().any()
            ]
            required_cols = [c for c in family_feature_cols if c not in OPTIONAL_FEATURES]

            X = group[family_feature_cols].copy()
            y = group["fuel_burn_kg"].copy()

            # Drop rows missing a required feature or the target; optional
            # features may be NaN per-row.
            valid = X[required_cols].notna().all(axis=1) & y.notna()
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

            # Cross-validated RMSE. Shuffled folds matter here: rows are
            # concatenated file-by-file before grouping, so an unshuffled
            # split could hand an optional feature's fold all-NaN if, say,
            # one source file never recorded it and happens to sort into
            # its own fold.
            cv = KFold(n_splits=min(5, len(X)), shuffle=True, random_state=42)
            scores = cross_val_score(
                model, X, y, cv=cv, scoring="neg_root_mean_squared_error"
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
            importance = dict(zip(family_feature_cols, [round(float(v), 4) for v in perm.importances_mean]))

            profile = PerformanceProfile(
                aircraft_family=str(family),
                model=model,
                feature_names=family_feature_cols,
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

    def reset(self) -> None:
        """Clear all learned profiles — call before a fresh training run
        so aircraft families from a previous, unrelated run can't
        silently stick around."""
        self._profiles = {}

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
        "fp_trip_fuel",
    ]
    return [c for c in potential_features if c in df.columns]
