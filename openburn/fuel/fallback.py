"""Fallback fuel estimation when per-aircraft-family model is unavailable.

Uses simple distance-based regression grouped by broad aircraft category
(widebody, narrowbody, turboprop) as a last-resort estimator.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# Broad category mapping based on common ICAO type codes
AIRCRAFT_CATEGORIES = {
    "widebody": [
        "A332", "A333", "A338", "A339", "A342", "A343", "A345", "A346",
        "A359", "A35K", "A388", "B744", "B748", "B762", "B763", "B764",
        "B772", "B773", "B77L", "B77W", "B788", "B789", "B78X",
    ],
    "narrowbody": [
        "A319", "A320", "A20N", "A321", "A21N", "B712", "B732", "B733",
        "B734", "B735", "B736", "B737", "B738", "B739", "B38M", "B39M",
        "B752", "B753", "E170", "E175", "E190", "E195", "E290", "E295",
        "CRJ7", "CRJ9", "CRJX",
    ],
    "turboprop": [
        "AT43", "AT45", "AT72", "AT76", "DH8A", "DH8B", "DH8C", "DH8D",
        "SF34", "JS41", "BE20", "SW4",
    ],
}


def _classify_aircraft(icao_code: str) -> str:
    """Classify aircraft into broad category."""
    code = icao_code.strip().upper()
    for category, codes in AIRCRAFT_CATEGORIES.items():
        if code in codes:
            return category
    # Heuristic: if code starts with A3x or B7x → probably narrowbody/widebody
    if code.startswith(("A3", "B73", "B38", "B39", "E1", "E2", "CR")):
        return "narrowbody"
    if code.startswith(("A3", "B74", "B76", "B77", "B78")):
        return "widebody"
    return "narrowbody"  # Default assumption


class FallbackEstimator:
    """Simple distance-based fuel estimator as a fallback.

    Uses average fuel-per-km rates computed from training data,
    grouped by broad aircraft category.
    """

    def __init__(self) -> None:
        self._rates: dict[str, dict] = {}
        self._global_rate: Optional[float] = None

    def fit(self, df: pd.DataFrame) -> None:
        """Learn fuel-per-km rates from historical data."""
        if "fuel_burn_kg" not in df.columns or "distance_km" not in df.columns:
            raise ValueError("Need fuel_burn_kg and distance_km columns")

        df = df[df["distance_km"] > 0].copy()
        df["fuel_per_km"] = df["fuel_burn_kg"] / df["distance_km"]

        # Global average
        self._global_rate = float(df["fuel_per_km"].median())

        # Per aircraft family if available
        if "aircraft_family" in df.columns:
            df["_category"] = df["aircraft_family"].apply(_classify_aircraft)
            for cat, group in df.groupby("_category"):
                self._rates[str(cat)] = {
                    "fuel_per_km_median": float(group["fuel_per_km"].median()),
                    "fuel_per_km_mean": float(group["fuel_per_km"].mean()),
                    "n_flights": len(group),
                }

    def get_state(self) -> dict:
        """Return picklable state for model persistence."""
        return {"rates": self._rates, "global_rate": self._global_rate}

    def set_state(self, state: dict) -> None:
        """Restore state saved by get_state()."""
        self._rates = state["rates"]
        self._global_rate = state["global_rate"]

    def estimate(self, aircraft_type: str, distance_km: float) -> dict:
        """Estimate fuel burn using the fallback model."""
        category = _classify_aircraft(aircraft_type)
        rate_info = self._rates.get(category)

        if rate_info and rate_info["n_flights"] >= 5:
            fuel_kg = rate_info["fuel_per_km_median"] * distance_km
            source = f"fallback_{category}"
        elif self._global_rate:
            fuel_kg = self._global_rate * distance_km
            source = "fallback_global"
        else:
            # Last resort: rough industry averages (kg fuel per km)
            defaults = {"widebody": 8.0, "narrowbody": 3.5, "turboprop": 1.5}
            fuel_kg = defaults.get(category, 3.5) * distance_km
            source = "fallback_default"

        return {
            "fuel_burn_kg": round(max(0, fuel_kg), 2),
            "aircraft_family": aircraft_type,
            "category": category,
            "estimation_source": source,
        }
