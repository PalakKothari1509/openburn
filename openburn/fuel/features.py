"""Feature engineering — extract ML features from flight/trajectory data."""

from __future__ import annotations

import pandas as pd


class FeatureExtractor:
    """Extract features from raw flight data for the fuel burn model.

    Takes a CSV of flight records (with fuel burn) and produces
    a clean feature matrix ready for model training.
    """

    # Column name mappings: standard_name -> common CSV variations
    COLUMN_ALIASES = {
        "fuel_burn_kg": [
            "fuel_burn_kg", "fuel_kg", "fuel_burn", "fuel_used_kg",
            "total_fuel_kg", "fuel_consumption_kg", "burn_kg",
        ],
        "distance_km": [
            "distance_km", "distance", "dist_km", "great_circle_distance_km",
            "gc_distance_km", "route_distance_km",
        ],
        "flight_duration_min": [
            "flight_duration_min", "duration_min", "flight_time_min",
            "block_time_min", "duration", "flight_duration",
        ],
        "aircraft_family": [
            "aircraft_family", "aircraft_type", "icao_type", "actype",
            "aircraft_code", "type_code",
        ],
        "cruise_altitude_ft": [
            "cruise_altitude_ft", "cruise_alt", "altitude_ft", "cruise_fl",
            "flight_level",
        ],
        "average_speed_kts": [
            "average_speed_kts", "avg_speed", "ground_speed_kts", "speed_kts",
        ],
        "payload_kg": [
            "payload_kg", "payload", "load_kg", "total_payload_kg",
        ],
        "passengers": [
            "passengers", "pax", "passenger_count", "num_passengers",
        ],
        "cargo_kg": [
            "cargo_kg", "cargo", "freight_kg",
        ],
        "takeoff_weight_kg": [
            "takeoff_weight_kg", "tow_kg", "takeoff_mass_kg", "gross_weight_kg",
        ],
        "headwind_kts": [
            "headwind_kts", "headwind", "wind_component_kts",
        ],
        "temperature_c": [
            "temperature_c", "temp_c", "oat_c", "sat_c",
        ],
        "origin": [
            "origin", "departure", "dep_icao", "origin_icao", "dep",
        ],
        "destination": [
            "destination", "arrival", "arr_icao", "dest_icao", "arr",
        ],
    }

    def load_fuel_csv(self, path: str, **kwargs) -> pd.DataFrame:
        """Load a CSV of flight fuel records and normalise column names."""
        df = pd.read_csv(path, **kwargs)
        df = self._normalize_columns(df)
        return df

    def prepare_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data for model training.

        - Drops rows missing fuel_burn_kg or distance_km
        - Removes obvious outliers (negative fuel, zero distance)
        - Converts flight levels to feet if needed
        """
        df = df.copy()

        # Convert flight level to feet if cruise_altitude_ft looks like FL
        if "cruise_altitude_ft" in df.columns:
            fl_mask = df["cruise_altitude_ft"] < 600  # Likely flight level, not feet
            df.loc[fl_mask, "cruise_altitude_ft"] = df.loc[fl_mask, "cruise_altitude_ft"] * 100

        # Basic filters
        df = df[df["fuel_burn_kg"] > 0]
        df = df[df["distance_km"] > 0]

        # Compute fuel efficiency as a sanity check
        df["fuel_per_km"] = df["fuel_burn_kg"] / df["distance_km"]

        # Remove extreme outliers (> 3 std from mean per aircraft family)
        if "aircraft_family" in df.columns:
            df = df.groupby("aircraft_family", group_keys=False).apply(_remove_outliers)
        else:
            df = _remove_outliers(df)

        df = df.drop(columns=["fuel_per_km"])
        return df.reset_index(drop=True)

    def summary(self, df: pd.DataFrame) -> dict:
        """Summary statistics of the fuel dataset."""
        info = {
            "total_flights": len(df),
            "columns": list(df.columns),
        }
        if "aircraft_family" in df.columns:
            family_counts = df["aircraft_family"].value_counts().to_dict()
            info["flights_per_family"] = family_counts
            info["n_aircraft_families"] = len(family_counts)
        if "fuel_burn_kg" in df.columns:
            info["fuel_burn_kg_stats"] = {
                "mean": round(df["fuel_burn_kg"].mean(), 1),
                "median": round(df["fuel_burn_kg"].median(), 1),
                "min": round(df["fuel_burn_kg"].min(), 1),
                "max": round(df["fuel_burn_kg"].max(), 1),
            }
        if "distance_km" in df.columns:
            info["distance_km_stats"] = {
                "mean": round(df["distance_km"].mean(), 1),
                "median": round(df["distance_km"].median(), 1),
                "min": round(df["distance_km"].min(), 1),
                "max": round(df["distance_km"].max(), 1),
            }
        return info

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        lower_map = {c.lower().strip().replace(" ", "_"): c for c in df.columns}
        col_rename = {}
        for standard, aliases in self.COLUMN_ALIASES.items():
            for alias in aliases:
                if alias in lower_map:
                    col_rename[lower_map[alias]] = standard
                    break
        return df.rename(columns=col_rename)


def _remove_outliers(group: pd.DataFrame, n_std: float = 3) -> pd.DataFrame:
    """Remove fuel efficiency outliers beyond n standard deviations."""
    if "fuel_per_km" not in group.columns or len(group) < 5:
        return group
    mean = group["fuel_per_km"].mean()
    std = group["fuel_per_km"].std()
    if std == 0:
        return group
    return group[(group["fuel_per_km"] - mean).abs() <= n_std * std]
