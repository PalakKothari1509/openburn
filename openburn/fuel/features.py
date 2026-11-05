"""Feature engineering — extract ML features from flight/trajectory data.

Handles both generic fuel CSVs and airline operational formats
(OFFBLOCK_FUEL/ONBLOCK_FUEL, block times, etc.)
"""

from __future__ import annotations

from math import radians, sin, cos, asin, sqrt

import numpy as np
import pandas as pd


# IATA/ICAO airport coordinates for great-circle distance calculation
# This is a subset — extend as needed or load from a full database
_AIRPORT_COORDS: dict[str, tuple[float, float]] = {}


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two points."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))


def load_airport_coords(path: str) -> None:
    """Load airport coordinates CSV (columns: iata or icao, lat, lon)."""
    df = pd.read_csv(path)
    cols = [c.lower().strip() for c in df.columns]
    df.columns = cols
    code_col = "iata" if "iata" in cols else "icao" if "icao" in cols else cols[0]
    for _, row in df.iterrows():
        _AIRPORT_COORDS[str(row[code_col]).strip().upper()] = (
            float(row["lat"]), float(row["lon"])
        )


class FeatureExtractor:
    """Extract features from raw flight data for the fuel burn model.

    Handles two data formats:
    1. Pre-computed: CSV already has fuel_burn_kg, distance_km columns
    2. Airline operational: CSV has OFFBLOCK_FUEL, ONBLOCK_FUEL, block times, airports
       — fuel burn and duration are computed from these
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
            "aircraft_code", "type_code", "ac_subtype",
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
            "passengers", "pax", "passenger_count", "num_passengers", "pax_total",
        ],
        "cargo_kg": [
            "cargo_kg", "cargo", "freight_kg", "fret_kg",
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
            "dep_ap_actual",
        ],
        "destination": [
            "destination", "arrival", "arr_icao", "dest_icao", "arr",
            "arr_ap_actual",
        ],
    }

    # Columns that map directly from airline operational format
    OPERATIONAL_COLUMNS = {
        "offblock_fuel": ["offblock_fuel"],
        "onblock_fuel": ["onblock_fuel"],
        "offblock_time": ["offblock"],
        "onblock_time": ["onblock"],
        "airborne_time": ["airborne"],
        "landing_time": ["landing"],
        "flight_number": ["flight_number"],
        "ac_registration": ["ac_registration"],
        "mail_kg": ["mail_kg"],
        "fp_trip_fuel": ["fp_trip_fuel"],
        "fuel_density": ["density"],
        "fuel_type": ["fuel_type"],
        "fau_kg": ["fau_kg"],
    }

    def load_fuel_csv(self, path: str, **kwargs) -> pd.DataFrame:
        """Load a CSV of flight fuel records and normalise column names."""
        df = pd.read_csv(path, **kwargs)
        df = self._normalize_columns(df)
        df = self._compute_derived_fields(df)
        return df

    def prepare_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data for model training."""
        df = df.copy()

        # Convert flight level to feet if cruise_altitude_ft looks like FL
        if "cruise_altitude_ft" in df.columns:
            fl_mask = df["cruise_altitude_ft"] < 600
            df.loc[fl_mask, "cruise_altitude_ft"] = df.loc[fl_mask, "cruise_altitude_ft"] * 100

        # Must have fuel and distance
        df = df.dropna(subset=["fuel_burn_kg", "distance_km"])
        df = df[df["fuel_burn_kg"] > 0]
        df = df[df["distance_km"] > 0]

        # Must have an aircraft family for per-type training
        if "aircraft_family" in df.columns:
            df = df.dropna(subset=["aircraft_family"])
            df = df[df["aircraft_family"].str.strip() != ""]

        # Compute fuel efficiency as a sanity check
        df["fuel_per_km"] = df["fuel_burn_kg"] / df["distance_km"]

        # Remove extreme outliers (> 3 std from mean per aircraft family)
        if "aircraft_family" in df.columns:
            keep = []
            for _, group in df.groupby("aircraft_family"):
                keep.append(_remove_outliers(group))
            df = pd.concat(keep, ignore_index=True) if keep else df
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
            stats = df["fuel_burn_kg"].describe()
            info["fuel_burn_kg_stats"] = {
                "mean": round(stats["mean"], 1),
                "median": round(df["fuel_burn_kg"].median(), 1),
                "min": round(stats["min"], 1),
                "max": round(stats["max"], 1),
                "std": round(stats["std"], 1),
            }
        if "distance_km" in df.columns:
            valid_dist = df["distance_km"].dropna()
            if len(valid_dist) > 0:
                info["distance_km_stats"] = {
                    "mean": round(valid_dist.mean(), 1),
                    "median": round(valid_dist.median(), 1),
                    "min": round(valid_dist.min(), 1),
                    "max": round(valid_dist.max(), 1),
                }
        if "flight_duration_min" in df.columns:
            valid_dur = df["flight_duration_min"].dropna()
            if len(valid_dur) > 0:
                info["flight_duration_min_stats"] = {
                    "mean": round(valid_dur.mean(), 1),
                    "median": round(valid_dur.median(), 1),
                    "min": round(valid_dur.min(), 1),
                    "max": round(valid_dur.max(), 1),
                }
        return info

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map column names to standard names."""
        lower_map = {c.lower().strip().replace(" ", "_"): c for c in df.columns}
        col_rename = {}

        # Standard aliases
        for standard, aliases in self.COLUMN_ALIASES.items():
            for alias in aliases:
                if alias in lower_map and lower_map[alias] not in col_rename:
                    col_rename[lower_map[alias]] = standard
                    break

        # Operational columns
        for standard, aliases in self.OPERATIONAL_COLUMNS.items():
            for alias in aliases:
                if alias in lower_map and lower_map[alias] not in col_rename:
                    col_rename[lower_map[alias]] = standard
                    break

        return df.rename(columns=col_rename)

    def _compute_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute fuel_burn_kg, flight_duration_min, distance_km from operational data."""
        df = df.copy()

        # Fuel burn = offblock fuel - onblock fuel
        if "fuel_burn_kg" not in df.columns:
            if "offblock_fuel" in df.columns and "onblock_fuel" in df.columns:
                df["offblock_fuel"] = pd.to_numeric(df["offblock_fuel"], errors="coerce")
                df["onblock_fuel"] = pd.to_numeric(df["onblock_fuel"], errors="coerce")
                df["fuel_burn_kg"] = df["offblock_fuel"] - df["onblock_fuel"]

        # Block time duration
        if "flight_duration_min" not in df.columns:
            if "offblock_time" in df.columns and "onblock_time" in df.columns:
                offblock = pd.to_datetime(df["offblock_time"], dayfirst=True, errors="coerce")
                onblock = pd.to_datetime(df["onblock_time"], dayfirst=True, errors="coerce")
                df["flight_duration_min"] = (onblock - offblock).dt.total_seconds() / 60

        # Airborne time (more accurate for fuel modelling)
        if "airborne_time" in df.columns and "landing_time" in df.columns:
            airborne = pd.to_datetime(df["airborne_time"], dayfirst=True, errors="coerce")
            landing = pd.to_datetime(df["landing_time"], dayfirst=True, errors="coerce")
            df["airborne_duration_min"] = (landing - airborne).dt.total_seconds() / 60

        # Taxi time
        if "flight_duration_min" in df.columns and "airborne_duration_min" in df.columns:
            df["taxi_time_min"] = df["flight_duration_min"] - df["airborne_duration_min"]

        # Total payload
        if "cargo_kg" in df.columns:
            cargo = pd.to_numeric(df["cargo_kg"], errors="coerce").fillna(0)
        else:
            cargo = 0
        if "mail_kg" in df.columns:
            mail = pd.to_numeric(df["mail_kg"], errors="coerce").fillna(0)
        else:
            mail = 0
        if "passengers" in df.columns:
            pax = pd.to_numeric(df["passengers"], errors="coerce").fillna(0)
            # Estimate passenger weight at ~100kg (person + luggage)
            df["payload_kg"] = (pax * 100) + cargo + mail

        # Flight plan trip fuel (for comparison)
        if "fp_trip_fuel" in df.columns:
            df["fp_trip_fuel"] = pd.to_numeric(df["fp_trip_fuel"], errors="coerce")

        # Distance from airport pairs (great circle)
        if "distance_km" not in df.columns and "origin" in df.columns and "destination" in df.columns:
            if _AIRPORT_COORDS:
                df["distance_km"] = df.apply(
                    lambda r: _airport_distance(r.get("origin"), r.get("destination")),
                    axis=1,
                )
            else:
                # Use airborne duration + typical speed as rough distance proxy
                if "airborne_duration_min" in df.columns:
                    # ~800 km/h average for jets, convert min to hours
                    df["distance_km"] = df["airborne_duration_min"] / 60 * 800

        return df


def _airport_distance(origin: str, destination: str) -> float:
    """Compute great-circle distance between two airports."""
    if not origin or not destination:
        return np.nan
    orig = _AIRPORT_COORDS.get(str(origin).strip().upper())
    dest = _AIRPORT_COORDS.get(str(destination).strip().upper())
    if orig and dest:
        return _haversine(orig[0], orig[1], dest[0], dest[1])
    return np.nan


def _remove_outliers(group: pd.DataFrame, n_std: float = 3) -> pd.DataFrame:
    """Remove fuel efficiency outliers beyond n standard deviations."""
    if "fuel_per_km" not in group.columns or len(group) < 5:
        return group
    mean = group["fuel_per_km"].mean()
    std = group["fuel_per_km"].std()
    if std == 0:
        return group
    return group[(group["fuel_per_km"] - mean).abs() <= n_std * std]
