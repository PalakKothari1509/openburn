"""ADS-B data ingestion and cleaning."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class ADSBProcessor:
    """Ingest, clean, and standardise ADS-B position reports."""

    REQUIRED_COLS = {"timestamp", "latitude", "longitude", "altitude_ft"}
    OPTIONAL_COLS = {"speed_kts", "heading", "vertical_rate_fpm", "icao24", "callsign", "flight_id"}

    def load_csv(self, path: str | Path, **kwargs) -> pd.DataFrame:
        """Load ADS-B data from CSV.

        Accepts flexible column naming — will attempt to map common
        variations (lat/lon/alt) to the standard schema.
        """
        df = pd.read_csv(path, **kwargs)
        df = self._normalize_columns(df)
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean ADS-B data: remove invalid positions, sort by time, deduplicate."""
        df = df.copy()

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Remove invalid coordinates
        df = df[
            (df["latitude"].between(-90, 90))
            & (df["longitude"].between(-180, 180))
        ]

        # Remove negative altitudes (below sea level not valid for flights)
        if "altitude_ft" in df.columns:
            df = df[df["altitude_ft"] >= 0]

        # Remove duplicate timestamps per flight
        group_col = "flight_id" if "flight_id" in df.columns else None
        if group_col:
            df = df.drop_duplicates(subset=[group_col, "timestamp"])
        else:
            df = df.drop_duplicates(subset=["timestamp"])

        # Sort
        sort_cols = [group_col, "timestamp"] if group_col else ["timestamp"]
        df = df.sort_values(sort_cols).reset_index(drop=True)

        return df

    def compute_derived(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived columns: segment distance, time delta, ground speed."""
        df = df.copy()

        # Time deltas
        df["dt_seconds"] = df["timestamp"].diff().dt.total_seconds().fillna(0)

        # Haversine distance between consecutive points (km)
        df["segment_dist_km"] = _haversine_series(
            df["latitude"].shift(1), df["longitude"].shift(1),
            df["latitude"], df["longitude"],
        ).fillna(0)

        # Cumulative distance
        df["cumulative_dist_km"] = df["segment_dist_km"].cumsum()

        # Altitude rate of change
        if "altitude_ft" in df.columns:
            df["alt_change_ft"] = df["altitude_ft"].diff().fillna(0)

        return df

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map common column name variations to standard names."""
        col_map = {}
        lower_cols = {c.lower().strip(): c for c in df.columns}

        mappings = {
            "timestamp": ["timestamp", "time", "datetime", "date_time", "ts"],
            "latitude": ["latitude", "lat"],
            "longitude": ["longitude", "lon", "lng", "long"],
            "altitude_ft": ["altitude_ft", "altitude", "alt", "alt_ft", "baroaltitude"],
            "speed_kts": ["speed_kts", "speed", "groundspeed", "gs", "velocity"],
            "heading": ["heading", "track", "trk", "course"],
            "vertical_rate_fpm": ["vertical_rate_fpm", "vertical_rate", "vertrate", "vs"],
            "icao24": ["icao24", "icao", "hex", "hex_ident"],
            "callsign": ["callsign", "call_sign", "flight"],
            "flight_id": ["flight_id", "flightid", "id"],
        }

        for standard, variants in mappings.items():
            for v in variants:
                if v in lower_cols:
                    col_map[lower_cols[v]] = standard
                    break

        df = df.rename(columns=col_map)

        missing = self.REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. "
                f"Available: {list(df.columns)}"
            )

        return df


def _haversine_series(
    lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series
) -> pd.Series:
    """Vectorised haversine distance in km."""
    R = 6371.0
    lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))
