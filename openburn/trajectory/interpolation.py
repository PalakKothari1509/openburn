"""Trajectory interpolation — fill gaps in ADS-B data."""

from __future__ import annotations

import numpy as np
import pandas as pd


class TrajectoryInterpolator:
    """Fill gaps in ADS-B trajectory data using interpolation.

    ADS-B coverage has gaps (poor ground station coverage over oceans,
    aircraft transponder issues, etc). This module detects gaps and
    fills them with reasonable interpolated positions.
    """

    def __init__(self, max_gap_seconds: float = 300, method: str = "linear"):
        """
        Args:
            max_gap_seconds: Maximum gap to interpolate across (default 5 min).
                Gaps larger than this are left as-is.
            method: Interpolation method — 'linear' or 'cubic'.
        """
        self.max_gap_seconds = max_gap_seconds
        self.method = method

    def detect_gaps(self, df: pd.DataFrame, threshold_seconds: float = 60) -> pd.DataFrame:
        """Identify gaps in the trajectory.

        Returns a dataframe of gap locations with start/end times and duration.
        """
        if "dt_seconds" not in df.columns:
            raise ValueError("Run ADSBProcessor.compute_derived() first")

        gaps = df[df["dt_seconds"] > threshold_seconds].copy()
        gaps = gaps[["timestamp", "dt_seconds", "latitude", "longitude", "altitude_ft"]].copy()
        gaps.rename(columns={"dt_seconds": "gap_seconds"}, inplace=True)
        return gaps.reset_index(drop=True)

    def interpolate(self, df: pd.DataFrame, interval_seconds: float = 10) -> pd.DataFrame:
        """Interpolate missing positions in the trajectory.

        Args:
            df: Trajectory dataframe (must have timestamp, lat, lon, altitude_ft, dt_seconds)
            interval_seconds: Target interval for interpolated points.

        Returns:
            Trajectory with interpolated points inserted (marked with is_interpolated=True).
        """
        df = df.copy()
        df["is_interpolated"] = False

        if "dt_seconds" not in df.columns:
            raise ValueError("Run ADSBProcessor.compute_derived() first")

        gap_mask = (df["dt_seconds"] > interval_seconds * 2) & (
            df["dt_seconds"] <= self.max_gap_seconds
        )
        gap_indices = df.index[gap_mask].tolist()

        new_rows = []
        for idx in gap_indices:
            if idx == 0:
                continue
            prev = df.loc[idx - 1] if (idx - 1) in df.index else None
            curr = df.loc[idx]
            if prev is None:
                continue

            gap_sec = curr["dt_seconds"]
            n_points = int(gap_sec / interval_seconds) - 1
            if n_points <= 0:
                continue

            fractions = np.linspace(0, 1, n_points + 2)[1:-1]
            for frac in fractions:
                new_rows.append({
                    "timestamp": prev["timestamp"] + pd.Timedelta(seconds=gap_sec * frac),
                    "latitude": prev["latitude"] + frac * (curr["latitude"] - prev["latitude"]),
                    "longitude": prev["longitude"] + frac * (curr["longitude"] - prev["longitude"]),
                    "altitude_ft": prev["altitude_ft"]
                    + frac * (curr["altitude_ft"] - prev["altitude_ft"]),
                    "is_interpolated": True,
                })

        if new_rows:
            interp_df = pd.DataFrame(new_rows)
            df = pd.concat([df, interp_df], ignore_index=True)
            df = df.sort_values("timestamp").reset_index(drop=True)

        return df
