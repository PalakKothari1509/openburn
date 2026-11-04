"""Flight phase segmentation — split trajectory into climb, cruise, descent."""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd


class FlightPhase(str, Enum):
    GROUND = "ground"
    CLIMB = "climb"
    CRUISE = "cruise"
    DESCENT = "descent"
    UNKNOWN = "unknown"


class FlightSegmenter:
    """Segment a flight trajectory into phases based on altitude profile.

    Uses altitude rate of change and absolute altitude thresholds
    to classify each position report into a flight phase.
    """

    def __init__(
        self,
        ground_alt_ft: float = 2000,
        climb_rate_threshold_fpm: float = 300,
        descent_rate_threshold_fpm: float = -300,
        smoothing_window: int = 5,
    ):
        self.ground_alt_ft = ground_alt_ft
        self.climb_rate_threshold_fpm = climb_rate_threshold_fpm
        self.descent_rate_threshold_fpm = descent_rate_threshold_fpm
        self.smoothing_window = smoothing_window

    def segment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a 'flight_phase' column to the trajectory dataframe.

        Expects columns: altitude_ft, dt_seconds (from ADSBProcessor.compute_derived)
        """
        df = df.copy()

        # Smooth altitude to reduce noise
        alt = df["altitude_ft"].rolling(
            self.smoothing_window, min_periods=1, center=True
        ).mean()

        # Compute vertical rate (ft/min)
        dt_min = df["dt_seconds"] / 60
        alt_change = alt.diff().fillna(0)
        vrate = np.where(dt_min > 0, alt_change / dt_min, 0)

        phases = []
        for a, vr in zip(alt, vrate):
            if a < self.ground_alt_ft:
                phases.append(FlightPhase.GROUND)
            elif vr > self.climb_rate_threshold_fpm:
                phases.append(FlightPhase.CLIMB)
            elif vr < self.descent_rate_threshold_fpm:
                phases.append(FlightPhase.DESCENT)
            else:
                phases.append(FlightPhase.CRUISE)

        df["flight_phase"] = phases
        return df

    def summarise(self, df: pd.DataFrame) -> dict:
        """Summarise a segmented flight: distance and time per phase."""
        if "flight_phase" not in df.columns:
            df = self.segment(df)

        summary = {}
        for phase in FlightPhase:
            mask = df["flight_phase"] == phase
            phase_df = df[mask]
            if len(phase_df) == 0:
                continue
            summary[phase.value] = {
                "distance_km": round(phase_df["segment_dist_km"].sum(), 2)
                if "segment_dist_km" in phase_df.columns
                else None,
                "duration_seconds": round(phase_df["dt_seconds"].sum(), 1)
                if "dt_seconds" in phase_df.columns
                else None,
                "points": len(phase_df),
            }
        return summary
