"""Validation framework — compare estimated vs actual fuel burn."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ValidationMetrics:
    mae_kg: float           # Mean Absolute Error
    rmse_kg: float          # Root Mean Squared Error
    mape_pct: float         # Mean Absolute Percentage Error
    median_error_kg: float
    r_squared: float
    n_flights: int
    bias_kg: float          # Positive = overestimate, negative = underestimate


class ValidationReport:
    """Compare estimated fuel burn against actual fuel data."""

    def validate(
        self,
        df: pd.DataFrame,
        actual_col: str = "fuel_burn_kg",
        estimated_col: str = "fuel_burn_estimated_kg",
    ) -> ValidationMetrics:
        """Compute validation metrics.

        Args:
            df: DataFrame with actual and estimated fuel columns.
            actual_col: Column name for actual fuel burn.
            estimated_col: Column name for estimated fuel burn.
        """
        valid = df[[actual_col, estimated_col]].dropna()
        actual = valid[actual_col].values
        estimated = valid[estimated_col].values

        if len(actual) == 0:
            raise ValueError("No valid rows to compare")

        errors = estimated - actual
        abs_errors = np.abs(errors)
        pct_errors = np.where(actual > 0, abs_errors / actual * 100, 0)

        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((actual - actual.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return ValidationMetrics(
            mae_kg=round(float(abs_errors.mean()), 2),
            rmse_kg=round(float(np.sqrt((errors ** 2).mean())), 2),
            mape_pct=round(float(pct_errors.mean()), 2),
            median_error_kg=round(float(np.median(errors)), 2),
            r_squared=round(float(r2), 4),
            n_flights=len(actual),
            bias_kg=round(float(errors.mean()), 2),
        )

    def validate_by_family(
        self,
        df: pd.DataFrame,
        family_col: str = "aircraft_family",
        actual_col: str = "fuel_burn_kg",
        estimated_col: str = "fuel_burn_estimated_kg",
    ) -> dict[str, ValidationMetrics]:
        """Run validation per aircraft family."""
        results = {}
        for family, group in df.groupby(family_col):
            if len(group) < 2:
                continue
            try:
                results[str(family)] = self.validate(group, actual_col, estimated_col)
            except ValueError:
                continue
        return results

    def summary_table(
        self,
        by_family: dict[str, ValidationMetrics],
    ) -> pd.DataFrame:
        """Convert per-family validation to a summary DataFrame."""
        rows = []
        for family, m in sorted(by_family.items()):
            rows.append({
                "aircraft_family": family,
                "n_flights": m.n_flights,
                "mae_kg": m.mae_kg,
                "rmse_kg": m.rmse_kg,
                "mape_pct": m.mape_pct,
                "r_squared": m.r_squared,
                "bias_kg": m.bias_kg,
            })
        return pd.DataFrame(rows)
