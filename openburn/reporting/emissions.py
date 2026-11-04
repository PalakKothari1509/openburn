"""Emissions reporting — convert fuel burn to CO2 and other emissions."""

from __future__ import annotations

import pandas as pd

# Standard emission factors (kg per kg of jet fuel)
# Source: ICAO, IPCC
EMISSION_FACTORS = {
    "co2_kg_per_kg_fuel": 3.16,       # CO2
    "h2o_kg_per_kg_fuel": 1.24,       # Water vapour
    "nox_kg_per_kg_fuel": 0.014,      # NOx (average, varies by engine)
    "so2_kg_per_kg_fuel": 0.0012,     # Sulphur dioxide
    "co_kg_per_kg_fuel": 0.0028,      # Carbon monoxide
    "hc_kg_per_kg_fuel": 0.0004,      # Hydrocarbons
    "pm_kg_per_kg_fuel": 0.00004,     # Particulate matter
}

# Jet fuel density (kg per litre)
JET_A1_DENSITY_KG_PER_L = 0.804


class EmissionsReport:
    """Convert fuel burn estimates to emissions for reporting."""

    def __init__(self, factors: dict[str, float] | None = None):
        self.factors = factors or EMISSION_FACTORS

    def from_fuel_kg(self, fuel_kg: float) -> dict[str, float]:
        """Calculate all emissions from fuel burn in kg."""
        return {
            name.replace("_kg_per_kg_fuel", ""): round(fuel_kg * factor, 4)
            for name, factor in self.factors.items()
        }

    def from_fuel_litres(self, fuel_litres: float) -> dict[str, float]:
        """Calculate emissions from fuel burn in litres."""
        return self.from_fuel_kg(fuel_litres * JET_A1_DENSITY_KG_PER_L)

    def batch_report(
        self,
        df: pd.DataFrame,
        fuel_col: str = "fuel_burn_kg",
    ) -> pd.DataFrame:
        """Add emission columns to a DataFrame of flights."""
        df = df.copy()
        for name, factor in self.factors.items():
            col_name = name.replace("_kg_per_kg_fuel", "_kg")
            df[col_name] = (df[fuel_col] * factor).round(4)
        return df

    def aggregate(
        self,
        df: pd.DataFrame,
        group_by: str | list[str] | None = None,
        fuel_col: str = "fuel_burn_kg",
    ) -> pd.DataFrame:
        """Aggregate emissions, optionally grouped by airline/route/period."""
        df = self.batch_report(df, fuel_col)
        emission_cols = [c for c in df.columns if c.endswith("_kg") and c != fuel_col]
        cols_to_sum = [fuel_col] + emission_cols

        if group_by:
            return df.groupby(group_by)[cols_to_sum].sum().round(2).reset_index()
        else:
            totals = df[cols_to_sum].sum().round(2)
            totals["n_flights"] = len(df)
            return pd.DataFrame([totals])
