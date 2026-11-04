"""Tests for validation framework."""

import pandas as pd
from openburn.validation.compare import ValidationReport


def test_validation_metrics():
    df = pd.DataFrame({
        "fuel_burn_kg": [1000, 2000, 3000, 4000, 5000],
        "fuel_burn_estimated_kg": [1050, 1950, 3100, 3900, 5200],
    })
    report = ValidationReport()
    metrics = report.validate(df)
    assert metrics.n_flights == 5
    assert metrics.mae_kg > 0
    assert metrics.rmse_kg > 0
    assert 0 <= metrics.mape_pct <= 100
    assert metrics.r_squared > 0.9  # Should be well correlated
