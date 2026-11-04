"""Tests for fallback estimator."""

import pandas as pd
from openburn.fuel.fallback import FallbackEstimator, _classify_aircraft


def test_classify_aircraft():
    assert _classify_aircraft("B738") == "narrowbody"
    assert _classify_aircraft("B773") == "widebody"
    assert _classify_aircraft("AT72") == "turboprop"
    assert _classify_aircraft("DH8D") == "turboprop"


def test_fallback_estimator_defaults():
    est = FallbackEstimator()
    result = est.estimate("B738", 1000)
    assert result["fuel_burn_kg"] > 0
    assert result["estimation_source"] == "fallback_default"


def test_fallback_estimator_fitted():
    df = pd.DataFrame({
        "aircraft_family": ["B738"] * 20,
        "distance_km": [1000] * 20,
        "fuel_burn_kg": [3500] * 20,
    })
    est = FallbackEstimator()
    est.fit(df)
    result = est.estimate("B738", 1000)
    assert result["fuel_burn_kg"] > 0
    assert "fallback_default" not in result["estimation_source"]
