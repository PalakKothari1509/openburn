"""Tests for the end-to-end fuel burn model."""

import os
import pytest
import numpy as np
import pandas as pd

from openburn.fuel.model import FuelBurnModel


@pytest.fixture
def synthetic_data(tmp_path):
    """Generate synthetic training data with enough flights per family."""
    rng = np.random.default_rng(42)
    rows = []
    for family, base_fuel_rate in [("B738", 4.0), ("A320", 3.8)]:
        for _ in range(50):
            dist = rng.uniform(500, 5000)
            dur = dist / rng.uniform(12, 15)
            fuel = dist * base_fuel_rate + rng.normal(0, 200)
            rows.append({
                "aircraft_family": family,
                "distance_km": round(dist, 1),
                "flight_duration_min": round(dur, 1),
                "fuel_burn_kg": round(max(500, fuel), 1),
            })
    df = pd.DataFrame(rows)
    path = tmp_path / "test_fuel.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def trained_model(synthetic_data):
    model = FuelBurnModel()
    model.train_from_csv(synthetic_data, test_size=0)
    return model


def test_train_from_csv(synthetic_data):
    model = FuelBurnModel()
    result = model.train_from_csv(synthetic_data, test_size=0)
    assert result["total_flights"] > 0
    assert len(result["families_trained"]) > 0
    assert model._is_trained


def test_train_with_holdout(synthetic_data):
    model = FuelBurnModel()
    result = model.train_from_csv(synthetic_data, test_size=0.3)
    assert result["training_flights"] > 0
    assert result["test_flights"] > 0
    assert result["training_flights"] + result["test_flights"] == result["total_flights"]
    assert "holdout_validation" in result


def test_estimate_known_family(trained_model):
    est = trained_model.estimate("B738", distance_km=1500, flight_duration_min=120)
    assert est["fuel_burn_kg"] > 0
    assert est["method"] == "trained_model"
    assert est["confidence"] == "high"


def test_estimate_unknown_family(trained_model):
    est = trained_model.estimate("ZZZZ", distance_km=1000)
    assert est["fuel_burn_kg"] > 0
    assert est["method"] == "fallback"
    assert est["confidence"] == "low"


def test_estimate_not_trained():
    model = FuelBurnModel()
    with pytest.raises(RuntimeError, match="not trained"):
        model.estimate("B738", distance_km=1000)


def test_batch_estimate(trained_model):
    df = pd.DataFrame([
        {"aircraft_family": "B738", "distance_km": 1500},
        {"aircraft_family": "A320", "distance_km": 800},
        {"aircraft_family": "ZZZZ", "distance_km": 600},
    ])
    result = trained_model.batch_estimate(df)
    assert "fuel_burn_estimated_kg" in result.columns
    assert (result["fuel_burn_estimated_kg"] > 0).all()
    assert result.loc[2, "estimation_method"] == "fallback"
