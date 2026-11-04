"""Tests for feature extraction and data loading."""

import os
import pytest
import pandas as pd

from openburn.fuel.features import FeatureExtractor

SAMPLE_DATA = os.path.join(os.path.dirname(__file__), "..", "data", "sample_fuel_data.csv")


@pytest.fixture
def extractor():
    return FeatureExtractor()


def test_load_csv(extractor):
    df = extractor.load_fuel_csv(SAMPLE_DATA)
    assert len(df) == 30
    assert "fuel_burn_kg" in df.columns
    assert "distance_km" in df.columns
    assert "aircraft_family" in df.columns


def test_prepare_training_data(extractor):
    df = extractor.load_fuel_csv(SAMPLE_DATA)
    cleaned = extractor.prepare_training_data(df)
    assert len(cleaned) > 0
    assert (cleaned["fuel_burn_kg"] > 0).all()
    assert (cleaned["distance_km"] > 0).all()


def test_summary(extractor):
    df = extractor.load_fuel_csv(SAMPLE_DATA)
    s = extractor.summary(df)
    assert s["total_flights"] == 30
    assert "flights_per_family" in s
    assert "B738" in s["flights_per_family"]
