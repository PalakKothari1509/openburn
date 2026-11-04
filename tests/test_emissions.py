"""Tests for emissions reporting."""

from openburn.reporting.emissions import EmissionsReport, JET_A1_DENSITY_KG_PER_L


def test_from_fuel_kg():
    report = EmissionsReport()
    result = report.from_fuel_kg(1000)
    assert result["co2"] == 3160.0
    assert result["h2o"] == 1240.0
    assert "nox" in result


def test_from_fuel_litres():
    report = EmissionsReport()
    result = report.from_fuel_litres(1000)
    expected_kg = 1000 * JET_A1_DENSITY_KG_PER_L
    assert result["co2"] == round(expected_kg * 3.16, 4)
