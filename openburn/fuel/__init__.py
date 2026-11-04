"""Fuel burn estimation — the core of OpenBurn."""

from openburn.fuel.model import FuelBurnModel
from openburn.fuel.features import FeatureExtractor
from openburn.fuel.fallback import FallbackEstimator

__all__ = ["FuelBurnModel", "FeatureExtractor", "FallbackEstimator"]
