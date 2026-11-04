"""Aircraft type registry — maps ICAO type codes to aircraft metadata."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class AircraftType:
    icao_code: str          # e.g. "B738"
    name: str               # e.g. "Boeing 737-800"
    manufacturer: str       # e.g. "Boeing"
    engine_count: int
    engine_type: str        # "jet", "turboprop", "piston"
    mtow_kg: Optional[float] = None  # Max takeoff weight
    range_km: Optional[float] = None
    typical_cruise_speed_kts: Optional[float] = None
    typical_fuel_flow_kg_hr: Optional[float] = None


class AircraftRegistry:
    """Lookup aircraft performance data by ICAO type code."""

    def __init__(self) -> None:
        self._types: dict[str, AircraftType] = {}

    def load_csv(self, path: str | Path) -> None:
        """Load aircraft types from a CSV file.

        Expected columns: icao_code, name, manufacturer, engine_count,
        engine_type, mtow_kg, range_km, typical_cruise_speed_kts,
        typical_fuel_flow_kg_hr
        """
        path = Path(path)
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = row["icao_code"].strip().upper()
                self._types[code] = AircraftType(
                    icao_code=code,
                    name=row.get("name", "").strip(),
                    manufacturer=row.get("manufacturer", "").strip(),
                    engine_count=int(row.get("engine_count", 2)),
                    engine_type=row.get("engine_type", "jet").strip().lower(),
                    mtow_kg=_float_or_none(row.get("mtow_kg")),
                    range_km=_float_or_none(row.get("range_km")),
                    typical_cruise_speed_kts=_float_or_none(row.get("typical_cruise_speed_kts")),
                    typical_fuel_flow_kg_hr=_float_or_none(row.get("typical_fuel_flow_kg_hr")),
                )

    def get(self, icao_code: str) -> Optional[AircraftType]:
        return self._types.get(icao_code.strip().upper())

    def __contains__(self, icao_code: str) -> bool:
        return icao_code.strip().upper() in self._types

    def __len__(self) -> int:
        return len(self._types)

    @property
    def codes(self) -> list[str]:
        return sorted(self._types.keys())


def _float_or_none(val: str | None) -> Optional[float]:
    if val is None or val.strip() == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None
