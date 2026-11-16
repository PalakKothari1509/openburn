# OpenBurn

Open-source aviation fuel burn estimation — a data-driven alternative to
BADA/PyContrails. Instead of physics-based aerodynamic coefficients,
OpenBurn trains machine-learning models per aircraft family on real
airline operational data (actual fuel records from real flights), and
uses them to estimate fuel burn, CO2, and other emissions for a single
flight, a batch of flights, or a departure → arrival route.

See [HOW_IT_WORKS.md](HOW_IT_WORKS.md) for a full code walkthrough of
every module and how a request flows through the system end to end.

## Project layout

- `openburn/` — the core Python library (feature extraction, per-family
  ML models, fallback estimator, emissions reporting, validation)
- `api/` — the Flask REST API (`api/app.py`) that also serves the web
  frontend
- `frontend/` — a static HTML/CSS/JS dashboard (no build step)
- `data/` — airport coordinates and sample training CSVs
- `tests/` — pytest suite

## Requirements

- Python >= 3.10

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

## Running the app

```bash
python api/app.py
```

This serves both the API and the frontend dashboard at
`http://localhost:5000`.

Alternatively, with Docker:

```bash
docker-compose up --build
```

This runs the Flask API (via gunicorn) on port 5000 and the frontend
(via nginx) on port 8080.

## Training a model

The dashboard's "Train Model" tab, or directly via the API:

```bash
curl -X POST http://localhost:5000/train \
  -H "Content-Type: application/json" \
  -d '{"csv_path": "data/sample_fuel_data.csv"}'
```

Training data can use whatever column names an airline's export uses —
`openburn/fuel/features.py` auto-detects common aliases (e.g.
`OFFBLOCK_FUEL`, `fuel_kg`, `total_fuel_kg` are all recognized as the
same field).

## Estimating fuel burn

```bash
curl -X POST http://localhost:5000/estimate/route \
  -H "Content-Type: application/json" \
  -d '{"departure": "MAD", "arrival": "LHR", "aircraft_type": "A320", "passengers": 150}'
```

## Tests

```bash
pytest
```

## Status

Early-stage (alpha). No authentication is implemented on the API — it
is intended to run locally or behind your own access controls, not to
be exposed directly to the public internet.

## License

Apache-2.0 — see [LICENSE](LICENSE).
