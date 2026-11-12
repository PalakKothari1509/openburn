"""Quick test script — run with: python run_test.py"""

import warnings
warnings.filterwarnings("ignore")

from openburn.fuel.model import FuelBurnModel
from openburn.reporting.emissions import EmissionsReport
from openburn.validation.compare import ValidationReport

DATA_FILE = "flights_XXX_Air_01_01_2025_to_31_01_2025_20261104_120906.csv"

print("=" * 65)
print("  OpenBurn — Aviation Fuel Burn Estimation Test")
print("=" * 65)

# 1. Train model (80% train, 20% holdout test)
print("\n[1] Training model...")
model = FuelBurnModel()
result = model.train_from_csv(DATA_FILE, test_size=0.2)

print(f"    Total flights loaded:  {result['total_flights']:,}")
print(f"    Training flights:      {result['training_flights']:,}")
print(f"    Holdout test flights:  {result['test_flights']:,}")
print(f"    Aircraft families:     {', '.join(result['families_trained'])}")

# 2. Show holdout validation
if "holdout_validation" in result:
    hv = result["holdout_validation"]["overall"]
    print(f"\n[2] Holdout Validation (unseen data)")
    print(f"    MAE:   {hv['mae_kg']:,.0f} kg")
    print(f"    RMSE:  {hv['rmse_kg']:,.0f} kg")
    print(f"    MAPE:  {hv['mape_pct']:.1f}%")
    print(f"    R²:    {hv['r_squared']:.4f}")
    print(f"    Bias:  {hv['bias_kg']:+,.0f} kg")

    print(f"\n    Per aircraft family:")
    print(f"    {'Family':<8} {'MAE':>7} {'RMSE':>7} {'MAPE':>6} {'R²':>8} {'Flights':>8}")
    print(f"    {'-'*52}")
    for fam, m in sorted(result["holdout_validation"]["per_family"].items()):
        print(f"    {fam:<8} {m['mae_kg']:>5,.0f}kg {m['rmse_kg']:>5,.0f}kg {m['mape_pct']:>5.1f}% {m['r_squared']:>8.4f} {m['n_flights']:>7,}")

# 3. Try some predictions
print(f"\n[3] Sample Predictions")
em = EmissionsReport()
examples = [
    ("A320", 1500, 170, "Madrid-London"),
    ("A339", 5000, 385, "Transatlantic"),
    ("E190", 800, 100, "Short-haul regional"),
    ("A21N", 3500, 280, "Medium-haul"),
    ("A20N", 1200, 140, "European short-haul"),
    ("A332", 7000, 500, "Long-haul widebody"),
    ("A319", 2000, 190, "European mid-range"),
    ("E195", 600, 80, "Short regional"),
]
print(f"    {'Type':<6} {'Distance':>8} {'Duration':>8} {'Fuel':>10} {'CO2':>10} {'Method':<14} Route")
print(f"    {'-'*80}")
for ac, dist, dur, desc in examples:
    est = model.estimate(ac, distance_km=dist, flight_duration_min=dur)
    co2 = em.from_fuel_kg(est["fuel_burn_kg"])["co2"]
    print(f"    {ac:<6} {dist:>6,}km {dur:>6}min {est['fuel_burn_kg']:>8,.0f}kg {co2:>8,.0f}kg {est['method']:<14} {desc}")

# 4. Test unknown aircraft type (fallback)
print(f"\n[4] Fallback Test (unknown aircraft)")
est = model.estimate("B738", distance_km=1500, flight_duration_min=180)
co2 = em.from_fuel_kg(est["fuel_burn_kg"])["co2"]
print(f"    B738 @ 1,500km: {est['fuel_burn_kg']:,.0f}kg fuel, {co2:,.0f}kg CO2 [{est['method']}, {est['confidence']}]")

# 5. Save model summary
model.save_summary("model_summary.json")
print(f"\n[5] Model summary saved to model_summary.json")

print("\n" + "=" * 65)
print("  Test complete!")
print("=" * 65)
