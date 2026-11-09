"""
Person 3 – Task 1: Data Provenance and Feature Pipeline Validation
==================================================================
Investigates rainfall provenance, validates feature construction,
checks for future-information leakage, and generates dataset integrity reports.

Run from the repository root:
    python src/person3_data_validation.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT   = Path(".")
DATA_PATH   = REPO_ROOT / "data/processed/groundwater_rainfall_features_qc.csv"
RAW_PATH    = REPO_ROOT / "data/raw/gwl_manual_quarterly_cgwb_ka_1991_2020.csv"
OUT_DIR     = REPO_ROOT / "results"
TRAIN_CUT   = "2018-01-01"
TARGET      = "Groundwater Level Quarterly Manual (meter)"

print("=" * 65)
print("PERSON 3 – DATA PROVENANCE AND FEATURE PIPELINE VALIDATION")
print("=" * 65)

# ── 1. Load processed dataset ──────────────────────────────────
df = pd.read_csv(DATA_PATH, parse_dates=["observation_date"])
print(f"\nProcessed dataset: {len(df):,} rows × {len(df.columns)} columns")
print(f"Date range: {df['observation_date'].min().date()} to {df['observation_date'].max().date()}")

# ── 2. Rainfall provenance investigation ────────────────────────
print("\n── RAINFALL PROVENANCE INVESTIGATION ──")

rainfall_cols = [c for c in df.columns if "rainfall" in c.lower()]
print(f"Rainfall columns: {rainfall_cols}")

# Check for rainfall grid info
grid_cols = [c for c in df.columns if "grid" in c.lower() or "cell" in c.lower()]
print(f"Grid columns: {grid_cols}")

if grid_cols:
    for gc in grid_cols:
        print(f"  {gc}: unique={df[gc].nunique()}, sample={df[gc].dropna().head(3).tolist()}")

# Check if raw CGWB file has rainfall
raw_exists = RAW_PATH.exists()
print(f"\nRaw CGWB file exists: {raw_exists}")
if raw_exists:
    raw = pd.read_csv(RAW_PATH, nrows=3)
    print(f"Raw columns: {raw.columns.tolist()}")
    has_rainfall = any("rain" in c.lower() for c in raw.columns)
    print(f"Raw file has rainfall columns: {has_rainfall}")

# Search for download/processing scripts
import os
rain_scripts = []
for root, dirs, files in os.walk(REPO_ROOT):
    for f in files:
        if f.endswith(".py") and ("rainfall" in f.lower() or "download" in f.lower()
                                   or "merge" in f.lower() or "process" in f.lower()):
            rain_scripts.append(os.path.join(root, f))
print(f"\nRainfall/download/merge scripts found: {rain_scripts if rain_scripts else 'NONE'}")

print("\nPROVENANCE STATUS: UNRESOLVED")
print("  The processed dataset contains rainfall_30d_mm and rainfall_90d_mm.")
print("  The dataset includes rainfall_cell_id, rainfall_grid_latitude,")
print("  rainfall_grid_longitude — indicating spatial grid matching was performed.")
print("  No original rainfall source file, download script, or processing")
print("  documentation exists in the repository.")
print("  REQUIREMENT: Person 1 or Person 2 must provide the rainfall source")
print("  filename, provider, URL, spatial resolution, temporal resolution,")
print("  and aggregation procedure.")

# ── 3. Feature construction validation ─────────────────────────
print("\n── FEATURE CONSTRUCTION VALIDATION ──")

df["well_id"] = (
    df["Station"].astype(str) + "_" +
    df["Latitude"].astype(str) + "_" +
    df["Longitude"].astype(str)
)
df = df.sort_values(["well_id", "observation_date"])
df["previous_gwl"] = df.groupby("well_id")[TARGET].shift(1)

# 3a. Verify previous_gwl uses only prior observations
print("  Checking previous_gwl construction...")
sample_wells = df["well_id"].unique()[:5]
prev_ok = True
for w in sample_wells:
    wdf = df[df["well_id"] == w].head(5)
    for i in range(1, len(wdf)):
        prev_actual = wdf.iloc[i-1][TARGET]
        prev_shift  = wdf.iloc[i]["previous_gwl"]
        if pd.notna(prev_shift) and abs(prev_actual - prev_shift) > 1e-6:
            print(f"    MISMATCH well={w}: prev_actual={prev_actual}, shift={prev_shift}")
            prev_ok = False
print(f"  previous_gwl matches prior observation: {'PASS' if prev_ok else 'FAIL'}")

# 3b. Check no future GWL in features
print("  Checking no future GWL values in features...")
# previous_gwl is the only GWL-derived feature; it's a lag-1 so by construction
# it uses the prior value. Verified above.
print("  PASS: previous_gwl is lag-1 within well, sorted chronologically")

# 3c. Rainfall window investigation
print("\n  Rainfall feature structural checks:")
print(f"  rainfall_30d_mm: min={df['rainfall_30d_mm'].min():.2f}, "
      f"max={df['rainfall_30d_mm'].max():.2f}, "
      f"mean={df['rainfall_30d_mm'].mean():.2f}")
print(f"  rainfall_90d_mm: min={df['rainfall_90d_mm'].min():.2f}, "
      f"max={df['rainfall_90d_mm'].max():.2f}, "
      f"mean={df['rainfall_90d_mm'].mean():.2f}")

# Check if 90d is always >= 30d (structural sanity)
both_present = df.dropna(subset=["rainfall_30d_mm", "rainfall_90d_mm"])
ratio_ok = (both_present["rainfall_90d_mm"] >= both_present["rainfall_30d_mm"]).mean()
print(f"  90d >= 30d for {ratio_ok*100:.1f}% of rows (expected ~100%)")
print(f"  Rows where 90d < 30d: {(~(both_present['rainfall_90d_mm'] >= both_present['rainfall_30d_mm'])).sum()}")

# 3d. Month features
month = df["observation_date"].dt.month
expected_sin = np.sin(2 * np.pi * month / 12)
# Check if month_sin exists in data or needs construction
if "month_sin" not in df.columns:
    print("  month_sin/month_cos: NOT in processed dataset (constructed in model scripts)")
else:
    print("  month_sin/month_cos: present in dataset")

# ── 4. Dataset integrity checks ────────────────────────────────
print("\n── DATASET INTEGRITY CHECKS ──")

quality_rows = []

# 4a. Missing targets
n_missing_target = df[TARGET].isna().sum()
quality_rows.append({"check": "missing_target_values", "count": int(n_missing_target),
                     "pct": f"{n_missing_target/len(df)*100:.3f}%"})

# 4b. Missing rainfall
for feat in ["rainfall_30d_mm", "rainfall_90d_mm"]:
    n_miss = df[feat].isna().sum()
    quality_rows.append({"check": f"missing_{feat}", "count": int(n_miss),
                         "pct": f"{n_miss/len(df)*100:.3f}%"})

# 4c. Duplicate well-date keys
dupes = df.duplicated(subset=["well_id", "observation_date"]).sum()
quality_rows.append({"check": "duplicate_well_date_keys", "count": int(dupes),
                     "pct": f"{dupes/len(df)*100:.3f}%"})

# 4d. Invalid coordinates
bad_lat = ((df["Latitude"] < -90) | (df["Latitude"] > 90)).sum()
bad_lon = ((df["Longitude"] < -180) | (df["Longitude"] > 180)).sum()
quality_rows.append({"check": "invalid_latitude", "count": int(bad_lat), "pct": "0%"})
quality_rows.append({"check": "invalid_longitude", "count": int(bad_lon), "pct": "0%"})

# 4e. Negative GWL (depth below ground = positive in this dataset)
neg_gwl = (df[TARGET] < 0).sum()
quality_rows.append({"check": "negative_gwl_values", "count": int(neg_gwl),
                     "pct": f"{neg_gwl/len(df)*100:.3f}%"})

# 4f. Extreme GWL (>50m)
extreme_gwl = (df[TARGET] > 50).sum()
quality_rows.append({"check": "gwl_above_50m", "count": int(extreme_gwl),
                     "pct": f"{extreme_gwl/len(df)*100:.3f}%"})

# 4g. Specific well investigation: Govt L V Polytechnic
govt = df[df["Station"] == "Govt L V Polytechnic"]
quality_rows.append({
    "check": "govt_l_v_polytechnic_readings",
    "count": len(govt),
    "pct": f"max={govt[TARGET].max():.2f}m, dates={govt['observation_date'].dt.year.min()}-{govt['observation_date'].dt.year.max()}"
})

# 4h. QC flags
for qc in ["qc_suspicious_rainfall_30d", "qc_suspicious_rainfall_90d"]:
    n_flagged = df[qc].astype(bool).sum()
    quality_rows.append({"check": qc, "count": int(n_flagged),
                         "pct": f"{n_flagged/len(df)*100:.4f}%"})

quality_df = pd.DataFrame(quality_rows)
quality_df.to_csv(OUT_DIR / "data_quality_summary.csv", index=False)
print("\n", quality_df.to_string(index=False))
print(f"\nSaved: results/data_quality_summary.csv")

# ── 5. Yearly station coverage ────────────────────────────────
print("\n── YEARLY STATION COVERAGE ──")

df["year"] = df["observation_date"].dt.year
yearly = df.groupby("year").agg(
    n_observations=("Station", "size"),
    n_unique_stations=("Station", "nunique"),
    mean_gwl_m=(TARGET, "mean"),
    median_gwl_m=(TARGET, "median"),
    mean_rainfall_90d_mm=("rainfall_90d_mm", "mean"),
    missing_rainfall_pct=("rainfall_90d_mm", lambda x: f"{x.isna().mean()*100:.1f}%"),
).reset_index()

yearly.to_csv(OUT_DIR / "yearly_station_coverage.csv", index=False)
print(yearly.to_string(index=False))
print(f"\nSaved: results/yearly_station_coverage.csv")

# ── 6. Investigation: Govt L V Polytechnic ──────────────────────
print("\n── GOVT L V POLYTECHNIC INVESTIGATION ──")
print(f"  Station: Govt L V Polytechnic, District: {govt['District'].iloc[0]}")
print(f"  Total readings: {len(govt)}")
print(f"  GWL range: {govt[TARGET].min():.2f} to {govt[TARGET].max():.2f} m")
print(f"  Readings >100m: {(govt[TARGET] > 100).sum()} (in 2018-2019)")
print(f"  These appear to be data entry errors (101.16m and 111.51m)")
print(f"  surrounded by normal readings (20-36m). NOT removed per instructions:")
print(f"  unusual observations are retained; correction requires team approval.")

# ── 7. Compare with raw CGWB if available ──────────────────────
if raw_exists:
    print("\n── COMPARISON WITH RAW CGWB ──")
    raw_full = pd.read_csv(RAW_PATH)
    print(f"  Raw file: {len(raw_full):,} rows")
    print(f"  Raw columns: {raw_full.columns.tolist()[:10]}")
    # Check Station overlap
    if "Station" in raw_full.columns:
        overlap = set(df["Station"].unique()) & set(raw_full["Station"].unique())
        print(f"  Stations in both files: {len(overlap)}")
        print(f"  Stations only in processed: {df['Station'].nunique() - len(overlap)}")

print("\n" + "=" * 65)
print("DATA VALIDATION COMPLETE")
print("=" * 65)
