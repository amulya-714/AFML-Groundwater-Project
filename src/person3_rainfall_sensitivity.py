"""
Person 3 – Task D: Rainfall QC Sensitivity Analysis
======================================================
Compares model evaluation metrics on:
  1. Full test dataset (main analysis)
  2. Test subset excluding observations with rainfall QC flags
  3. Test subset with only high-confidence rainfall

Does NOT modify the original dataset.

Run from the repository root:
    python src/person3_rainfall_sensitivity.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REPO_ROOT = Path(".")
DATA_PATH = REPO_ROOT / "data/processed/groundwater_rainfall_features_qc.csv"
PRED_PATH = REPO_ROOT / "results/baseline_predictions.csv"
OUT_DIR   = REPO_ROOT / "results/person3"
FIG_DIR   = REPO_ROOT / "results/figures/person3"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CUT = "2018-01-01"
TARGET    = "Groundwater Level Quarterly Manual (meter)"

print("=" * 60)
print("PERSON 3 – RAINFALL QC SENSITIVITY ANALYSIS")
print("=" * 60)

# ── Load raw dataset ─────────────────────────────────────────────
df_raw = pd.read_csv(DATA_PATH, parse_dates=["observation_date"])
print(f"\nFull dataset columns: {list(df_raw.columns)}")
print(f"Total rows: {len(df_raw):,}")

df_raw["well_id"] = (
    df_raw["Station"].astype(str) + "_" +
    df_raw["Latitude"].astype(str) + "_" +
    df_raw["Longitude"].astype(str)
)
df_raw = df_raw.sort_values(["well_id", "observation_date"])
df_raw["previous_gwl"] = df_raw.groupby("well_id")[TARGET].shift(1)

# ── Identify QC columns ─────────────────────────────────────────
qc_cols = [c for c in df_raw.columns if "qc" in c.lower() or "flag" in c.lower()
           or "quality" in c.lower() or "valid" in c.lower()]
print(f"\nQC/flag columns found: {qc_cols}")

for col in qc_cols:
    print(f"  {col}: unique values = {sorted(df_raw[col].dropna().unique())[:15]}")
    print(f"         null count = {df_raw[col].isna().sum()}")

# ── Load predictions ─────────────────────────────────────────────
pred = pd.read_csv(PRED_PATH, parse_dates=["date"])

test_raw = df_raw[df_raw["observation_date"] >= TRAIN_CUT].copy()
test = test_raw.merge(
    pred, left_on=["observation_date", "well_id"],
    right_on=["date", "well_id"], how="inner"
)
test["abs_error_rf"]  = (test[TARGET] - test["random_forest_prediction_m"]).abs()
test["sq_error_rf"]   = (test[TARGET] - test["random_forest_prediction_m"]) ** 2
test["abs_error_pers"] = (test[TARGET] - test["persistence_prediction_m"]).abs()

print(f"\nTest set rows with predictions: {len(test):,}")

# ── Build scenario masks ─────────────────────────────────────────
scenarios = {}
scenarios["Full test set"] = np.ones(len(test), dtype=bool)

for col in qc_cols:
    if test[col].dtype == object or test[col].dtype.name == "category":
        # String-type QC: typical good values are 'G', 'Good', '1', etc.
        good_vals = [v for v in test[col].dropna().unique()
                     if str(v).upper() in {"G", "GOOD", "1", "OK", "VALID", "TRUE", "YES"}]
        if not good_vals:
            # Use mode as "good"
            mode_val = test[col].mode()
            good_vals = [mode_val.iloc[0]] if len(mode_val) > 0 else []
        if good_vals:
            mask = test[col].isin(good_vals)
            scenarios[f"Excl bad {col} ({', '.join(str(v) for v in good_vals)})"] = mask
            scenarios[f"Keep flagged {col} only"] = ~mask & test[col].notna()
    elif pd.api.types.is_numeric_dtype(test[col]):
        # Numeric QC: 0/1, or values where 0 = good, 1 = flagged
        unique_vals = sorted(test[col].dropna().unique())
        if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            # Assume 0 = good, 1 = flagged
            mask_good    = test[col] == 0
            mask_flagged = test[col] == 1
            scenarios[f"Excl flagged {col} (flag==1)"] = mask_good
            scenarios[f"Flagged only {col} (flag==1)"] = mask_flagged & (test[col].notna())
        elif len(unique_vals) > 2:
            # Treat non-null as having any value; check for NaN pattern
            no_nan_mask = test[col].notna()
            if no_nan_mask.sum() < len(test):
                scenarios[f"Non-null {col} only"] = no_nan_mask

# Also check for rainfall-related NaN patterns
for feat in ["rainfall_30d_mm", "rainfall_90d_mm"]:
    if feat in test.columns:
        not_missing = test[feat].notna()
        if not_missing.sum() < len(test):
            scenarios[f"Non-missing {feat}"] = not_missing

# Always include a "no-NaN rainfall" scenario
rain_complete = test["rainfall_30d_mm"].notna() & test["rainfall_90d_mm"].notna()
if rain_complete.sum() < len(test):
    scenarios["Both rainfall features present"] = rain_complete

print(f"\nScenarios to evaluate: {list(scenarios.keys())}")

# ── Report QC flag distribution in full dataset vs test ──────────
print("\n── QC FLAG DISTRIBUTION ──")
train_rawqc = df_raw[df_raw["observation_date"] < TRAIN_CUT]
for col in qc_cols:
    full_flagged  = (df_raw[col].astype(bool)).sum()
    test_flagged  = (test[col].astype(bool)).sum() if col in test.columns else "N/A"
    train_flagged = (train_rawqc[col].astype(bool)).sum() if col in train_rawqc.columns else "N/A"
    print(f"  {col}: "
          f"full_dataset={full_flagged}, "
          f"train_flagged={train_flagged}, "
          f"test_flagged={test_flagged}")

# ── Compute metrics per scenario ─────────────────────────────────
def scenario_metrics(df, mask, label):
    g = df[mask]
    if len(g) < 5:
        return {"scenario": label, "n_obs": len(g), "n_wells": g["well_id"].nunique(),
                "MAE_m": np.nan, "RMSE_m": np.nan, "R2": np.nan,
                "excluded_from_full": len(df) - len(g)}
    actual  = g[TARGET].to_numpy()
    rf_pred = g["random_forest_prediction_m"].to_numpy()
    return {
        "scenario": label,
        "n_obs":    len(g),
        "n_wells":  g["well_id"].nunique(),
        "MAE_m":    mean_absolute_error(actual, rf_pred),
        "RMSE_m":   np.sqrt(mean_squared_error(actual, rf_pred)),
        "R2":       r2_score(actual, rf_pred),
        "excluded_from_full": len(df) - len(g),
    }

rows = []
for label, mask in scenarios.items():
    rows.append(scenario_metrics(test, mask, label))

sensitivity_df = pd.DataFrame(rows)
print("\n── RAINFALL QC SENSITIVITY RESULTS ──")
print(sensitivity_df.round(4).to_string(index=False))

# ── Additional: check flagged rows in full dataset (train+test) ──
print("\n── QC FLAG IMPACT ON FULL DATASET ──")
for col in qc_cols:
    flagged_count = (df_raw[col].astype(bool)).sum()
    total         = len(df_raw)
    print(f"  {col}: {flagged_count}/{total} flagged rows in complete dataset")
    if flagged_count == 0:
        print(f"    → No flagged rows. Sensitivity check not applicable.")
    else:
        flagged_df = df_raw[df_raw[col].astype(bool)]
        train_flag = (flagged_df["observation_date"] < TRAIN_CUT).sum()
        test_flag  = (flagged_df["observation_date"] >= TRAIN_CUT).sum()
        print(f"    → In training period: {train_flag}")
        print(f"    → In test period    : {test_flag}")

# ── Check if excluding QC-flagged rows changes conclusions ───────
full_mae  = sensitivity_df.loc[sensitivity_df["scenario"] == "Full test set", "MAE_m"].values[0]
full_r2   = sensitivity_df.loc[sensitivity_df["scenario"] == "Full test set", "R2"].values[0]

print("\n── SENSITIVITY INTERPRETATION ──")
for _, row in sensitivity_df.iterrows():
    if row["scenario"] == "Full test set" or np.isnan(row["MAE_m"]):
        continue
    delta_mae = row["MAE_m"] - full_mae
    delta_r2  = row["R2"] - full_r2
    direction = "higher" if delta_mae > 0 else "lower"
    print(f"  {row['scenario']}: "
          f"ΔMAE={delta_mae:+.4f} m ({direction}), ΔR²={delta_r2:+.4f}, "
          f"n_obs={row['n_obs']:,} (excl={row['excluded_from_full']:,})")

# ── Save ─────────────────────────────────────────────────────────
sensitivity_df.to_csv(OUT_DIR / "rainfall_sensitivity.csv", index=False)
print(f"\nSaved: results/person3/rainfall_sensitivity.csv")

# ── Figure: Sensitivity comparison ───────────────────────────────
valid_rows = sensitivity_df[sensitivity_df["MAE_m"].notna()].copy()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Rainfall QC Sensitivity Analysis (Person 3)", fontsize=13)
metrics_plot = [("MAE_m", "MAE (m)"), ("RMSE_m", "RMSE (m)"), ("R2", "R²")]
for ax, (col, ylabel) in zip(axes, metrics_plot):
    colors = ["tomato" if s == "Full test set" else "steelblue"
              for s in valid_rows["scenario"]]
    bars = ax.barh(valid_rows["scenario"], valid_rows[col],
                   color=colors, alpha=0.85, edgecolor="black")
    ax.set_title(ylabel)
    ax.set_xlabel(ylabel)
    for bar, v in zip(bars, valid_rows[col]):
        ax.text(v + 0.005 * abs(v), bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=7)

plt.tight_layout()
plt.savefig(FIG_DIR / "coverage_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/figures/person3/coverage_comparison.png")

print("\n" + "=" * 60)
print("RAINFALL QC SENSITIVITY ANALYSIS COMPLETE")
print("=" * 60)
