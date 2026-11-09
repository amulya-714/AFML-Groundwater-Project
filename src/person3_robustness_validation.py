"""
Person 3 – Task A: Independent Evaluation Verification
=======================================================
Independently recalculates MAE, RMSE, R², and data-leakage checks
from row-level baseline predictions WITHOUT reading Person 2 metric CSVs.

Run from the repository root:
    python src/person3_robustness_validation.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REPO_ROOT = Path(".")
DATA_PATH = REPO_ROOT / "data/processed/groundwater_rainfall_features_qc.csv"
PRED_PATH = REPO_ROOT / "results/baseline_predictions.csv"
BASELINE_METRICS_PATH = REPO_ROOT / "results/baseline_metrics.csv"
OUT_DIR = REPO_ROOT / "results/person3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CUT = "2018-01-01"
RANDOM_SEED = 42

TARGET = "Groundwater Level Quarterly Manual (meter)"
FEATURES = [
    "Latitude", "Longitude", "rainfall_30d_mm",
    "rainfall_90d_mm", "previous_gwl", "month_sin", "month_cos",
]

print("=" * 60)
print("PERSON 3 – INDEPENDENT EVALUATION VERIFICATION")
print("=" * 60)

# ── 1. Load processed dataset ──────────────────────────────────
df_raw = pd.read_csv(DATA_PATH, parse_dates=["observation_date"])
df_raw["well_id"] = (
    df_raw["Station"].astype(str) + "_" +
    df_raw["Latitude"].astype(str) + "_" +
    df_raw["Longitude"].astype(str)
)
df_raw = df_raw.sort_values(["well_id", "observation_date"])
df_raw["previous_gwl"] = df_raw.groupby("well_id")[TARGET].shift(1)
month = df_raw["observation_date"].dt.month
df_raw["month_sin"] = np.sin(2 * np.pi * month / 12)
df_raw["month_cos"] = np.cos(2 * np.pi * month / 12)
df_raw = df_raw.dropna(subset=FEATURES + [TARGET])

train_df = df_raw[df_raw["observation_date"] < TRAIN_CUT].copy()
test_df  = df_raw[df_raw["observation_date"] >= TRAIN_CUT].copy()

print(f"\nDataset: {len(df_raw):,} total rows after feature construction")
print(f"Training rows: {len(train_df):,}")
print(f"Test rows    : {len(test_df):,}")

# ── 2. Load predictions ─────────────────────────────────────────
pred = pd.read_csv(PRED_PATH, parse_dates=["date"])
print(f"\nPredictions file rows: {len(pred):,}")
print(f"Prediction columns   : {list(pred.columns)}")

# ── 3. Data quality checks ──────────────────────────────────────
checks = []

# 3a. Duplicate well-date pairs in predictions
dup_pred = pred.duplicated(subset=["well_id", "date"]).sum()
checks.append({"check": "duplicate well-date pairs in predictions",
               "value": int(dup_pred), "pass": dup_pred == 0})
print(f"\nDuplicate well-date pairs in predictions: {dup_pred}")

# 3b. Missing predictions
missing_rf = pred["random_forest_prediction_m"].isna().sum()
missing_pers = pred["persistence_prediction_m"].isna().sum()
checks.append({"check": "missing RF predictions",
               "value": int(missing_rf), "pass": missing_rf == 0})
checks.append({"check": "missing persistence predictions",
               "value": int(missing_pers), "pass": missing_pers == 0})
print(f"Missing RF predictions        : {missing_rf}")
print(f"Missing persistence predictions: {missing_pers}")

# 3c. Prediction date range – all in test window?
pred_date_min = pred["date"].min()
pred_date_max = pred["date"].max()
all_in_test = pred["date"].min() >= pd.Timestamp(TRAIN_CUT)
checks.append({"check": "all prediction dates in test window",
               "value": str(all_in_test), "pass": bool(all_in_test)})
print(f"\nPrediction date range: {pred_date_min.date()} to {pred_date_max.date()}")
print(f"All predictions in test window (>= 2018-01-01): {all_in_test}")

# 3d. No training dates in prediction file
train_dates_in_pred = (pred["date"] < pd.Timestamp(TRAIN_CUT)).sum()
checks.append({"check": "no training dates in prediction file",
               "value": int(train_dates_in_pred), "pass": train_dates_in_pred == 0})
print(f"Training dates present in prediction file: {train_dates_in_pred}")

# 3e. Merge predictions back to test features for alignment verification
merged = test_df.merge(
    pred, left_on=["observation_date", "well_id"],
    right_on=["date", "well_id"], how="inner"
)
print(f"\nTest rows in features file: {len(test_df):,}")
print(f"Rows matched to predictions: {len(merged):,}")
unmatched_test = len(test_df) - len(merged)
checks.append({"check": "test rows with no matching prediction",
               "value": int(unmatched_test), "pass": unmatched_test == 0})

# 3f. Check that actual GWL in predictions agrees with features file
if len(merged) > 0:
    diff = (merged[TARGET] - merged["actual_gwl_m"]).abs()
    max_diff = diff.max()
    large_diffs = (diff > 1e-6).sum()
    checks.append({"check": "actual GWL matches between sources",
                   "value": f"max_diff={max_diff:.2e}", "pass": large_diffs == 0})
    print(f"Max difference in actual GWL between sources: {max_diff:.2e} m")
    print(f"Rows with discrepancy > 1e-6: {large_diffs}")

# 3g. Training/test overlap (same well+date in both splits)
train_keys = set(zip(train_df["well_id"], train_df["observation_date"].astype(str)))
test_keys  = set(zip(test_df["well_id"],  test_df["observation_date"].astype(str)))
overlap = len(train_keys & test_keys)
checks.append({"check": "training-test observation overlap",
               "value": int(overlap), "pass": overlap == 0})
print(f"\nTraining-test observation overlap: {overlap}")

# ── 4. Independent metric calculation ───────────────────────────
actual     = merged[TARGET].to_numpy()
rf_pred    = merged["random_forest_prediction_m"].to_numpy()
pers_pred  = merged["persistence_prediction_m"].to_numpy()

def calc_metrics(name, actual_arr, pred_arr):
    mae  = mean_absolute_error(actual_arr, pred_arr)
    rmse = np.sqrt(mean_squared_error(actual_arr, pred_arr))
    r2   = r2_score(actual_arr, pred_arr)
    return {"model": name, "n_rows": len(actual_arr),
            "MAE_m": mae, "RMSE_m": rmse, "R2": r2}

indep_metrics = pd.DataFrame([
    calc_metrics("Persistence baseline", actual, pers_pred),
    calc_metrics("Random Forest",        actual, rf_pred),
])
print("\n── INDEPENDENTLY CALCULATED METRICS ──")
print(indep_metrics.to_string(index=False))

# ── 5. Compare with reported metrics ────────────────────────────
rep = pd.read_csv(BASELINE_METRICS_PATH)
print("\n── REPORTED METRICS (from baseline_metrics.csv) ──")
print(rep.to_string(index=False))

tol = 0.001  # 0.001 m tolerance for floating point differences
for _, row in indep_metrics.iterrows():
    rep_row = rep[rep["model"] == row["model"]]
    if rep_row.empty:
        print(f"WARNING: Model '{row['model']}' not found in reported metrics.")
        continue
    for metric in ["MAE_m", "RMSE_m"]:
        col_rep = metric if metric in rep_row.columns else metric.replace("_m", "")
        if col_rep not in rep_row.columns:
            continue
        diff = abs(row[metric] - rep_row[col_rep].values[0])
        status = "PASS" if diff < tol else "FAIL"
        print(f"  {row['model']} {metric}: indep={row[metric]:.6f}, "
              f"reported={rep_row[col_rep].values[0]:.6f}, diff={diff:.2e} [{status}]")
        checks.append({"check": f"metric agreement {row['model']} {metric}",
                       "value": f"diff={diff:.2e}", "pass": diff < tol})

# ── 6. Save outputs ─────────────────────────────────────────────
indep_metrics.to_csv(OUT_DIR / "verified_metrics.csv", index=False)
pd.DataFrame(checks).to_csv(OUT_DIR / "validation_checks.csv", index=False)

print(f"\n── VALIDATION CHECKS SUMMARY ──")
checks_df = pd.DataFrame(checks)
n_pass = checks_df["pass"].sum()
n_fail = (~checks_df["pass"]).sum()
print(f"PASS: {n_pass}  FAIL: {n_fail}")
if n_fail > 0:
    print("Failed checks:")
    print(checks_df[~checks_df["pass"]].to_string(index=False))

print(f"\nSaved: results/person3/verified_metrics.csv")
print(f"Saved: results/person3/validation_checks.csv")
print("=" * 60)
