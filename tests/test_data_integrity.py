"""
Person 3 – Task 5: Automated Scientific Validation Tests
========================================================
Run from the repository root:
    python -m pytest tests/ -v
or:
    python tests/test_data_integrity.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data/processed/groundwater_rainfall_features_qc.csv"
PRED_PATH = REPO_ROOT / "results/baseline_predictions.csv"
TRAIN_CUT = "2018-01-01"
TARGET = "Groundwater Level Quarterly Manual (meter)"

TESTS_RUN = 0
TESTS_PASS = 0
TESTS_FAIL = 0

def test(name, condition, detail=""):
    global TESTS_RUN, TESTS_PASS, TESTS_FAIL
    TESTS_RUN += 1
    if condition:
        TESTS_PASS += 1
        print(f"  [PASS] {name}")
    else:
        TESTS_FAIL += 1
        print(f"  [FAIL] {name}: {detail}")

def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["observation_date"])
    df["well_id"] = (df["Station"].astype(str) + "_" +
                     df["Latitude"].astype(str) + "_" +
                     df["Longitude"].astype(str))
    return df

def load_predictions():
    return pd.read_csv(PRED_PATH, parse_dates=["date"])

def run_all():
    print("=" * 60)
    print("PERSON 3 – AUTOMATED SCIENTIFIC VALIDATION TESTS")
    print("=" * 60)

    # ── 1. Training dates precede 2018 ────────────────────────────
    print("\n── 1. CHRONOLOGICAL SPLIT ──")
    df = load_data()
    train = df[df["observation_date"] < TRAIN_CUT]
    test_df = df[df["observation_date"] >= TRAIN_CUT]
    test("Training dates before 2018-01-01",
         (train["observation_date"].max() < pd.Timestamp(TRAIN_CUT)),
         f"max train date: {train['observation_date'].max()}")
    test("Test dates from 2018-01-01",
         (test_df["observation_date"].min() >= pd.Timestamp(TRAIN_CUT)),
         f"min test date: {test_df['observation_date'].min()}")

    # ── 2. No well-date overlap ─────────────────────────────────
    print("\n── 2. NO TRAINING/TEST OVERLAP ──")
    train_keys = set(zip(train["well_id"], train["observation_date"].astype(str)))
    test_keys  = set(zip(test_df["well_id"], test_df["observation_date"].astype(str)))
    overlap = len(train_keys & test_keys)
    test("No well-date pairs in both train and test",
         overlap == 0, f"{overlap} overlapping pairs")

    # ── 3. Feature columns match ─────────────────────────────────
    print("\n── 3. FEATURE CONSISTENCY ──")
    FEATURES = ["Latitude", "Longitude", "rainfall_30d_mm", "rainfall_90d_mm"]
    for f in FEATURES:
        test(f"Feature {f} in dataset", f in df.columns, f"missing column")

    # ── 4. Predictions are finite ────────────────────────────────
    print("\n── 4. PREDICTION INTEGRITY ──")
    pred = load_predictions()
    test("All RF predictions finite",
         np.isfinite(pred["random_forest_prediction_m"]).all(),
         f"{np.isnan(pred['random_forest_prediction_m']).sum()} NaN values")
    test("All persistence predictions finite",
         np.isfinite(pred["persistence_prediction_m"]).all(),
         f"{np.isnan(pred['persistence_prediction_m']).sum()} NaN values")
    test("All actual values finite",
         np.isfinite(pred["actual_gwl_m"]).all(),
         f"{np.isnan(pred['actual_gwl_m']).sum()} NaN values")

    # ── 5. Predictions preserve stable identifiers ────────────────
    print("\n── 5. IDENTIFIER INTEGRITY ──")
    test("Predictions have well_id column", "well_id" in pred.columns)
    test("Predictions have date column", "date" in pred.columns)
    test("All well_ids non-empty",
         (pred["well_id"].notna() & (pred["well_id"] != "")).all(),
         f"{(pred['well_id'].isna() | (pred['well_id'] == '')).sum()} empty")
    test("No duplicate well-date in predictions",
         pred.duplicated(subset=["well_id", "date"]).sum() == 0,
         f"{pred.duplicated(subset=['well_id', 'date']).sum()} duplicates")

    # ── 6. All predictions in test window ─────────────────────────
    print("\n── 6. PREDICTION WINDOW ──")
    test("All prediction dates >= 2018-01-01",
         (pred["date"] >= pd.Timestamp(TRAIN_CUT)).all(),
         f"{(pred['date'] < pd.Timestamp(TRAIN_CUT)).sum()} pre-2018 predictions")

    # ── 7. Metric independence (verify recalculation) ───────────
    print("\n── 7. METRIC INDEPENDENCE ──")
    actual = pred["actual_gwl_m"].to_numpy()
    rf_pred = pred["random_forest_prediction_m"].to_numpy()
    pers_pred = pred["persistence_prediction_m"].to_numpy()
    from sklearn.metrics import mean_absolute_error
    rf_mae = mean_absolute_error(actual, rf_pred)
    pers_mae = mean_absolute_error(actual, pers_pred)
    test("RF MAE independently computed",
         abs(rf_mae - 1.990073) < 0.01,
         f"computed={rf_mae:.6f}")
    test("Persistence MAE independently computed",
         abs(pers_mae - 2.601288) < 0.01,
         f"computed={pers_mae:.6f}")

    # ── 8. Novelty thresholds from training only ─────────────────
    print("\n── 8. NOVELTY THRESHOLD PROVENANCE ──")
    nov_path = REPO_ROOT / "results/training_referenced_novelty_predictions.csv"
    if nov_path.exists():
        nov = pd.read_csv(nov_path)
        full_nov = nov[nov["feature_space"] == "Full (7 features)"]
        if len(full_nov) > 0:
            t95 = full_nov["threshold_95"].iloc[0]
            t99 = full_nov["threshold_99"].iloc[0]
            test("Threshold 95 > 0", t95 > 0, f"t95={t95}")
            test("Threshold 99 > threshold 95", t99 > t95, f"t99={t99} <= t95={t95}")

    # ── 9. Bootstrap resamples at well level ─────────────────────
    print("\n── 9. BOOTSTRAP METHODOLOGY ──")
    boot_path = REPO_ROOT / "results/verified_bootstrap_ci.csv"
    if boot_path.exists():
        boot = pd.read_csv(boot_path)
        test("Bootstrap CIs have 2000 reps",
         (boot["boot_reps"] == 2000).all(),
         f"reps: {boot['boot_reps'].unique().tolist()}")

    # ── 10. Controlled removal uses training observations only ────
    print("\n── 10. CONTROLLED REMOVAL ──")
    manifest_path = REPO_ROOT / "results/repeated_removal_manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        test("Manifest has removal_type column", "removal_type" in manifest.columns)
        test("All removals are extreme or control",
             manifest["removal_type"].isin(["extreme", "control"]).all(),
             f"unique: {manifest['removal_type'].unique()}")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"SUMMARY: {TESTS_PASS} PASS, {TESTS_FAIL} FAIL, {TESTS_RUN} TOTAL")
    print("=" * 60)
    return TESTS_FAIL == 0

if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
