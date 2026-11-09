"""
Person 3 – Task 3: Repeated Controlled Training-Data Removal
============================================================
Repeats the 3-model removal experiment across multiple random seeds.
For each seed:
  A. Full training model
  B. High-novelty-removed model (top 1% most novel training obs removed)
  C. Matched-control model (same count removed, matched by year)

Saves:
  - Per-seed metrics
  - Row-level predictions (all seeds)
  - Removal manifests
  - Support-change measurements

Run from the repository root:
    python src/repeated_controlled_removal.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REPO_ROOT   = Path(".")
DATA_PATH   = REPO_ROOT / "data/processed/groundwater_rainfall_features_qc.csv"
OUT_DIR     = REPO_ROOT / "results"
FIG_DIR     = REPO_ROOT / "results/figures"
TRAIN_CUT   = "2018-01-01"
TARGET      = "Groundwater Level Quarterly Manual (meter)"
FEATURES    = ["Latitude", "Longitude", "rainfall_30d_mm",
               "rainfall_90d_mm", "previous_gwl", "month_sin", "month_cos"]
K = 5
EXTREME_FRACTION = 0.01
N_SEEDS = 10  # Reduced from 20 for computational feasibility (each seed = 3 RF fits + support recompute)
MODEL_SETTINGS = {"n_estimators": 300, "min_samples_leaf": 3, "n_jobs": -1}

print("=" * 65)
print("PERSON 3 – REPEATED CONTROLLED TRAINING-DATA REMOVAL")
print("=" * 65)
print(f"Target seeds: {N_SEEDS}")

# ── Load data ──────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, parse_dates=["observation_date"])
df["well_id"] = (df["Station"].astype(str) + "_" +
                 df["Latitude"].astype(str) + "_" +
                 df["Longitude"].astype(str))
df = df.sort_values(["well_id", "observation_date"])
df["previous_gwl"] = df.groupby("well_id")[TARGET].shift(1)
month = df["observation_date"].dt.month
df["month_sin"] = np.sin(2 * np.pi * month / 12)
df["month_cos"] = np.cos(2 * np.pi * month / 12)
df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)

train_full = df[df["observation_date"] < TRAIN_CUT].copy().reset_index(drop=True)
test_df    = df[df["observation_date"] >= TRAIN_CUT].copy().reset_index(drop=True)

print(f"Training rows: {len(train_full):,} | Test rows: {len(test_df):,}")

# ── Calculate training novelty once (deterministic) ────────────
scaler_ref = StandardScaler()
train_scaled = scaler_ref.fit_transform(train_full[FEATURES])
nn = NearestNeighbors(n_neighbors=K + 1, n_jobs=-1)
nn.fit(train_scaled)
train_dists, _ = nn.kneighbors(train_scaled)
train_novelty = train_dists[:, 1:].mean(axis=1)
train_full["train_novelty"] = train_novelty

# Determine how many to remove
n_remove = int(len(train_full) * EXTREME_FRACTION)
print(f"Removal count per model: {n_remove} ({EXTREME_FRACTION*100:.0f}% of {len(train_full)})")

# ── Novelty groups for test (computed once) ─────────────────────
test_scaled = scaler_ref.transform(test_df[FEATURES])
nn_test = NearestNeighbors(n_neighbors=K, n_jobs=-1)
nn_test.fit(train_scaled)
test_dists, _ = nn_test.kneighbors(test_scaled)
test_novelty = test_dists.mean(axis=1)
t95 = np.quantile(train_novelty, 0.95)
t99 = np.quantile(train_novelty, 0.99)
test_groups = np.select(
    [test_novelty <= t95, test_novelty <= t99],
    ["Familiar", "Poorly represented"],
    default="Extreme novelty"
)
test_df["test_novelty"] = test_novelty
test_df["test_group"] = test_groups

all_metrics = []
all_predictions = []
all_manifests = []
all_support = []

# ── Run repetitions ────────────────────────────────────────────
for seed_idx, seed in enumerate(range(42, 42 + N_SEEDS)):
    print(f"\n--- Seed {seed} ({seed_idx+1}/{N_SEEDS}) ---")

    # A. Full model
    rf_full = RandomForestRegressor(random_state=seed, **MODEL_SETTINGS)
    rf_full.fit(train_full[FEATURES], train_full[TARGET])
    pred_full = rf_full.predict(test_df[FEATURES])

    # B. Extreme-removed: top n_remove by novelty
    extreme_idx = train_full.nlargest(n_remove, "train_novelty").index
    train_no_extreme = train_full.drop(index=extreme_idx).reset_index(drop=True)
    rf_extreme = RandomForestRegressor(random_state=seed, **MODEL_SETTINGS)
    rf_extreme.fit(train_no_extreme[FEATURES], train_no_extreme[TARGET])
    pred_extreme = rf_extreme.predict(test_df[FEATURES])

    # C. Matched-control: remove same count, matched by year
    rng = np.random.default_rng(seed)
    extreme_years = train_full.loc[extreme_idx, "observation_date"].dt.year
    control_idx = []
    for yr in extreme_years.unique():
        yr_extreme_count = (extreme_years == yr).sum()
        yr_pool = train_full[(train_full["observation_date"].dt.year == yr) &
                             (~train_full.index.isin(extreme_idx))].index
        if len(yr_pool) >= yr_extreme_count:
            sampled = rng.choice(yr_pool, size=yr_extreme_count, replace=False)
            control_idx.extend(sampled)
    control_idx = np.array(control_idx)[:n_remove]
    train_control = train_full.drop(index=control_idx).reset_index(drop=True)
    rf_control = RandomForestRegressor(random_state=seed, **MODEL_SETTINGS)
    rf_control.fit(train_control[FEATURES], train_control[TARGET])
    pred_control = rf_control.predict(test_df[FEATURES])

    # Metrics
    for model_name, preds in [("Full model", pred_full),
                                ("Extreme-removed model", pred_extreme),
                                ("Matched-control model", pred_control)]:
        actual = test_df[TARGET].to_numpy()
        all_metrics.append({
            "seed": seed,
            "model": model_name,
            "MAE_m": mean_absolute_error(actual, preds),
            "RMSE_m": np.sqrt(mean_squared_error(actual, preds)),
            "R2": r2_score(actual, preds),
        })
        # Per test group metrics
        for tg in ["All", "Familiar", "Poorly represented", "Extreme novelty"]:
            if tg == "All":
                mask = np.ones(len(test_df), dtype=bool)
            else:
                mask = test_df["test_group"].to_numpy() == tg
            if mask.sum() > 0:
                all_metrics.append({
                    "seed": seed,
                    "model": model_name,
                    "test_group": tg,
                    "MAE_m": mean_absolute_error(actual[mask], preds[mask]),
                    "RMSE_m": np.sqrt(mean_squared_error(actual[mask], preds[mask])),
                    "n_obs": int(mask.sum()),
                })

    # Save predictions for this seed
    pred_df = test_df[["well_id", "observation_date", TARGET, "test_group"]].copy()
    pred_df["seed"] = seed
    pred_df["pred_full"] = pred_full
    pred_df["pred_extreme_removed"] = pred_extreme
    pred_df["pred_control"] = pred_control
    pred_df["abs_err_full"] = np.abs(test_df[TARGET].to_numpy() - pred_full)
    pred_df["abs_err_extreme"] = np.abs(test_df[TARGET].to_numpy() - pred_extreme)
    pred_df["abs_err_control"] = np.abs(test_df[TARGET].to_numpy() - pred_control)
    all_predictions.append(pred_df)

    # Manifest
    all_manifests.append(pd.DataFrame({
        "seed": seed,
        "removed_row_index": np.concatenate([extreme_idx.to_numpy(),
                                                control_idx]),
        "removal_type": ["extreme"] * len(extreme_idx) + ["control"] * len(control_idx),
    }))

    # Support change: recompute test novelty after removal (extreme-removed only)
    scaler_b = StandardScaler()
    train_b_scaled = scaler_b.fit_transform(train_no_extreme[FEATURES])
    test_b_scaled = scaler_b.transform(test_df[FEATURES])
    nn_b = NearestNeighbors(n_neighbors=K, n_jobs=-1)
    nn_b.fit(train_b_scaled)
    dist_b, _ = nn_b.kneighbors(test_b_scaled)
    novelty_b = dist_b.mean(axis=1)

    for tg in ["All", "Familiar", "Poorly represented", "Extreme novelty"]:
        if tg == "All":
            mask = np.ones(len(test_df), dtype=bool)
        else:
            mask = test_df["test_group"].to_numpy() == tg
        all_support.append({
            "seed": seed,
            "test_group": tg,
            "mean_support_full": test_novelty[mask].mean(),
            "mean_support_removed": novelty_b[mask].mean(),
            "support_change_pct": ((novelty_b[mask].mean() - test_novelty[mask].mean())
                                   / test_novelty[mask].mean() * 100),
        })

    # Incremental save after each seed (survives timeouts)
    if seed_idx >= 2:
        pd.DataFrame(all_metrics).to_csv(OUT_DIR / "repeated_removal_metrics.csv", index=False)
        pd.concat(all_predictions, ignore_index=True).to_csv(OUT_DIR / "repeated_removal_predictions.csv", index=False)
        pd.concat(all_manifests, ignore_index=True).to_csv(OUT_DIR / "repeated_removal_manifest.csv", index=False)
        pd.DataFrame(all_support).to_csv(OUT_DIR / "repeated_removal_support.csv", index=False)

# ── Save outputs ───────────────────────────────────────────────
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv(OUT_DIR / "repeated_removal_metrics.csv", index=False)
print(f"\nSaved: results/repeated_removal_metrics.csv ({len(metrics_df)} rows)")

predictions_df = pd.concat(all_predictions, ignore_index=True)
predictions_df.to_csv(OUT_DIR / "repeated_removal_predictions.csv", index=False)
print(f"Saved: results/repeated_removal_predictions.csv ({len(predictions_df)} rows)")

manifest_df = pd.concat(all_manifests, ignore_index=True)
manifest_df.to_csv(OUT_DIR / "repeated_removal_manifest.csv", index=False)
print(f"Saved: results/repeated_removal_manifest.csv ({len(manifest_df)} rows)")

support_df = pd.DataFrame(all_support)
support_df.to_csv(OUT_DIR / "repeated_removal_support.csv", index=False)
print(f"Saved: results/repeated_removal_support.csv ({len(support_df)} rows)")

# ── Seed summary ────────────────────────────────────────────────
print("\n── SEED SUMMARY ──")
seed_summary = []
for seed in range(42, 42 + N_SEEDS):
    seed_metrics = metrics_df[(metrics_df["seed"] == seed) &
                              (metrics_df.get("test_group", pd.Series(dtype=str)).isna()
                               if "test_group" in metrics_df.columns else
                              ~metrics_df.index.isin([]))]
    # Simplified: just get overall MAE for each model
    seed_rows = metrics_df[metrics_df["seed"] == seed]
    for model_name in ["Full model", "Extreme-removed model", "Matched-control model"]:
        # Overall row (no test_group)
        overall = seed_rows[(seed_rows["model"] == model_name) &
                            (seed_rows.get("test_group", pd.Series([None])).isna()
                             if "test_group" in seed_rows.columns else
                            pd.isna(seed_rows.get("test_group", pd.Series([np.nan]))))]
        if len(overall) > 0:
            seed_summary.append({
                "seed": seed,
                "model": model_name,
                "MAE_m": overall["MAE_m"].values[0],
                "RMSE_m": overall["RMSE_m"].values[0],
            })

seed_summary_df = pd.DataFrame(seed_summary)
seed_summary_df.to_csv(OUT_DIR / "repeated_removal_seed_summary.csv", index=False)
print(f"Saved: results/repeated_removal_seed_summary.csv")

# Compute differences
print("\n── DIFFERENCES ACROSS SEEDS ──")
full_mae = metrics_df[(metrics_df["model"] == "Full model") &
                      (metrics_df.get("test_group", pd.Series(dtype=str)).isna()
                       if "test_group" in metrics_df.columns else True)]
ext_mae = metrics_df[(metrics_df["model"] == "Extreme-removed model") &
                     (metrics_df.get("test_group", pd.Series(dtype=str)).isna()
                      if "test_group" in metrics_df.columns else True)]

# Simplified: use seed_summary
full_vals = seed_summary_df[seed_summary_df["model"] == "Full model"]["MAE_m"].values
ext_vals  = seed_summary_df[seed_summary_df["model"] == "Extreme-removed model"]["MAE_m"].values
ctrl_vals = seed_summary_df[seed_summary_df["model"] == "Matched-control model"]["MAE_m"].values

diff_be = ext_vals - full_vals  # B - A
diff_bc = ext_vals - ctrl_vals  # B - C

print(f"  Full model MAE:           mean={full_vals.mean():.6f}, std={full_vals.std():.6f}")
print(f"  Extreme-removed MAE:       mean={ext_vals.mean():.6f}, std={ext_vals.std():.6f}")
print(f"  Control MAE:               mean={ctrl_vals.mean():.6f}, std={ctrl_vals.std():.6f}")
print(f"  B-A (extreme-full):         mean={diff_be.mean():.6f}, std={diff_be.std():.6f}")
print(f"  B-C (extreme-control):      mean={diff_bc.mean():.6f}, std={diff_bc.std():.6f}")
print(f"  Seeds completed: {N_SEEDS}")

# Bootstrap CI for B-A
rng_boot = np.random.default_rng(RANDOM_SEED := 42)
boot_ba = np.array([rng_boot.choice(diff_be, len(diff_be), replace=True).mean()
                    for _ in range(2000)])
ci_lo, ci_hi = np.percentile(boot_ba, [2.5, 97.5])
print(f"  B-A 95% CI: [{ci_lo:.6f}, {ci_hi:.6f}] m")
print(f"  CI includes zero: {ci_lo <= 0 <= ci_hi}")

# Save bootstrap
boot_df = pd.DataFrame([{
    "comparison": "Extreme-removed minus Full (B-A)",
    "mean_diff_m": diff_be.mean(),
    "std_diff_m": diff_be.std(),
    "ci_lo_m": ci_lo,
    "ci_hi_m": ci_hi,
    "n_seeds": N_SEEDS,
    "boot_reps": 2000,
    "ci_includes_zero": ci_lo <= 0 <= ci_hi,
}])
boot_df.to_csv(OUT_DIR / "repeated_removal_bootstrap.csv", index=False)
print(f"Saved: results/repeated_removal_bootstrap.csv")

print("\n" + "=" * 65)
print("REPEATED CONTROLLED REMOVAL COMPLETE")
print("=" * 65)
