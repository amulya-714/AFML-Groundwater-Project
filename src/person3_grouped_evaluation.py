"""
Person 3 – Task B: Grouped Evaluation
======================================
Independently validates:
  A. Familiar vs unfamiliar inputs (training percentile bounds)
  B. High rainfall vs other conditions (95th pctile of training 90d rainfall)
  C. Same-well comparisons (within-well paired differences)

Also generates comparison figures.

Run from the repository root:
    python src/person3_grouped_evaluation.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import mean_absolute_error, mean_squared_error

REPO_ROOT = Path(".")
DATA_PATH = REPO_ROOT / "data/processed/groundwater_rainfall_features_qc.csv"
PRED_PATH = REPO_ROOT / "results/baseline_predictions.csv"
OUT_DIR   = REPO_ROOT / "results/person3"
FIG_DIR   = REPO_ROOT / "results/figures/person3"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CUT  = "2018-01-01"
TARGET     = "Groundwater Level Quarterly Manual (meter)"
FEATURES   = ["rainfall_30d_mm", "rainfall_90d_mm", "previous_gwl"]

print("=" * 60)
print("PERSON 3 – GROUPED EVALUATION")
print("=" * 60)

# ── Load data ───────────────────────────────────────────────────
df_raw = pd.read_csv(DATA_PATH, parse_dates=["observation_date"])
df_raw["well_id"] = (
    df_raw["Station"].astype(str) + "_" +
    df_raw["Latitude"].astype(str) + "_" +
    df_raw["Longitude"].astype(str)
)
df_raw = df_raw.sort_values(["well_id", "observation_date"])
df_raw["previous_gwl"] = df_raw.groupby("well_id")[TARGET].shift(1)

pred = pd.read_csv(PRED_PATH, parse_dates=["date"])

# Split into train/test for threshold calculation
train_raw = df_raw[df_raw["observation_date"] < TRAIN_CUT].copy()
test_raw  = df_raw[df_raw["observation_date"] >= TRAIN_CUT].copy()

# Merge test rows with predictions
test = test_raw.merge(
    pred, left_on=["observation_date", "well_id"],
    right_on=["date", "well_id"], how="inner"
)
test["abs_error_rf"]   = (test[TARGET] - test["random_forest_prediction_m"]).abs()
test["sq_error_rf"]    = (test[TARGET] - test["random_forest_prediction_m"]) ** 2
test["abs_error_pers"] = (test[TARGET] - test["persistence_prediction_m"]).abs()

print(f"\nTest rows merged with predictions: {len(test):,}")

# ────────────────────────────────────────────────────────────────
# A. FAMILIAR vs UNFAMILIAR INPUTS
#    Definition: observations outside the training 5th–95th percentile
#    of rainfall_30d_mm OR rainfall_90d_mm are "unfamiliar".
#    We also compute a stricter version using 1st–99th percentiles.
# ────────────────────────────────────────────────────────────────
print("\n── A. FAMILIAR vs UNFAMILIAR INPUTS ──")

def compute_bounds(train_df, feature, lo_q=0.05, hi_q=0.95):
    lo = train_df[feature].quantile(lo_q)
    hi = train_df[feature].quantile(hi_q)
    return lo, hi

bounds_5_95   = {}
bounds_1_99   = {}
rain_features = ["rainfall_30d_mm", "rainfall_90d_mm"]
for feat in rain_features:
    lo5, hi5 = compute_bounds(train_raw, feat, 0.05, 0.95)
    lo1, hi1 = compute_bounds(train_raw, feat, 0.01, 0.99)
    bounds_5_95[feat] = (lo5, hi5)
    bounds_1_99[feat] = (lo1, hi1)
    print(f"  {feat}: 5th–95th = [{lo5:.2f}, {hi5:.2f}] mm | "
          f"1st–99th = [{lo1:.2f}, {hi1:.2f}] mm")

# Build unfamiliar flag (5th–95th, either rainfall feature)
test["unfamiliar_595"] = False
for feat, (lo, hi) in bounds_5_95.items():
    test["unfamiliar_595"] |= (test[feat] < lo) | (test[feat] > hi)

# Build unfamiliar flag (1st–99th, either rainfall feature)
test["unfamiliar_199"] = False
for feat, (lo, hi) in bounds_1_99.items():
    test["unfamiliar_199"] |= (test[feat] < lo) | (test[feat] > hi)

def group_metrics(df, mask, label):
    g = df[mask]
    if len(g) == 0:
        return {"group": label, "n_obs": 0, "n_wells": 0,
                "MAE_m": np.nan, "RMSE_m": np.nan}
    return {
        "group":  label,
        "n_obs":  len(g),
        "n_wells": g["well_id"].nunique(),
        "MAE_m":  g["abs_error_rf"].mean(),
        "RMSE_m": np.sqrt(g["sq_error_rf"].mean()),
    }

familiar_rows = []
for suffix, flag_col in [("5th-95th", "unfamiliar_595"),
                          ("1st-99th", "unfamiliar_199")]:
    familiar_rows.append(
        group_metrics(test, ~test[flag_col], f"Familiar ({suffix} bounds)")
    )
    familiar_rows.append(
        group_metrics(test,  test[flag_col], f"Unfamiliar ({suffix} bounds)")
    )

familiar_df = pd.DataFrame(familiar_rows)
print("\n", familiar_df.round(4).to_string(index=False))

# ────────────────────────────────────────────────────────────────
# B. HIGH RAINFALL vs OTHER CONDITIONS
#    Threshold = 95th percentile of training 90-day rainfall
# ────────────────────────────────────────────────────────────────
print("\n── B. HIGH RAINFALL vs OTHER CONDITIONS ──")

rain_cutoff = train_raw["rainfall_90d_mm"].quantile(0.95)
print(f"  Training 90d rainfall 95th percentile: {rain_cutoff:.2f} mm")
test["high_rainfall_95"] = test["rainfall_90d_mm"] >= rain_cutoff

rainfall_rows = []
for label, mask in [
    ("Below 95th pctile rainfall", ~test["high_rainfall_95"]),
    ("High rainfall (≥95th pctile)", test["high_rainfall_95"]),
]:
    rainfall_rows.append(group_metrics(test, mask, label))

rainfall_df = pd.DataFrame(rainfall_rows)
print("\n", rainfall_df.round(4).to_string(index=False))

# ────────────────────────────────────────────────────────────────
# C. SAME-WELL COMPARISONS
#    For wells that have observations in both high and low rainfall
#    conditions, compare per-well mean absolute error within wells.
# ────────────────────────────────────────────────────────────────
print("\n── C. SAME-WELL COMPARISONS ──")

# Per-well mean error by condition
well_cond = (
    test.groupby(["well_id", "high_rainfall_95"])["abs_error_rf"]
    .mean()
    .unstack()
)
well_cond.columns.name = None
well_cond = well_cond.rename(columns={False: "mae_normal", True: "mae_high_rain"})
eligible_wells = well_cond.dropna(subset=["mae_normal", "mae_high_rain"])
eligible_wells = eligible_wells.copy()
eligible_wells["paired_diff"] = eligible_wells["mae_high_rain"] - eligible_wells["mae_normal"]

n_eligible = len(eligible_wells)
mean_diff  = eligible_wells["paired_diff"].mean()
median_diff = eligible_wells["paired_diff"].median()
frac_higher = (eligible_wells["paired_diff"] > 0).mean()

print(f"\n  Wells with observations in both conditions: {n_eligible}")
print(f"  Mean paired MAE difference (high – normal): {mean_diff:.4f} m")
print(f"  Median paired MAE difference:               {median_diff:.4f} m")
print(f"  Fraction of wells with higher error in high rainfall: {frac_higher:.3f}")

# Quick 95% bootstrap CI (well-level)
rng   = np.random.default_rng(42)
diffs = eligible_wells["paired_diff"].to_numpy()
boot  = np.array([rng.choice(diffs, len(diffs), replace=True).mean()
                  for _ in range(2000)])
ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
print(f"  Bootstrap 95% CI (2000 reps): [{ci_lo:.4f}, {ci_hi:.4f}] m")
includes_zero = ci_lo <= 0 <= ci_hi
print(f"  CI includes zero: {includes_zero}")

same_well_df = pd.DataFrame([{
    "n_eligible_wells": n_eligible,
    "mean_paired_diff_m": mean_diff,
    "median_paired_diff_m": median_diff,
    "frac_wells_higher_error_in_high_rain": frac_higher,
    "boot_ci_lo_m": ci_lo,
    "boot_ci_hi_m": ci_hi,
    "boot_reps": 2000,
    "ci_includes_zero": includes_zero,
}])

# ────────────────────────────────────────────────────────────────
# Combine into one grouped metrics CSV
# ────────────────────────────────────────────────────────────────
all_groups = pd.concat([
    familiar_df.assign(analysis="familiar_unfamiliar"),
    rainfall_df.assign(analysis="high_rainfall"),
], ignore_index=True)
all_groups.to_csv(OUT_DIR / "verified_group_metrics.csv", index=False)
eligible_wells.reset_index().to_csv(OUT_DIR / "same_well_per_well_differences.csv", index=False)
same_well_df.to_csv(OUT_DIR / "same_well_summary.csv", index=False)

print(f"\nSaved: results/person3/verified_group_metrics.csv")
print(f"Saved: results/person3/same_well_per_well_differences.csv")
print(f"Saved: results/person3/same_well_summary.csv")

# ────────────────────────────────────────────────────────────────
# FIGURES
# ────────────────────────────────────────────────────────────────

# Figure: Familiar vs unfamiliar comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Familiar vs Unfamiliar Inputs (Person 3 Verification)", fontsize=13)

for ax_idx, (suffix, flag_col) in enumerate([("5th–95th bounds", "unfamiliar_595"),
                                               ("1st–99th bounds", "unfamiliar_199")]):
    sub = familiar_df[familiar_df["group"].str.contains(suffix.replace("–", "-"))]
    if sub.empty:
        continue
    ax = axes[ax_idx]
    ax.bar(["Familiar", "Unfamiliar"], sub["MAE_m"].values,
           color=["steelblue", "tomato"], alpha=0.85, edgecolor="black")
    ax.set_title(f"MAE by Support ({suffix})")
    ax.set_ylabel("MAE (m)")
    ax.set_xlabel("Input Support Group")
    for i, (_, row) in enumerate(sub.iterrows()):
        ax.text(i, row["MAE_m"] + 0.03, f"n={row['n_obs']:,}\nwells={row['n_wells']:,}",
                ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(FIG_DIR / "familiar_unfamiliar_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/figures/person3/familiar_unfamiliar_comparison.png")

# Figure: Rainfall comparison
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(rainfall_df["group"], rainfall_df["MAE_m"],
       color=["steelblue", "tomato"], alpha=0.85, edgecolor="black")
ax.set_title(f"MAE by 90-day Rainfall Condition\n(cutoff={rain_cutoff:.1f} mm, 95th pctile training)")
ax.set_ylabel("MAE (m)")
ax.set_xlabel("Condition")
for i, row in rainfall_df.iterrows():
    ax.text(i, row["MAE_m"] + 0.03,
            f"n={row['n_obs']:,}\nwells={row['n_wells']:,}",
            ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "rainfall_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/figures/person3/rainfall_comparison.png")

# Figure: Same-well paired differences histogram
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(eligible_wells["paired_diff"], bins=40, color="steelblue",
        edgecolor="white", alpha=0.85)
ax.axvline(mean_diff, color="tomato", linewidth=2, label=f"Mean diff = {mean_diff:.3f} m")
ax.axvline(0, color="black", linewidth=1, linestyle="--", label="Zero line")
ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 50],
                  ci_lo, ci_hi, alpha=0.15, color="tomato", label=f"95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
ax.set_title(f"Same-Well Paired MAE Differences (High – Normal Rainfall)\nn={n_eligible} wells")
ax.set_xlabel("MAE difference (m)")
ax.set_ylabel("Number of wells")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "same_well_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/figures/person3/same_well_comparison.png")

# Figure: Baseline model comparison (RF vs Persistence)
baseline_metrics = pd.read_csv(REPO_ROOT / "results/baseline_metrics.csv")
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.suptitle("Baseline Model Comparison (Person 3 Verification)", fontsize=13)
metrics_to_plot = [("MAE_m", "MAE (m)"), ("RMSE_m", "RMSE (m)"), ("R2", "R²")]
for ax, (col, ylabel) in zip(axes, metrics_to_plot):
    vals = baseline_metrics[col].values
    models = baseline_metrics["model"].values
    colors = ["steelblue", "tomato"] if len(vals) == 2 else ["steelblue"] * len(vals)
    ax.bar(models, vals, color=colors, alpha=0.85, edgecolor="black")
    ax.set_title(ylabel)
    ax.set_ylabel(ylabel)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01 * max(vals), f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "baseline_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/figures/person3/baseline_comparison.png")

print("\n" + "=" * 60)
print("GROUPED EVALUATION COMPLETE")
print("=" * 60)
