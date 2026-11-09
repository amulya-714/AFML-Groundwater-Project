"""
Person 3 – Task 4: Temporal Predictive Stability Validation
============================================================
Validates existing temporal analyses and investigates:
  1. Year-wise MAE/RMSE
  2. Temporal error changes by novelty group (familiar vs unfamiliar)
  3. Conditional residual relationships
  4. Alternative explanations (well coverage, missingness)

Run from the repository root:
    python src/temporal_stability_validation.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT   = Path(".")
DATA_PATH   = REPO_ROOT / "data/processed/groundwater_rainfall_features_qc.csv"
PRED_PATH   = REPO_ROOT / "results/baseline_predictions.csv"
NOV_PATH    = REPO_ROOT / "results/training_referenced_novelty_predictions.csv"
OUT_DIR     = REPO_ROOT / "results"
FIG_DIR     = REPO_ROOT / "results/figures"
TRAIN_CUT   = "2018-01-01"
TARGET      = "Groundwater Level Quarterly Manual (meter)"

print("=" * 65)
print("PERSON 3 – TEMPORAL STABILITY VALIDATION")
print("=" * 65)

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
df = df.dropna(subset=["Latitude", "Longitude", "rainfall_30d_mm",
                        "rainfall_90d_mm", "previous_gwl", "month_sin",
                        "month_cos", TARGET])

train = df[df["observation_date"] < TRAIN_CUT].copy()
test  = df[df["observation_date"] >= TRAIN_CUT].copy()

pred = pd.read_csv(PRED_PATH, parse_dates=["date"])
test = test.merge(pred, left_on=["observation_date", "well_id"],
                  right_on=["date", "well_id"], how="inner").reset_index(drop=True)
test["abs_error"] = (test[TARGET] - test["random_forest_prediction_m"]).abs()
test["sq_error"]  = (test[TARGET] - test["random_forest_prediction_m"]) ** 2

# Load novelty scores
nov_df = pd.read_csv(NOV_PATH)
nov_full = nov_df[nov_df["feature_space"] == "Full (7 features)"][["well_id", "observation_date", "novelty_group"]].copy()
nov_full["observation_date"] = pd.to_datetime(nov_full["observation_date"])
test = test.merge(nov_full, on=["well_id", "observation_date"], how="left")
test["novelty_group"] = test["novelty_group"].fillna("Unknown")
test["year"] = test["observation_date"].dt.year

print(f"Test rows: {len(test):,}")
print(f"Novelty groups: {test['novelty_group'].value_counts().to_dict()}")

# ── 1. Year-wise MAE/RMSE ───────────────────────────────────────
print("\n── 1. YEAR-WISE ERROR ──")
yearly = test.groupby("year").agg(
    n_obs=("well_id", "size"),
    n_wells=("well_id", "nunique"),
    MAE_m=("abs_error", "mean"),
    RMSE_m=("sq_error", lambda x: np.sqrt(x.mean())),
    mean_signed_error_m=(TARGET, lambda x: (x - test.loc[x.index, "random_forest_prediction_m"]).mean()),
).reset_index()
yearly.to_csv(OUT_DIR / "temporal_validation_predictions.csv", index=False)
print(yearly.round(4).to_string(index=False))

# ── 2. Temporal error by novelty group ──────────────────────────
print("\n── 2. TEMPORAL ERROR BY NOVELTY GROUP ──")
nov_yearly = test.groupby(["year", "novelty_group"]).agg(
    n_obs=("well_id", "size"),
    n_wells=("well_id", "nunique"),
    MAE_m=("abs_error", "mean"),
    RMSE_m=("sq_error", lambda x: np.sqrt(x.mean())),
).reset_index()
nov_yearly.to_csv(OUT_DIR / "familiar_input_temporal_summary.csv", index=False)
print(nov_yearly.round(4).to_string(index=False))

# ── 3. Familiar-input temporal comparison (2018 vs 2020) ─────────
print("\n── 3. FAMILIAR-INPUT TEMPORAL COMPARISON (2018 vs 2020) ──")
# Compare 2018 vs 2020 for wells with observations in both years
# and within each novelty group
for group in ["Well-represented", "Sparse-support", "Very-low-support"]:
    g_test = test[(test["novelty_group"] == group)]
    if len(g_test) == 0:
        continue
    wells_2018 = g_test[g_test["year"] == 2018]["well_id"].unique()
    wells_2020 = g_test[g_test["year"] == 2020]["well_id"].unique()
    paired_wells = set(wells_2018) & set(wells_2020)
    if len(paired_wells) < 5:
        continue
    paired = g_test[g_test["well_id"].isin(paired_wells)]
    per_well = paired.groupby(["well_id", "year"])["abs_error"].mean().unstack()
    per_well = per_well.dropna(subset=[2018, 2020])
    diff = per_well[2020] - per_well[2018]
    # Bootstrap CI
    rng = np.random.default_rng(42)
    diffs = diff.to_numpy()
    boot = np.array([rng.choice(diffs, len(diffs), replace=True).mean()
                     for _ in range(2000)])
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    print(f"  {group:20s}: wells={len(per_well)}, "
          f"MAE_2018={per_well[2018].mean():.4f}, "
          f"MAE_2020={per_well[2020].mean():.4f}, "
          f"diff={diff.mean():.4f} [{ci_lo:.4f}, {ci_hi:.4f}] "
          f"{'(excl 0)' if not (ci_lo <= 0 <= ci_hi) else '(incl 0)'}")

# ── 4. Alternative explanations: coverage changes ───────────────
print("\n── 4. ALTERNATIVE EXPLANATIONS ──")
# Well coverage by year
coverage = test.groupby("year").agg(
    n_obs=("well_id", "size"),
    n_wells=("well_id", "nunique"),
    mean_rainfall_90d=("rainfall_90d_mm", "mean"),
    mean_previous_gwl=("previous_gwl", "mean"),
    mean_gwl=("Groundwater Level Quarterly Manual (meter)", "mean"),
).reset_index()
coverage.to_csv(OUT_DIR / "temporal_coverage_validation.csv", index=False)
print(coverage.round(2).to_string(index=False))

# Missingness by year
missing_by_year = test.groupby("year").agg(
    pct_missing_rainfall_30d=("rainfall_30d_mm", lambda x: x.isna().mean() * 100),
    pct_missing_rainfall_90d=("rainfall_90d_mm", lambda x: x.isna().mean() * 100),
).reset_index()
print("\n  Missingness by year:")
print(missing_by_year.round(2).to_string(index=False))

# ── 5. Temporal interaction summary ─────────────────────────────
print("\n── 5. TEMPORAL INTERACTION RESULTS ──")
# Read existing multivariate temporal coefficients
try:
    coef = pd.read_csv(OUT_DIR / "multivariate_temporal_coefficients.csv")
    # Extract interaction terms
    interaction_rows = coef[coef["term"].str.contains("year_2019|year_2020", na=False) &
                            coef["term"].str.contains(":", na=False)]
    if len(interaction_rows) > 0:
        print("  Existing interaction coefficients (from multivariate_temporal_coefficients.csv):")
        for _, row in interaction_rows.iterrows():
            sig = "***" if row["P>|z|"] < 0.001 else ("**" if row["P>|z|"] < 0.01 else ("*" if row["P>|z|"] < 0.05 else ""))
            print(f"    {row['term']:30s} coef={row['Coef.']:.4f} p={row['P>|z|']:.4e} {sig}")
    interaction_rows.to_csv(OUT_DIR / "temporal_interaction_results.csv", index=False)
    print(f"  Saved: results/temporal_interaction_results.csv")
except Exception as e:
    print(f"  Could not read existing coefficients: {e}")

# ── 6. Residual validation ──────────────────────────────────────
print("\n── 6. RESIDUAL VALIDATION ──")
test["residual"] = test[TARGET] - test["random_forest_prediction_m"]
resid_by_year = test.groupby("year").agg(
    mean_residual=("residual", "mean"),
    std_residual=("residual", "std"),
    median_residual=("residual", "median"),
    n_obs=("residual", "size"),
).reset_index()
resid_by_year.to_csv(OUT_DIR / "temporal_residual_validation.csv", index=False)
print(resid_by_year.round(4).to_string(index=False))

# ── 7. Relationship comparison: rainfall effect by year ─────────
print("\n── 7. CONDITIONAL RELATIONSHIP COMPARISON ──")
rel_rows = []
for yr in [2018, 2019, 2020]:
    sub = test[test["year"] == yr]
    if len(sub) < 50:
        continue
    # Correlation of residual with rainfall
    corr_30 = sub["residual"].corr(sub["rainfall_30d_mm"])
    corr_90 = sub["residual"].corr(sub["rainfall_90d_mm"])
    corr_prev = sub["residual"].corr(sub["previous_gwl"])
    rel_rows.append({
        "year": yr,
        "n_obs": len(sub),
        "corr_residual_rainfall_30d": corr_30,
        "corr_residual_rainfall_90d": corr_90,
        "corr_residual_previous_gwl": corr_prev,
        "mean_residual": sub["residual"].mean(),
        "std_residual": sub["residual"].std(),
    })
rel_df = pd.DataFrame(rel_rows)
rel_df.to_csv(OUT_DIR / "temporal_relationship_comparison.csv", index=False)
print(rel_df.round(4).to_string(index=False))

# ── 8. Figures ──────────────────────────────────────────────────
print("\n── 8. GENERATING FIGURES ──")

# Figure 8: Familiar-input temporal comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Temporal Error by Novelty Group (Person 3 Validation)", fontsize=13)

# Plot 1: MAE by year and novelty group
ax = axes[0]
for group in ["Well-represented", "Sparse-support", "Very-low-support"]:
    sub = nov_yearly[nov_yearly["novelty_group"] == group]
    if len(sub) > 0:
        ax.plot(sub["year"], sub["MAE_m"], marker="o", label=group, linewidth=2)
ax.set_xlabel("Year")
ax.set_ylabel("MAE (m)")
ax.set_title("MAE by Year and Novelty Group")
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Well coverage by year
ax = axes[1]
ax2 = ax.twinx()
bars = ax.bar(coverage["year"], coverage["n_wells"], color="steelblue", alpha=0.7, label="Wells")
ax.set_xlabel("Year")
ax.set_ylabel("Number of wells", color="steelblue")
ax2.plot(coverage["year"], coverage["mean_rainfall_90d"], color="tomato", marker="s",
         linewidth=2, label="Mean 90d rainfall (mm)")
ax2.set_ylabel("Mean 90d rainfall (mm)", color="tomato")
ax.set_title("Well Coverage and Rainfall by Year")
ax.legend(loc="upper left")
ax2.legend(loc="upper right")

plt.tight_layout()
plt.savefig(FIG_DIR / "figure8_familiar_input_temporal_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/figures/figure8_familiar_input_temporal_comparison.png")

# Figure 9: Temporal residuals
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Temporal Residual Analysis (Person 3 Validation)", fontsize=13)

# Plot 1: Residual distribution by year
ax = axes[0]
for yr in [2018, 2019, 2020]:
    sub = test[test["year"] == yr]
    ax.hist(sub["residual"], bins=50, alpha=0.4, density=True, label=str(yr))
ax.axvline(0, color="black", linestyle="--", linewidth=1)
ax.set_xlabel("Residual (m)")
ax.set_ylabel("Density")
ax.set_title("Residual Distribution by Year")
ax.legend(title="Year")

# Plot 2: Residual vs rainfall by year
ax = axes[1]
for yr, color in [(2018, "steelblue"), (2019, "orange"), (2020, "tomato")]:
    sub = test[test["year"] == yr].sample(min(2000, len(test[test["year"]==yr])), random_state=42)
    ax.scatter(sub["rainfall_90d_mm"], sub["residual"], alpha=0.2, s=10, color=color, label=str(yr))
ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.set_xlabel("90-day rainfall (mm)")
ax.set_ylabel("Residual (m)")
ax.set_title("Residual vs 90-day Rainfall by Year")
ax.legend(title="Year", markerscale=3)

plt.tight_layout()
plt.savefig(FIG_DIR / "figure9_temporal_residuals.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/figures/figure9_temporal_residuals.png")

print("\n" + "=" * 65)
print("TEMPORAL STABILITY VALIDATION COMPLETE")
print("=" * 65)
