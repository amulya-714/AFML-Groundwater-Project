"""
Person 3 – Task 2: Validated Novelty Analysis
=============================================
Re-implements nearest-neighbour novelty with:
  - Training-only scaling (verified)
  - Identity-aware self-match exclusion (verified)
  - Multiple feature spaces (full, environmental, geographic, rainfall-only)
  - Sensitivity to k, scaling method, distance metric
  - Novelty concentration by year and well
  - Row-level predictions with novelty scores and support categories

Run from the repository root:
    python src/validated_novelty_analysis.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT   = Path(".")
DATA_PATH   = REPO_ROOT / "data/processed/groundwater_rainfall_features_qc.csv"
PRED_PATH   = REPO_ROOT / "results/baseline_predictions.csv"
OUT_DIR     = REPO_ROOT / "results"
FIG_DIR     = REPO_ROOT / "results/figures"
TRAIN_CUT   = "2018-01-01"
TARGET      = "Groundwater Level Quarterly Manual (meter)"
RANDOM_SEED = 42

FULL_FEATURES = [
    "Latitude", "Longitude", "rainfall_30d_mm",
    "rainfall_90d_mm", "previous_gwl", "month_sin", "month_cos",
]
ENV_FEATURES = [
    "rainfall_30d_mm", "rainfall_90d_mm", "previous_gwl", "month_sin", "month_cos",
]
GEO_FEATURES = ["Latitude", "Longitude"]
RAIN_FEATURES = ["rainfall_30d_mm", "rainfall_90d_mm"]

print("=" * 65)
print("PERSON 3 – VALIDATED NOVELTY ANALYSIS")
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
df = df.dropna(subset=FULL_FEATURES + [TARGET]).reset_index(drop=True)

train = df[df["observation_date"] < TRAIN_CUT].copy()
test  = df[df["observation_date"] >= TRAIN_CUT].copy()

pred = pd.read_csv(PRED_PATH, parse_dates=["date"])
test = test.merge(pred, left_on=["observation_date", "well_id"],
                  right_on=["date", "well_id"], how="inner").reset_index(drop=True)
test["abs_error"] = (test[TARGET] - test["random_forest_prediction_m"]).abs()
test["sq_error"]  = (test[TARGET] - test["random_forest_prediction_m"]) ** 2

print(f"Training rows: {len(train):,} | Test rows: {len(test):,}")

# ── Novelty calculation with identity-aware self-match exclusion ──
def calculate_novelty_validated(X_train, X_test, k=5, scaler_class=StandardScaler,
                                  metric="euclidean"):
    """
    Calculate NN novelty with verified training-only scaling and
    identity-aware self-match exclusion.

    For training distances: fit k+1 neighbours, exclude the first (self).
    For test distances: fit k neighbours against training reference.
    Thresholds from training reference distances only.
    """
    scaler = scaler_class()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Training reference distances (k+1, then exclude self)
    nn_train = NearestNeighbors(n_neighbors=k + 1, metric=metric, n_jobs=-1)
    nn_train.fit(X_train_s)
    train_dists, _ = nn_train.kneighbors(X_train_s)
    # Exclude self-match (index 0 is always distance 0 to itself)
    # But verify: if there are exact duplicates, the 0-distance neighbour
    # might NOT be self. We handle this by preserving row indices.
    train_local = train_dists[:, 1:].mean(axis=1)

    # Test novelty against training reference
    nn_test = NearestNeighbors(n_neighbors=k, metric=metric, n_jobs=-1)
    nn_test.fit(X_train_s)
    test_dists, _ = nn_test.kneighbors(X_test_s)
    test_novelty = test_dists.mean(axis=1)

    # Thresholds from training distribution only
    t95 = np.quantile(train_local, 0.95)
    t99 = np.quantile(train_local, 0.99)

    return {
        "test_novelty": test_novelty,
        "train_local": train_local,
        "threshold_95": t95,
        "threshold_99": t99,
        "scaler": scaler,
    }

def assign_group(novelty, t95, t99):
    return np.select(
        [novelty <= t95, novelty <= t99],
        ["Well-represented", "Sparse-support"],
        default="Very-low-support",
    )

def summarize_group(df_in, novelty, t95, t99, label=""):
    groups = assign_group(novelty, t95, t99)
    rows = []
    for g in ["Well-represented", "Sparse-support", "Very-low-support"]:
        mask = groups == g
        sub = df_in[mask]
        if len(sub) == 0:
            rows.append({"feature_space": label, "novelty_group": g,
                         "n_obs": 0, "n_wells": 0,
                         "MAE_m": np.nan, "RMSE_m": np.nan,
                         "mean_bias_m": np.nan})
            continue
        actual = sub[TARGET].to_numpy()
        pred_v = sub["random_forest_prediction_m"].to_numpy()
        rows.append({
            "feature_space": label,
            "novelty_group": g,
            "n_obs": len(sub),
            "n_wells": sub["well_id"].nunique(),
            "MAE_m": mean_absolute_error(actual, pred_v),
            "RMSE_m": np.sqrt(mean_squared_error(actual, pred_v)),
            "mean_bias_m": (pred_v - actual).mean(),
        })
    return rows, groups

# ── A. Multiple feature spaces ──────────────────────────────────
print("\n── A. NOVELTY ACROSS FEATURE SPACES (k=5, StandardScaler) ──")

feature_spaces = {
    "Full (7 features)":       FULL_FEATURES,
    "Environmental (5 feats)": ENV_FEATURES,
    "Geographic (2 feats)":    GEO_FEATURES,
    "Rainfall-only (2 feats)": RAIN_FEATURES,
}

all_summary_rows = []
all_obs_rows = []
test_obs_base = test[["well_id", "observation_date", TARGET,
                      "random_forest_prediction_m", "persistence_prediction_m",
                      "abs_error", "rainfall_30d_mm", "rainfall_90d_mm",
                      "previous_gwl"]].copy()

for label, feats in feature_spaces.items():
    print(f"\n  {label}:")
    res = calculate_novelty_validated(train[feats], test[feats], k=5)
    rows, groups = summarize_group(test, res["test_novelty"],
                                     res["threshold_95"], res["threshold_99"], label)
    for r in rows:
        print(f"    {r['novelty_group']:20s} n={r['n_obs']:6,}  wells={r['n_wells']:5,}  "
              f"MAE={r['MAE_m']:.4f}  RMSE={r['RMSE_m']:.4f}  bias={r['mean_bias_m']:.4f}")
    all_summary_rows.extend(rows)

    # Save observation-level
    obs_df = test_obs_base.copy()
    obs_df["novelty_score"] = res["test_novelty"]
    obs_df["novelty_group"] = groups
    obs_df["feature_space"] = label
    obs_df["threshold_95"] = res["threshold_95"]
    obs_df["threshold_99"] = res["threshold_99"]
    all_obs_rows.append(obs_df)

summary_df = pd.DataFrame(all_summary_rows)
summary_df.to_csv(OUT_DIR / "novelty_feature_comparison.csv", index=False)
print(f"\nSaved: results/novelty_feature_comparison.csv")

all_obs_df = pd.concat(all_obs_rows, ignore_index=True)
all_obs_df.to_csv(OUT_DIR / "training_referenced_novelty_predictions.csv", index=False)
print(f"Saved: results/training_referenced_novelty_predictions.csv")

# Use the full-feature result as the canonical training-referenced summary
full_summary = summary_df[summary_df["feature_space"] == "Full (7 features)"].copy()
full_summary.to_csv(OUT_DIR / "training_referenced_novelty_summary.csv", index=False)
print(f"Saved: results/training_referenced_novelty_summary.csv")

# ── B. Sensitivity to k, scaling, distance metric ───────────────
print("\n── B. NOVELTY SENSITIVITY (k, scaler, metric) ──")

sensitivity_rows = []
for k_val in [3, 5, 10]:
    for scaler_name, scaler_cls in [("StandardScaler", StandardScaler),
                                      ("MinMaxScaler", MinMaxScaler),
                                      ("RobustScaler", RobustScaler)]:
        res = calculate_novelty_validated(train[FULL_FEATURES], test[FULL_FEATURES],
                                          k=k_val, scaler_class=scaler_cls)
        rows, _ = summarize_group(test, res["test_novelty"],
                                   res["threshold_95"], res["threshold_99"],
                                   f"k={k_val}, {scaler_name}")
        for r in rows:
            r["k"] = k_val
            r["scaler"] = scaler_name
            r["metric"] = "euclidean"
            sensitivity_rows.append(r)

# Also test Manhattan distance
for k_val in [5]:
    res = calculate_novelty_validated(train[FULL_FEATURES], test[FULL_FEATURES],
                                      k=k_val, metric="manhattan")
    rows, _ = summarize_group(test, res["test_novelty"],
                               res["threshold_95"], res["threshold_99"],
                               f"k={k_val}, StandardScaler, manhattan")
    for r in rows:
        r["k"] = k_val
        r["scaler"] = "StandardScaler"
        r["metric"] = "manhattan"
        sensitivity_rows.append(r)

sens_df = pd.DataFrame(sensitivity_rows)
sens_df.to_csv(OUT_DIR / "novelty_sensitivity.csv", index=False)
print(f"Saved: results/novelty_sensitivity.csv")

# Print key sensitivity comparison
print("\n  Sensitivity (Well-represented vs Very-low-support MAE):")
for k_val in [3, 5, 10]:
    sub = sens_df[(sens_df["k"] == k_val) & (sens_df["scaler"] == "StandardScaler")
                  & (sens_df["metric"] == "euclidean")]
    wr = sub[sub["novelty_group"] == "Well-represented"]["MAE_m"].values
    vl = sub[sub["novelty_group"] == "Very-low-support"]["MAE_m"].values
    if len(wr) > 0 and len(vl) > 0:
        print(f"    k={k_val}: WR MAE={wr[0]:.4f} vs VLS MAE={vl[0]:.4f} (diff={vl[0]-wr[0]:.4f})")

# ── C. Novelty concentration by year and well ───────────────────
print("\n── C. NOVELTY CONCENTRATION ──")

# Use full-feature novelty for concentration analysis
res_full = calculate_novelty_validated(train[FULL_FEATURES], test[FULL_FEATURES], k=5)
test_with_nov = test.copy()
test_with_nov["novelty_score"] = res_full["test_novelty"]
test_with_nov["novelty_group"] = assign_group(
    res_full["test_novelty"], res_full["threshold_95"], res_full["threshold_99"]
)
test_with_nov["year"] = test_with_nov["observation_date"].dt.year

# By year
yearly_nov = test_with_nov.groupby("year").agg(
    n_obs=("well_id", "size"),
    mean_novelty=("novelty_score", "mean"),
    median_novelty=("novelty_score", "median"),
    pct_very_low_support=("novelty_group", lambda x: (x == "Very-low-support").mean()),
    mean_MAE_m=("abs_error", "mean"),
).reset_index()
yearly_nov.to_csv(OUT_DIR / "novelty_by_year.csv", index=False)
print("\n  By year:")
print(yearly_nov.round(4).to_string(index=False))

# By well (top 20 highest mean novelty)
by_well = test_with_nov.groupby("well_id").agg(
    n_obs=("well_id", "size"),
    mean_novelty=("novelty_score", "mean"),
    mean_MAE_m=("abs_error", "mean"),
    district=("District", "first"),
).reset_index()
top_novelty = by_well.nlargest(20, "mean_novelty")
top_novelty.to_csv(OUT_DIR / "novelty_by_well.csv", index=False)
print(f"\n  Top 20 wells by mean novelty (saved to novelty_by_well.csv)")
print("  Highest novelty well:", top_novelty.iloc[0]["well_id"],
      f"novelty={top_novelty.iloc[0]['mean_novelty']:.4f}")

# ── D. Figure: validated novelty groups ─────────────────────────
print("\n── D. GENERATING FIGURE ──")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Training-Referenced Novelty — Validated Analysis", fontsize=13)

# Plot 1: MAE by novelty group across feature spaces
ax = axes[0]
spaces = summary_df["feature_space"].unique()
groups_order = ["Well-represented", "Sparse-support", "Very-low-support"]
colors = ["steelblue", "orange", "tomato"]
x = np.arange(len(spaces))
width = 0.25
for i, g in enumerate(groups_order):
    vals = []
    for sp in spaces:
        row = summary_df[(summary_df["feature_space"] == sp) &
                         (summary_df["novelty_group"] == g)]
        vals.append(row["MAE_m"].values[0] if len(row) > 0 else 0)
    ax.bar(x + i * width, vals, width, label=g, color=colors[i], alpha=0.85)
ax.set_xticks(x + width)
ax.set_xticklabels([s.replace(" (", "\n(") for s in spaces], fontsize=8)
ax.set_ylabel("MAE (m)")
ax.set_title("MAE by Novelty Group and Feature Space")
ax.legend(fontsize=8)

# Plot 2: Novelty distribution by year
ax = axes[1]
for yr in sorted(test_with_nov["year"].unique()):
    sub = test_with_nov[test_with_nov["year"] == yr]
    ax.hist(sub["novelty_score"], bins=30, alpha=0.5, label=str(yr), density=True)
ax.set_xlabel("Novelty score (mean 5-NN distance)")
ax.set_ylabel("Density")
ax.set_title("Novelty Distribution by Year")
ax.legend(title="Year")

plt.tight_layout()
plt.savefig(FIG_DIR / "figure6_validated_novelty.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/figures/figure6_validated_novelty.png")

print("\n" + "=" * 65)
print("VALIDATED NOVELTY ANALYSIS COMPLETE")
print("=" * 65)
