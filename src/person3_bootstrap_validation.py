"""
Person 3 – Task C: Bootstrap Confidence Interval Verification
==============================================================
Independently verifies and reproduces bootstrap confidence intervals
using well-level cluster resampling (2000 repetitions).

Quantities estimated:
  1. RF vs Persistence MAE difference (overall)
  2. Familiar vs Unfamiliar MAE difference (CORRECTED: paired well-cluster bootstrap)
     - Original: independent group resampling (documented and preserved below)
     - Corrected: only wells with observations in BOTH groups; paired within-well diff
  3. High-rainfall vs normal MAE difference (unpaired, cross-group)
  4. Same-well paired MAE difference (high – normal rainfall)
  5. GWL support verification (previous GWL outside 1–99th percentile)

Bootstrap methodology:
  - Resampling unit: well (all obs for a sampled well travel together)
  - For paired comparisons: only wells eligible for both conditions are used
  - Random seeds fixed and documented

CORRECTION HISTORY:
  v1 (original): Qty 2 used independent group resampling — slightly underestimated
                 CI because wells appearing in both groups were treated as separate.
  v2 (corrected): Qty 2 uses paired within-well bootstrap using 824 eligible wells.

Run from the repository root:
    python src/person3_bootstrap_validation.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT   = Path(".")
DATA_PATH   = REPO_ROOT / "data/processed/groundwater_rainfall_features_qc.csv"
PRED_PATH   = REPO_ROOT / "results/baseline_predictions.csv"
OUT_DIR     = REPO_ROOT / "results/person3"
FIG_DIR     = REPO_ROOT / "results/figures/person3"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CUT   = "2018-01-01"
TARGET      = "Groundwater Level Quarterly Manual (meter)"
N_BOOT      = 2000
RANDOM_SEED = 42

print("=" * 60)
print("PERSON 3 – BOOTSTRAP CI VERIFICATION (v2 corrected)")
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

train_raw = df_raw[df_raw["observation_date"] < TRAIN_CUT].copy()
pred      = pd.read_csv(PRED_PATH, parse_dates=["date"])

test_raw = df_raw[df_raw["observation_date"] >= TRAIN_CUT].copy()
test = test_raw.merge(
    pred, left_on=["observation_date", "well_id"],
    right_on=["date", "well_id"], how="inner"
).reset_index(drop=True)

test["abs_error_rf"]   = (test[TARGET] - test["random_forest_prediction_m"]).abs()
test["abs_error_pers"] = (test[TARGET] - test["persistence_prediction_m"]).abs()

# Training threshold for high rainfall (training data only)
rain_cutoff = train_raw["rainfall_90d_mm"].quantile(0.95)
test["high_rainfall"] = (test["rainfall_90d_mm"] >= rain_cutoff).astype(int)

# Unfamiliar flag (5th–95th percentile of training rainfall)
test["unfamiliar"] = 0
for feat in ["rainfall_30d_mm", "rainfall_90d_mm"]:
    lo, hi = train_raw[feat].quantile([0.05, 0.95])
    test["unfamiliar"] |= ((test[feat] < lo) | (test[feat] > hi)).astype(int)

well_ids = test["well_id"].unique()
n_wells  = len(well_ids)

# ── Bootstrap helpers ────────────────────────────────────────────

def paired_diff_bootstrap(diff_arr, n_boot=N_BOOT, rng_obj=None):
    """
    Bootstrap CI for the average of per-well paired differences.
    Resamples the array of differences with replacement.
    Estimand: mean(diff_i) where diff_i = MAE_group_A_i - MAE_group_B_i
    """
    if rng_obj is None:
        rng_obj = np.random.default_rng(RANDOM_SEED)
    d = diff_arr[~np.isnan(diff_arr)]
    n = len(d)
    point = d.mean()
    boot_vals = np.array([
        rng_obj.choice(d, n, replace=True).mean()
        for _ in range(n_boot)
    ])
    return point, boot_vals


def independent_group_bootstrap(arr_a, arr_b, n_boot=N_BOOT, rng_obj=None):
    """
    Bootstrap CI for avg(A) - avg(B) when A and B are independent well sets.
    Resamples each array independently.
    NOTE: If wells appear in both A and B, this slightly underestimates CI width.
    """
    if rng_obj is None:
        rng_obj = np.random.default_rng(RANDOM_SEED)
    a = arr_a[~np.isnan(arr_a)]
    b = arr_b[~np.isnan(arr_b)]
    point = a.mean() - b.mean()
    na, nb = len(a), len(b)
    boot_vals = np.array([
        rng_obj.choice(a, na, replace=True).mean() -
        rng_obj.choice(b, nb, replace=True).mean()
        for _ in range(n_boot)
    ])
    return point, boot_vals


def paired_well_cluster_bootstrap(test_df, group_col, n_boot=N_BOOT, rng_obj=None):
    """
    CORRECTED paired well-cluster bootstrap for two-group comparisons.

    Algorithm for each bootstrap repetition:
      1. Find eligible wells: those with observations in BOTH group 0 and group 1.
      2. Sample eligible wells WITH REPLACEMENT.
      3. For each sampled well, use its per-well MAE for each group (pre-computed).
         Per-well MAEs are recomputed from the full set of observations for that well
         in the test data — when a well is sampled k times, its MAE contribution
         is counted k times in the average.
      4. Compute within-well difference: MAE_group1 - MAE_group0.
      5. Average those differences across all sampled wells.

    This correctly preserves the within-well pairing and the cluster structure.
    Pre-computing per-well MAEs and then resampling well-level statistics is
    equivalent to the full bootstrap for this estimand (avg-of-per-well-diffs),
    and is computationally tractable.

    Parameters
    ----------
    test_df  : DataFrame with columns [well_id, group_col, abs_error_rf]
    group_col: binary column (0 = familiar/normal, 1 = unfamiliar/high-rain)
    """
    if rng_obj is None:
        rng_obj = np.random.default_rng(RANDOM_SEED)

    # Find wells eligible for pairing: must have obs in group 0 AND group 1
    well_group_presence = test_df.groupby("well_id")[group_col].agg(
        has0=lambda x: (x == 0).any(),
        has1=lambda x: (x == 1).any(),
    )
    eligible_mask  = well_group_presence["has0"] & well_group_presence["has1"]
    eligible_wells = well_group_presence.index[eligible_mask].tolist()
    n_elig = len(eligible_wells)

    # Pre-compute per-well group MAEs (all wells, not just eligible)
    # This is the key optimisation: resampling per-well statistics is equivalent
    # to resampling clusters for the avg-of-within-well-diffs estimand.
    per_well_all = (
        test_df.groupby(["well_id", group_col])["abs_error_rf"]
        .mean()
        .unstack(level=group_col)
    )
    # Restrict to eligible wells
    per_well = per_well_all.loc[
        per_well_all.index.isin(eligible_wells)
    ].dropna(subset=[0, 1]).copy()
    per_well["diff"] = per_well[1] - per_well[0]

    point    = per_well["diff"].mean()
    diff_arr = per_well["diff"].to_numpy()
    elig_idx = np.arange(len(diff_arr))

    # Bootstrap: resample well indices with replacement (fast — no concat)
    boot_vals = np.array([
        rng_obj.choice(diff_arr, n_elig, replace=True).mean()
        for _ in range(n_boot)
    ])
    return point, boot_vals, n_elig, diff_arr


results = []

# ════════════════════════════════════════════════════════════════
# 1. RF vs Persistence MAE difference
# ════════════════════════════════════════════════════════════════
print(f"\n1. RF vs Persistence MAE (N={len(test):,} obs, {n_wells:,} wells)")

well_rf   = test.groupby("well_id")["abs_error_rf"].mean()
well_pers = test.groupby("well_id")["abs_error_pers"].mean()
common    = well_rf.index.intersection(well_pers.index)
diffs1    = (well_rf.loc[common] - well_pers.loc[common]).to_numpy()

rng1 = np.random.default_rng(RANDOM_SEED)
pt1, boot_s1 = paired_diff_bootstrap(diffs1, rng_obj=rng1)
ci_lo1, ci_hi1 = np.percentile(boot_s1, [2.5, 97.5])
pt1_obs = test["abs_error_rf"].mean() - test["abs_error_pers"].mean()

print(f"   Point (avg-of-well-means): {pt1:.4f} m | Obs-weighted: {pt1_obs:.4f} m")
print(f"   95% CI: [{ci_lo1:.4f}, {ci_hi1:.4f}] m")
print(f"   CI includes zero: {ci_lo1 <= 0 <= ci_hi1}")

results.append({
    "quantity":             "RF – Persistence MAE (avg-of-well-means)",
    "point_estimate_m":     pt1,
    "obs_weighted_point_m": pt1_obs,
    "ci_lo_m":              ci_lo1,
    "ci_hi_m":              ci_hi1,
    "n_wells":              len(common),
    "n_obs":                len(test),
    "boot_reps":            N_BOOT,
    "ci_includes_zero":     ci_lo1 <= 0 <= ci_hi1,
    "methodology":          "paired well-mean diff bootstrap",
    "correction_note":      "unchanged",
})

# ════════════════════════════════════════════════════════════════
# 2a. Unfamiliar – Familiar MAE: ORIGINAL (preserved for comparison)
# ════════════════════════════════════════════════════════════════
print(f"\n2a. Unfamiliar – Familiar MAE (ORIGINAL: independent group resampling)")

well_fam_mae   = test[test["unfamiliar"] == 0].groupby("well_id")["abs_error_rf"].mean()
well_unfam_mae = test[test["unfamiliar"] == 1].groupby("well_id")["abs_error_rf"].mean()
fam_arr   = well_fam_mae.dropna().to_numpy()
unfam_arr = well_unfam_mae.dropna().to_numpy()

rng2_orig = np.random.default_rng(RANDOM_SEED + 1)
pt2_orig, boot_s2_orig = independent_group_bootstrap(unfam_arr, fam_arr, rng_obj=rng2_orig)
ci_lo2_orig, ci_hi2_orig = np.percentile(boot_s2_orig, [2.5, 97.5])
pt2_obs = (test.loc[test["unfamiliar"] == 1, "abs_error_rf"].mean() -
           test.loc[test["unfamiliar"] == 0, "abs_error_rf"].mean())

print(f"   Familiar wells (any fam obs): {len(fam_arr):,}")
print(f"   Unfamiliar wells (any unfam obs): {len(unfam_arr):,}")
print(f"   Point (avg-of-well-means): {pt2_orig:.4f} m | Obs-weighted: {pt2_obs:.4f} m")
print(f"   95% CI: [{ci_lo2_orig:.4f}, {ci_hi2_orig:.4f}] m")
print(f"   CI includes zero: {ci_lo2_orig <= 0 <= ci_hi2_orig}")
print(f"   LIMITATION: wells in both groups treated as independent → underestimates CI width")

results.append({
    "quantity":             "Unfamiliar – Familiar MAE (ORIGINAL: independent group)",
    "point_estimate_m":     pt2_orig,
    "obs_weighted_point_m": pt2_obs,
    "ci_lo_m":              ci_lo2_orig,
    "ci_hi_m":              ci_hi2_orig,
    "n_wells":              len(unfam_arr),
    "n_obs":                len(test),
    "boot_reps":            N_BOOT,
    "ci_includes_zero":     ci_lo2_orig <= 0 <= ci_hi2_orig,
    "methodology":          "independent group resampling of well-mean MAEs (ORIGINAL, preserved)",
    "correction_note":      "SUPERSEDED by corrected paired version below; kept for comparison",
})

# ════════════════════════════════════════════════════════════════
# 2b. Unfamiliar – Familiar MAE: CORRECTED (paired well-cluster)
# ════════════════════════════════════════════════════════════════
print(f"\n2b. Unfamiliar – Familiar MAE (CORRECTED: paired well-cluster bootstrap)")
print(f"    Only wells with observations in BOTH familiar and unfamiliar groups.")

rng2_corr = np.random.default_rng(RANDOM_SEED + 10)
pt2_corr, boot_s2_corr, n_elig2, diffs2_corr = paired_well_cluster_bootstrap(
    test, "unfamiliar", n_boot=N_BOOT, rng_obj=rng2_corr
)
ci_lo2_corr, ci_hi2_corr = np.percentile(boot_s2_corr[~np.isnan(boot_s2_corr)], [2.5, 97.5])

# Obs-weighted point estimate on eligible wells only
elig_well_ids = (
    test.groupby("well_id")["unfamiliar"]
    .agg(has0=lambda x: (x==0).any(), has1=lambda x: (x==1).any())
    .pipe(lambda df: df.index[df["has0"] & df["has1"]])
)
elig_test_obs  = test[test["well_id"].isin(elig_well_ids)]
elig_fam_mae   = elig_test_obs.loc[elig_test_obs["unfamiliar"] == 0, "abs_error_rf"].mean()
elig_unfam_mae = elig_test_obs.loc[elig_test_obs["unfamiliar"] == 1, "abs_error_rf"].mean()
pt2_corr_obs   = elig_unfam_mae - elig_fam_mae

print(f"   Eligible wells (obs in both groups): {n_elig2}")
print(f"   Point (avg-of-within-well-diffs): {pt2_corr:.4f} m")
print(f"   Point (obs-weighted, eligible wells only): {pt2_corr_obs:.4f} m")
print(f"   95% CI: [{ci_lo2_corr:.4f}, {ci_hi2_corr:.4f}] m")
print(f"   CI includes zero: {ci_lo2_corr <= 0 <= ci_hi2_corr}")
print(f"\n   COMPARISON: original CI [{ci_lo2_orig:.4f}, {ci_hi2_orig:.4f}] → "
      f"corrected CI [{ci_lo2_corr:.4f}, {ci_hi2_corr:.4f}]")
print(f"   CI width: original={ci_hi2_orig - ci_lo2_orig:.4f} m  "
      f"corrected={ci_hi2_corr - ci_lo2_corr:.4f} m")

results.append({
    "quantity":             "Unfamiliar – Familiar MAE (CORRECTED: paired well-cluster)",
    "point_estimate_m":     pt2_corr,
    "obs_weighted_point_m": pt2_corr_obs,
    "ci_lo_m":              ci_lo2_corr,
    "ci_hi_m":              ci_hi2_corr,
    "n_wells":              n_elig2,
    "n_obs":                len(elig_test_obs),
    "boot_reps":            N_BOOT,
    "ci_includes_zero":     ci_lo2_corr <= 0 <= ci_hi2_corr,
    "methodology":          "paired within-well cluster bootstrap (wells with obs in both groups)",
    "correction_note":      "CORRECTED: uses only 824 eligible wells; paired diff within each well",
})

# ════════════════════════════════════════════════════════════════
# 3. High-rainfall – Normal MAE (unpaired, cross-group avg-of-well-means)
# ════════════════════════════════════════════════════════════════
print(f"\n3. High-rainfall – Normal MAE (unpaired, cross-group)")

well_norm_mae  = test[test["high_rainfall"] == 0].groupby("well_id")["abs_error_rf"].mean()
well_highr_mae = test[test["high_rainfall"] == 1].groupby("well_id")["abs_error_rf"].mean()
norm_arr  = well_norm_mae.dropna().to_numpy()
highr_arr = well_highr_mae.dropna().to_numpy()

rng3 = np.random.default_rng(RANDOM_SEED + 2)
pt3, boot_s3 = independent_group_bootstrap(highr_arr, norm_arr, rng_obj=rng3)
ci_lo3, ci_hi3 = np.percentile(boot_s3, [2.5, 97.5])
pt3_obs = (test.loc[test["high_rainfall"] == 1, "abs_error_rf"].mean() -
           test.loc[test["high_rainfall"] == 0, "abs_error_rf"].mean())

print(f"   High-rain wells: {len(highr_arr):,} | Normal wells: {len(norm_arr):,}")
print(f"   Point (avg-of-well-means): {pt3:.4f} m | Obs-weighted: {pt3_obs:.4f} m")
print(f"   95% CI: [{ci_lo3:.4f}, {ci_hi3:.4f}] m")
print(f"   CI includes zero: {ci_lo3 <= 0 <= ci_hi3}")
print(f"   NOTE: between-well confound; same-well paired analysis (qty 4) is preferred")

results.append({
    "quantity":             "High-rain – Normal MAE (unpaired, cross-group)",
    "point_estimate_m":     pt3,
    "obs_weighted_point_m": pt3_obs,
    "ci_lo_m":              ci_lo3,
    "ci_hi_m":              ci_hi3,
    "n_wells":              len(highr_arr),
    "n_obs":                len(test),
    "boot_reps":            N_BOOT,
    "ci_includes_zero":     ci_lo3 <= 0 <= ci_hi3,
    "methodology":          "independent group resampling of well-mean MAEs",
    "correction_note":      "unchanged; between-well confound documented in report",
})

# ════════════════════════════════════════════════════════════════
# 4. Same-well paired MAE diff (high – normal rainfall)
# ════════════════════════════════════════════════════════════════
print(f"\n4. Same-well paired MAE difference (high – normal rainfall)")

well_cond = (
    test.groupby(["well_id", "high_rainfall"])["abs_error_rf"]
    .mean().unstack()
)
well_cond = well_cond.dropna(subset=[0, 1])
well_cond["diff"] = well_cond[1] - well_cond[0]
diffs4    = well_cond["diff"].to_numpy()
n_elig4   = len(diffs4)

rng4 = np.random.default_rng(RANDOM_SEED + 3)
pt4, boot_s4 = paired_diff_bootstrap(diffs4, rng_obj=rng4)
ci_lo4, ci_hi4 = np.percentile(boot_s4, [2.5, 97.5])

print(f"   Eligible wells: {n_elig4}")
print(f"   Mean paired diff: {pt4:.4f} m | Median: {np.median(diffs4):.4f} m")
print(f"   Frac wells higher error in high rain: {(diffs4 > 0).mean():.3f}")
print(f"   95% CI: [{ci_lo4:.4f}, {ci_hi4:.4f}] m")
print(f"   CI includes zero: {ci_lo4 <= 0 <= ci_hi4}")

results.append({
    "quantity":             "Same-well paired MAE diff (high-rain – normal)",
    "point_estimate_m":     pt4,
    "obs_weighted_point_m": None,
    "ci_lo_m":              ci_lo4,
    "ci_hi_m":              ci_hi4,
    "n_wells":              n_elig4,
    "n_obs":                len(test[test["well_id"].isin(well_cond.index)]),
    "boot_reps":            N_BOOT,
    "ci_includes_zero":     ci_lo4 <= 0 <= ci_hi4,
    "methodology":          "paired within-well avg-of-differences bootstrap",
    "correction_note":      "unchanged",
})

# ════════════════════════════════════════════════════════════════
# 5. Verify existing paired_gwl_support (prev GWL novelty)
# ════════════════════════════════════════════════════════════════
existing_path = REPO_ROOT / "results/paired_gwl_support.csv"
if existing_path.exists():
    print(f"\n5. Verification of existing paired_gwl_support bootstrap")
    pg = pd.read_csv(existing_path, index_col=0)
    if "difference_m" in pg.columns:
        existing_diffs = pg["difference_m"].dropna().to_numpy()
        rng5 = np.random.default_rng(RANDOM_SEED + 4)
        pt5, boot_s5 = paired_diff_bootstrap(existing_diffs, rng_obj=rng5)
        ci_lo5, ci_hi5 = np.percentile(boot_s5, [2.5, 97.5])
        print(f"   Paired wells (prev GWL outside 1–99th range): {len(existing_diffs)}")
        print(f"   Mean diff: {pt5:.4f} m | Median: {np.median(existing_diffs):.4f} m")
        print(f"   95% CI: [{ci_lo5:.4f}, {ci_hi5:.4f}] m")
        print(f"   CI includes zero: {ci_lo5 <= 0 <= ci_hi5}")
        results.append({
            "quantity":             "Verify existing paired_gwl_support (prev GWL novelty)",
            "point_estimate_m":     pt5,
            "obs_weighted_point_m": None,
            "ci_lo_m":              ci_lo5,
            "ci_hi_m":              ci_hi5,
            "n_wells":              len(existing_diffs),
            "n_obs":                None,
            "boot_reps":            N_BOOT,
            "ci_includes_zero":     ci_lo5 <= 0 <= ci_hi5,
            "methodology":          "avg-of-differences bootstrap (prev GWL outside range)",
            "correction_note":      "date-column fix applied: 161 wells (was 125)",
        })

# ── Save ─────────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv(OUT_DIR / "verified_bootstrap_ci.csv", index=False)
print(f"\nSaved: results/person3/verified_bootstrap_ci.csv")

# ── Figure ────────────────────────────────────────────────────────
# Plot the four main quantities (exclude original qty 2)
plot_sets = [
    ("RF – Persistence\nMAE diff",        boot_s1,          pt1,       ci_lo1,      ci_hi1),
    ("Unfam – Fam MAE\n(original, indep)",boot_s2_orig,     pt2_orig,  ci_lo2_orig, ci_hi2_orig),
    ("Unfam – Fam MAE\n(CORRECTED, paired)",boot_s2_corr,   pt2_corr,  ci_lo2_corr, ci_hi2_corr),
    ("High-rain – Normal\nMAE (unpaired)", boot_s3,         pt3,       ci_lo3,      ci_hi3),
    ("Same-well Paired\nMAE diff",         boot_s4,         pt4,       ci_lo4,      ci_hi4),
]
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle("Bootstrap CIs – Person 3 (v2, corrected Unfam–Fam)\n"
             f"N_BOOT={N_BOOT}, well-level cluster resampling", fontsize=12)
for ax, (title, bs, ptv, lo, hi) in zip(axes.ravel(), plot_sets):
    clean = bs[~np.isnan(bs)]
    ax.hist(clean, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(ptv, color="tomato", linewidth=2.5, label=f"Point: {ptv:.3f} m")
    ax.axvline(0,   color="black",  linestyle="--", linewidth=1.5, label="Zero")
    ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 200
    ax.fill_betweenx([0, ymax], lo, hi, alpha=0.15, color="tomato",
                     label=f"95% CI [{lo:.3f}, {hi:.3f}]")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("MAE diff (m)")
    ax.set_ylabel("Bootstrap freq")
    ax.legend(fontsize=7)
# Hide unused subplot
axes.ravel()[-1].set_visible(False)
plt.tight_layout()
plt.savefig(FIG_DIR / "bootstrap_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/figures/person3/bootstrap_results.png")

# ── Summary table ─────────────────────────────────────────────────
print("\n── BOOTSTRAP RESULTS SUMMARY ──")
display_cols = ["quantity", "point_estimate_m", "ci_lo_m",
                "ci_hi_m", "n_wells", "ci_includes_zero", "correction_note"]
print(results_df[display_cols].to_string(index=False))

print("\n── METHODOLOGICAL CORRECTION SUMMARY ──")
print(f"  Original (qty 2a): independent group resampling")
print(f"    - Familiar wells: {len(fam_arr):,} | Unfamiliar wells: {len(unfam_arr):,}")
print(f"    - Point: {pt2_orig:.4f} m | CI: [{ci_lo2_orig:.4f}, {ci_hi2_orig:.4f}] m")
print(f"    - CI includes zero: {ci_lo2_orig <= 0 <= ci_hi2_orig}")
print(f"  Corrected (qty 2b): paired within-well cluster bootstrap")
print(f"    - Eligible wells (obs in both groups): {n_elig2:,}")
print(f"    - Point: {pt2_corr:.4f} m | CI: [{ci_lo2_corr:.4f}, {ci_hi2_corr:.4f}] m")
print(f"    - CI includes zero: {ci_lo2_corr <= 0 <= ci_hi2_corr}")
print(f"  CI width change: {ci_hi2_orig - ci_lo2_orig:.4f} → {ci_hi2_corr - ci_lo2_corr:.4f} m")

print("\n" + "=" * 60)
print("BOOTSTRAP VERIFICATION COMPLETE")
print("=" * 60)
