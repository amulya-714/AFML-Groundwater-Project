from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "results" / "controlled_removal_predictions.csv"

OUTPUT_WELL = ROOT / "results" / "paired_temporal_well_errors.csv"
OUTPUT_SUMMARY = ROOT / "results" / "paired_temporal_summary.csv"

N_BOOTSTRAPS = 5000
SEED = 42


def main():

    print("Loading predictions...")

    df = pd.read_csv(INPUT, parse_dates=["date"])

    df["year"] = df["date"].dt.year

    df["residual_m"] = (
        df["full_model_prediction_m"] - df["actual_gwl_m"]
    )

    df["absolute_error_m"] = df["residual_m"].abs()

    # Restrict to familiar observations.
    familiar = df[
        df["novelty_group"] == "Familiar"
    ].copy()

    # Calculate mean absolute error for each well in each year.
    well_year = (
        familiar.groupby(["well_id", "year"])
        .agg(
            n_observations=("absolute_error_m", "size"),
            MAE_m=("absolute_error_m", "mean"),
            mean_signed_error_m=("residual_m", "mean"),
        )
        .reset_index()
    )

    # Keep wells that have observations in both 2018 and 2020.
    pivot = well_year.pivot(
        index="well_id",
        columns="year",
        values="MAE_m",
    )

    paired = pivot.dropna(
        subset=[2018, 2020]
    ).copy()

    paired = paired.rename(
        columns={
            2018: "MAE_2018_m",
            2020: "MAE_2020_m",
        }
    )

    paired["MAE_difference_2020_minus_2018_m"] = (
        paired["MAE_2020_m"] - paired["MAE_2018_m"]
    )

    paired["error_increased"] = (
        paired["MAE_difference_2020_minus_2018_m"] > 0
    )

    if len(paired) < 2:
        raise ValueError(
            "Not enough wells have familiar observations in both years."
        )

    paired.to_csv(OUTPUT_WELL)

    # -------------------------------------------------
    # PAIRED WELL-LEVEL STATISTICS
    # -------------------------------------------------

    differences = paired[
        "MAE_difference_2020_minus_2018_m"
    ].to_numpy()

    rng = np.random.default_rng(SEED)

    bootstrap_differences = []

    for _ in range(N_BOOTSTRAPS):

        sampled = rng.choice(
            differences,
            size=len(differences),
            replace=True,
        )

        bootstrap_differences.append(
            sampled.mean()
        )

    ci_lower, ci_upper = np.percentile(
        bootstrap_differences,
        [2.5, 97.5],
    )

    if np.allclose(differences, 0):
        statistic = 0.0
        p_value = 1.0
    else:
        test = wilcoxon(
            differences,
            alternative="greater",
        )

        statistic = float(test.statistic)
        p_value = float(test.pvalue)

    summary = pd.DataFrame(
        [
            {
                "n_paired_wells": len(paired),
                "mean_MAE_2018_m": paired["MAE_2018_m"].mean(),
                "mean_MAE_2020_m": paired["MAE_2020_m"].mean(),
                "mean_paired_difference_m": differences.mean(),
                "median_paired_difference_m": np.median(differences),
                "fraction_wells_with_increased_error": paired[
                    "error_increased"
                ].mean(),
                "bootstrap_95CI_lower_m": ci_lower,
                "bootstrap_95CI_upper_m": ci_upper,
                "wilcoxon_statistic": statistic,
                "wilcoxon_one_sided_p_value": p_value,
            }
        ]
    )

    summary.to_csv(OUTPUT_SUMMARY, index=False)

    print("\nPAIRED FAMILIAR-INPUT TEMPORAL COMPARISON")
    print("=" * 65)

    print(summary.round(5).to_string(index=False))

    print("\nSaved:")
    print(OUTPUT_WELL)
    print(OUTPUT_SUMMARY)


if __name__ == "__main__":
    main()
