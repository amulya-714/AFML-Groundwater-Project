from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "results/conditional_residual_observations.csv"

OUTPUT_FILE = ROOT / "results/paired_temporal_bin_summary.csv"
DETAIL_FILE = ROOT / "results/paired_temporal_bin_per_well.csv"

FEATURES = [
    "rainfall_30d_mm",
    "rainfall_90d_mm",
    "previous_gwl",
]

N_BOOTSTRAP = 2000
RANDOM_STATE = 42


def bootstrap_ci(values, rng):

    values = np.asarray(values, dtype=float)
    n = len(values)

    if n < 2:
        return np.nan, np.nan

    bootstrap_means = np.empty(N_BOOTSTRAP)

    for i in range(N_BOOTSTRAP):
        sample = rng.choice(values, size=n, replace=True)
        bootstrap_means[i] = sample.mean()

    return (
        np.percentile(bootstrap_means, 2.5),
        np.percentile(bootstrap_means, 97.5),
    )


def main():

    print("Loading conditional residual observations...")

    df = pd.read_csv(INPUT_FILE, parse_dates=["date"])

    required = {
        "well_id",
        "date",
        "year",
        "residual_m",
        "absolute_error_m",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df[df["year"].isin([2018, 2020])].copy()

    rng = np.random.default_rng(RANDOM_STATE)

    all_results = []
    all_details = []

    for feature in FEATURES:

        bin_column = f"{feature}_bin"

        if bin_column not in df.columns:
            raise ValueError(f"Missing bin column: {bin_column}")

        print(f"\nAnalysing {feature}...")

        # Calculate error metrics for each well, year and bin.
        per_well = (
            df.groupby(
                ["well_id", bin_column, "year"],
                observed=True,
            )
            .agg(
                n_observations=("absolute_error_m", "size"),
                MAE_m=("absolute_error_m", "mean"),
                mean_signed_error_m=("residual_m", "mean"),
            )
            .reset_index()
        )

        # Separate 2018 and 2020.
        year_2018 = per_well[
            per_well["year"] == 2018
        ].drop(columns="year")

        year_2020 = per_well[
            per_well["year"] == 2020
        ].drop(columns="year")

        paired = year_2018.merge(
            year_2020,
            on=["well_id", bin_column],
            how="inner",
            suffixes=("_2018", "_2020"),
        )

        if paired.empty:
            print(f"No paired wells found for {feature}.")
            continue

        paired["MAE_difference_m"] = (
            paired["MAE_m_2020"] - paired["MAE_m_2018"]
        )

        paired["signed_error_difference_m"] = (
            paired["mean_signed_error_m_2020"]
            - paired["mean_signed_error_m_2018"]
        )

        paired["feature"] = feature

        all_details.append(paired)

        # Test each feature bin separately.
        for bin_value, group in paired.groupby(
            bin_column,
            observed=True,
        ):

            mae_diff = group["MAE_difference_m"].to_numpy()

            bias_diff = group[
                "signed_error_difference_m"
            ].to_numpy()

            mae_ci = bootstrap_ci(mae_diff, rng)
            bias_ci = bootstrap_ci(bias_diff, rng)

            if np.allclose(mae_diff, 0):
                mae_statistic = 0.0
                mae_p = 1.0
            else:
                test = wilcoxon(
                    mae_diff,
                    alternative="greater",
                )
                mae_statistic = float(test.statistic)
                mae_p = float(test.pvalue)

            if np.allclose(bias_diff, 0):
                bias_statistic = 0.0
                bias_p = 1.0
            else:
                test = wilcoxon(
                    bias_diff,
                    alternative="two-sided",
                )
                bias_statistic = float(test.statistic)
                bias_p = float(test.pvalue)

            all_results.append({
                "feature": feature,
                "input_bin": str(bin_value),
                "n_paired_wells": len(group),
                "mean_MAE_2018_m": group["MAE_m_2018"].mean(),
                "mean_MAE_2020_m": group["MAE_m_2020"].mean(),
                "mean_paired_MAE_difference_m": mae_diff.mean(),
                "MAE_CI_lower_m": mae_ci[0],
                "MAE_CI_upper_m": mae_ci[1],
                "fraction_wells_with_increased_MAE": np.mean(
                    mae_diff > 0
                ),
                "MAE_wilcoxon_p": mae_p,
                "mean_signed_error_2018_m": group[
                    "mean_signed_error_m_2018"
                ].mean(),
                "mean_signed_error_2020_m": group[
                    "mean_signed_error_m_2020"
                ].mean(),
                "mean_signed_error_difference_m": bias_diff.mean(),
                "bias_CI_lower_m": bias_ci[0],
                "bias_CI_upper_m": bias_ci[1],
                "bias_wilcoxon_p": bias_p,
            })

    if not all_results:
        raise ValueError("No paired comparisons could be calculated.")

    summary = pd.DataFrame(all_results)
    details = pd.concat(all_details, ignore_index=True)

    summary.to_csv(OUTPUT_FILE, index=False)
    details.to_csv(DETAIL_FILE, index=False)

    print("\nPAIRED TEMPORAL COMPARISON: 2018 VS 2020")
    print("=" * 90)
    print(summary.round(4).to_string(index=False))

    print("\nSaved:")
    print(OUTPUT_FILE)
    print(DETAIL_FILE)


if __name__ == "__main__":
    main()
