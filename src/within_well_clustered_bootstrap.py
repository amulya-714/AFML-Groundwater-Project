"""Well-clustered bootstrap for matched temporal prediction errors."""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "results" / "within_well_matched_input_pairs.csv"
OUTPUT_FILE = ROOT / "results" / "within_well_clustered_bootstrap.csv"

N_BOOTSTRAPS = 5000
RANDOM_STATE = 42


def main():
    print("Loading within-well matched observations...")

    df = pd.read_csv(INPUT_FILE)

    required = {
        "well_id",
        "absolute_error_2018_m",
        "absolute_error_2020_m",
        "absolute_error_difference_m",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.dropna(subset=list(required)).copy()

    # Calculate the error difference for each matched pair.
    df["error_difference"] = (
        df["absolute_error_2020_m"]
        - df["absolute_error_2018_m"]
    )

    # Calculate the mean difference separately for every well.
    well_results = (
        df.groupby("well_id")
        .agg(
            n_pairs=("error_difference", "size"),
            mean_error_2018=("absolute_error_2018_m", "mean"),
            mean_error_2020=("absolute_error_2020_m", "mean"),
            mean_error_difference=("error_difference", "mean"),
        )
        .reset_index()
    )

    n_wells = len(well_results)

    # Original overall statistics.
    observed_difference = df["error_difference"].mean()
    observed_mae_2018 = df["absolute_error_2018_m"].mean()
    observed_mae_2020 = df["absolute_error_2020_m"].mean()

    print(f"Matched pairs: {len(df):,}")
    print(f"Distinct wells: {n_wells:,}")
    print(f"Mean MAE in 2018: {observed_mae_2018:.4f} m")
    print(f"Mean MAE in 2020: {observed_mae_2020:.4f} m")
    print(f"Observed error increase: {observed_difference:.4f} m")

    # Bootstrap entire wells, not individual observations.
    # This preserves dependence between observations from the same well.
    rng = np.random.default_rng(RANDOM_STATE)

    bootstrap_differences = []

    print("\nRunning well-clustered bootstrap...")

    for i in range(N_BOOTSTRAPS):

        sampled_indices = rng.integers(
            0, n_wells, size=n_wells
        )

        sampled_wells = well_results.iloc[sampled_indices]

        # Reconstruct the pair-weighted difference using sampled wells.
        sampled_difference = np.average(
            sampled_wells["mean_error_difference"],
            weights=sampled_wells["n_pairs"],
        )

        bootstrap_differences.append(sampled_difference)

        if (i + 1) % 1000 == 0:
            print(f"Completed {i + 1}/{N_BOOTSTRAPS}")

    bootstrap_differences = np.array(bootstrap_differences)

    # Percentile confidence interval.
    ci_lower, ci_upper = np.percentile(
        bootstrap_differences, [2.5, 97.5]
    )

    # Fraction of bootstrap estimates above zero.
    probability_positive = np.mean(
        bootstrap_differences > 0
    )

    fraction_wells_increased = np.mean(
        well_results["mean_error_difference"] > 0
    )

    summary = pd.DataFrame([{
        "n_matched_pairs": len(df),
        "n_wells": n_wells,
        "mean_MAE_2018_m": observed_mae_2018,
        "mean_MAE_2020_m": observed_mae_2020,
        "mean_error_increase_m": observed_difference,
        "bootstrap_CI_lower_m": ci_lower,
        "bootstrap_CI_upper_m": ci_upper,
        "fraction_bootstrap_estimates_positive": probability_positive,
        "fraction_wells_with_increased_error": fraction_wells_increased,
        "n_bootstraps": N_BOOTSTRAPS,
    }])

    summary.to_csv(OUTPUT_FILE, index=False)

    print("\nWELL-CLUSTERED BOOTSTRAP RESULTS")
    print("=" * 55)
    print(summary.to_string(index=False))

    print(f"\nSaved results to: {OUTPUT_FILE}")

    if ci_lower > 0:
        print(
            "\nInterpretation: The 95% bootstrap confidence interval "
            "is entirely above zero."
        )
    elif ci_upper < 0:
        print(
            "\nInterpretation: The 95% bootstrap confidence interval "
            "is entirely below zero."
        )
    else:
        print(
            "\nInterpretation: The 95% bootstrap confidence interval "
            "includes zero."
        )


if __name__ == "__main__":
    main()
