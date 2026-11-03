"""Compare prediction errors between low- and high-novelty
observations within the same groundwater well.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "results" / "within_well_novelty_observations.csv"
OUTPUT_FILE = ROOT / "results" / "within_well_paired_test_summary.csv"
WELL_OUTPUT_FILE = ROOT / "results" / "within_well_paired_test_per_well.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    required = {
        "well_id",
        "absolute_error_m",
        "within_well_novelty_quartile",
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Keep observations with valid errors and novelty groups.
    df = df.dropna(
        subset=[
            "well_id",
            "absolute_error_m",
            "within_well_novelty_quartile",
        ]
    )

    # Average absolute error within each well and novelty group.
    per_well = (
        df.groupby(["well_id", "within_well_novelty_quartile"])
        .agg(
            n_observations=("absolute_error_m", "size"),
            mean_absolute_error_m=("absolute_error_m", "mean"),
        )
        .reset_index()
    )

    # Compare only wells with both low- and high-novelty observations.
    low = per_well[
        per_well["within_well_novelty_quartile"] == "Q1_lowest"
    ][["well_id", "n_observations", "mean_absolute_error_m"]].rename(
        columns={
            "n_observations": "n_low",
            "mean_absolute_error_m": "mae_low_m",
        }
    )

    high = per_well[
        per_well["within_well_novelty_quartile"] == "Q4_highest"
    ][["well_id", "n_observations", "mean_absolute_error_m"]].rename(
        columns={
            "n_observations": "n_high",
            "mean_absolute_error_m": "mae_high_m",
        }
    )

    paired = low.merge(high, on="well_id", how="inner")

    if paired.empty:
        raise ValueError(
            "No wells contain both Q1_lowest and Q4_highest observations."
        )

    paired["mae_difference_m"] = (
        paired["mae_high_m"] - paired["mae_low_m"]
    )

    paired["high_novelty_higher_error"] = (
        paired["mae_difference_m"] > 0
    )

    paired.to_csv(WELL_OUTPUT_FILE, index=False)

    differences = paired["mae_difference_m"].to_numpy()

    # Wilcoxon signed-rank test on paired within-well differences.
    if np.allclose(differences, 0):
        statistic = 0.0
        p_value = 1.0
    else:
        test = wilcoxon(differences, alternative="greater")
        statistic = float(test.statistic)
        p_value = float(test.pvalue)

    summary = pd.DataFrame(
        [
            {
                "n_paired_wells": len(paired),
                "mean_mae_low_m": paired["mae_low_m"].mean(),
                "mean_mae_high_m": paired["mae_high_m"].mean(),
                "median_within_well_difference_m": paired[
                    "mae_difference_m"
                ].median(),
                "mean_within_well_difference_m": differences.mean(),
                "fraction_wells_higher_error_at_high_novelty": (
                    paired["high_novelty_higher_error"].mean()
                ),
                "wilcoxon_statistic": statistic,
                "wilcoxon_one_sided_p_value": p_value,
            }
        ]
    )

    summary.to_csv(OUTPUT_FILE, index=False)

    print("\nWITHIN-WELL PAIRED NOVELTY TEST")
    print("=" * 45)
    print(summary.to_string(index=False))
    print(f"\nSaved summary: {OUTPUT_FILE}")
    print(f"Saved per-well results: {WELL_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
