from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "results" / "paired_temporal_well_errors.csv"

OUTPUT = ROOT / "results" / "paired_temporal_distribution_summary.csv"


def main():

    df = pd.read_csv(INPUT)

    column = "MAE_difference_2020_minus_2018_m"

    differences = df[column].dropna()

    positive = int((differences > 0).sum())
    negative = int((differences < 0).sum())
    unchanged = int((differences == 0).sum())

    sign_test = binomtest(
        positive,
        positive + negative,
        p=0.5,
        alternative="greater",
    )

    quantiles = differences.quantile(
        [0.05, 0.25, 0.50, 0.75, 0.95]
    )

    print("\nPAIRED TEMPORAL ERROR DISTRIBUTION")
    print("=" * 60)

    print(f"Paired wells: {len(differences)}")
    print(f"Error increased: {positive}")
    print(f"Error decreased: {negative}")
    print(f"Error unchanged: {unchanged}")

    print("\nDistribution of 2020 minus 2018 per-well MAE:")
    print(quantiles.to_string())

    print("\nAdditional statistics:")
    print(f"Mean difference: {differences.mean():.5f} m")
    print(f"Median difference: {differences.median():.5f} m")
    print(f"Standard deviation: {differences.std():.5f} m")
    trimmed = differences.clip(
    lower=differences.quantile(0.10),
    upper=differences.quantile(0.90),)
    print(f"10% winsorized mean: {trimmed.mean():.5f} m")

    print("\nOne-sided sign test:")
    print(f"p-value: {sign_test.pvalue:.8g}")

    summary = pd.DataFrame(
        [
            {
                "n_paired_wells": len(differences),
                "n_error_increased": positive,
                "n_error_decreased": negative,
                "n_unchanged": unchanged,
                "mean_difference_m": differences.mean(),
                "median_difference_m": differences.median(),
                "standard_deviation_m": differences.std(),
                "sign_test_p_value": sign_test.pvalue,
            }
        ]
    )

    summary.to_csv(OUTPUT, index=False)

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
