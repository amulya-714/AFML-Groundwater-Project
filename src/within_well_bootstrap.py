"""Bootstrap confidence interval for within-well novelty differences."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "results" / "within_well_paired_test_per_well.csv"
)

OUTPUT_FILE = (
    ROOT / "results" / "within_well_bootstrap_summary.csv"
)

N_BOOTSTRAPS = 10000
RANDOM_SEED = 42


def main():
    df = pd.read_csv(INPUT_FILE)

    required = {
        "well_id",
        "mae_low_m",
        "mae_high_m",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = df.dropna(
        subset=["well_id", "mae_low_m", "mae_high_m"]
    )

    if df.empty:
        raise ValueError("No paired wells available.")

    # Each well contributes one paired difference.
    df["mae_difference_m"] = (
        df["mae_high_m"] - df["mae_low_m"]
    )

    differences = df["mae_difference_m"].to_numpy()

    rng = np.random.default_rng(RANDOM_SEED)

    bootstrap_means = np.empty(N_BOOTSTRAPS)

    # Resample wells, keeping each well's paired difference intact.
    for i in range(N_BOOTSTRAPS):
        sample = rng.choice(
            differences,
            size=len(differences),
            replace=True,
        )
        bootstrap_means[i] = sample.mean()

    ci_lower, ci_upper = np.percentile(
        bootstrap_means, [2.5, 97.5]
    )

    summary = pd.DataFrame(
        [
            {
                "n_paired_wells": len(differences),
                "mean_mae_difference_m": differences.mean(),
                "median_mae_difference_m": np.median(
                    differences
                ),
                "bootstrap_95ci_lower_m": ci_lower,
                "bootstrap_95ci_upper_m": ci_upper,
                "bootstrap_iterations": N_BOOTSTRAPS,
                "random_seed": RANDOM_SEED,
            }
        ]
    )

    summary.to_csv(OUTPUT_FILE, index=False)

    print("\nWITHIN-WELL BOOTSTRAP RESULTS")
    print("=" * 40)
    print(summary.to_string(index=False))
    print(f"\nSaved results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
