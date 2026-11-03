from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "results" / "controlled_removal_predictions.csv"
OUTPUT = ROOT / "results" / "controlled_removal_bootstrap.csv"

N_BOOTSTRAPS = 5000
SEED = 42

GROUPS = [
    "All test observations",
    "Familiar",
    "Poorly represented",
    "Extreme novelty",
]

MODEL_ERRORS = {
    "Full model": "full_model_absolute_error_m",
    "Extreme-removed model": "extreme-removed_model_absolute_error_m",
    "Matched-control model": "matched-control_model_absolute_error_m",
}


def paired_cluster_bootstrap(group, comparison_column, rng):

    wells = group["well_id"].unique()

    error_data = {
        well: group.loc[
            group["well_id"] == well,
            ["full_model_absolute_error_m", comparison_column],
        ].to_numpy()
        for well in wells
    }

    observed_difference = (
        group[comparison_column].mean()
        - group["full_model_absolute_error_m"].mean()
    )

    bootstrap_differences = []

    for _ in range(N_BOOTSTRAPS):

        sampled_wells = rng.choice(
            wells,
            size=len(wells),
            replace=True,
        )

        sampled_errors = np.concatenate(
            [error_data[well] for well in sampled_wells],
            axis=0,
        )

        full_mae = sampled_errors[:, 0].mean()
        comparison_mae = sampled_errors[:, 1].mean()

        bootstrap_differences.append(
            comparison_mae - full_mae
        )

    lower, upper = np.percentile(
        bootstrap_differences,
        [2.5, 97.5],
    )

    return observed_difference, lower, upper


def main():

    df = pd.read_csv(INPUT)

    required = {
        "well_id",
        "novelty_group",
        "full_model_absolute_error_m",
        "extreme-removed_model_absolute_error_m",
        "matched-control_model_absolute_error_m",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    rng = np.random.default_rng(SEED)

    results = []

    for group_name in GROUPS:

        if group_name == "All test observations":
            group = df.copy()
        else:
            group = df[
                df["novelty_group"] == group_name
            ].copy()

        if group.empty:
            continue

        for model_name, error_column in MODEL_ERRORS.items():

            if model_name == "Full model":
                continue

            difference, lower, upper = (
                paired_cluster_bootstrap(
                    group,
                    error_column,
                    rng,
                )
            )

            results.append(
                {
                    "test_group": group_name,
                    "comparison": f"{model_name} minus Full model",
                    "n_observations": len(group),
                    "n_wells": group["well_id"].nunique(),
                    "MAE_difference_m": difference,
                    "CI_95_lower_m": lower,
                    "CI_95_upper_m": upper,
                }
            )

    summary = pd.DataFrame(results)

    summary.to_csv(OUTPUT, index=False)

    print("\nPAIRED WELL-LEVEL BOOTSTRAP")
    print("=" * 75)

    print(summary.round(5).to_string(index=False))

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
