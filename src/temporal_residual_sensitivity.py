from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data/processed/groundwater_rainfall_features_qc.csv"
PREDICTION_FILE = ROOT / "results/controlled_removal_predictions.csv"

OUTPUT_FILE = ROOT / "results/temporal_residual_sensitivity.csv"

TARGET = "Groundwater Level Quarterly Manual (meter)"

RAW_FEATURES = [
    "Latitude",
    "Longitude",
    "rainfall_30d_mm",
    "rainfall_90d_mm",
    "previous_gwl",
    "month_sin",
    "month_cos",
]

MODEL_FEATURES = [
    "latitude_z",
    "longitude_z",
    "rainfall_30d_z",
    "rainfall_90d_z",
    "previous_gwl_z",
    "month_sin_z",
    "month_cos_z",
]


def main():

    print("Loading dataset and existing predictions...")

    data = pd.read_csv(DATA_FILE, parse_dates=["observation_date"])
    predictions = pd.read_csv(PREDICTION_FILE, parse_dates=["date"])

    data["well_id"] = (
        data["Station"].astype(str) + "_"
        + data["Latitude"].astype(str) + "_"
        + data["Longitude"].astype(str)
    )

    data = data.sort_values(["well_id", "observation_date"]).copy()

    data["previous_gwl"] = (
        data.groupby("well_id")[TARGET].shift(1)
    )

    month = data["observation_date"].dt.month

    data["month_sin"] = np.sin(2 * np.pi * month / 12)
    data["month_cos"] = np.cos(2 * np.pi * month / 12)

    training = data[
        data["observation_date"] < "2018-01-01"
    ].copy()

    familiar = predictions[
        predictions["novelty_group"] == "Familiar"
    ].copy()

    inputs = data[
        ["well_id", "observation_date"] + RAW_FEATURES
    ].copy()

    merged = familiar.merge(
        inputs,
        left_on=["well_id", "date"],
        right_on=["well_id", "observation_date"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if (merged["_merge"] != "both").any():
        raise ValueError("Some predictions could not be matched.")

    merged = merged.drop(
        columns=["_merge", "observation_date"]
    )

    merged["year"] = merged["date"].dt.year

    merged["year_cat"] = pd.Categorical(
        merged["year"],
        categories=[2018, 2019, 2020],
    )

    merged["residual_m"] = (
        merged["full_model_prediction_m"]
        - merged["actual_gwl_m"]
    )

    feature_names = dict(zip(RAW_FEATURES, MODEL_FEATURES))

    for raw, standardized in feature_names.items():

        mean = training[raw].mean()
        std = training[raw].std()

        merged[standardized] = (
            merged[raw] - mean
        ) / std

    merged = merged.dropna(
        subset=MODEL_FEATURES + ["residual_m", "well_id", "year_cat"]
    ).copy()

    feature_expression = " + ".join(MODEL_FEATURES)

    formula_1 = f"residual_m ~ {feature_expression}"

    formula_2 = (
        f"residual_m ~ {feature_expression} + C(year_cat)"
    )

    formula_3 = (
        f"residual_m ~ C(year_cat) * ({feature_expression})"
    )

    scenarios = [
        ("Original", 0.00),
        ("Trimmed_1_percent", 0.01),
        ("Trimmed_5_percent", 0.05),
    ]

    results = []

    for scenario, trim_fraction in scenarios:

        print(f"\nRunning scenario: {scenario}")

        sample = merged.copy()

        if trim_fraction > 0:

            threshold = sample["residual_m"].abs().quantile(
                1 - trim_fraction
            )

            sample = sample[
                sample["residual_m"].abs() <= threshold
            ].copy()

        print(f"Observations: {len(sample):,}")
        print(f"Wells: {sample['well_id'].nunique():,}")

        model_1 = smf.ols(
            formula_1, data=sample
        ).fit(
            cov_type="cluster",
            cov_kwds={"groups": sample["well_id"]},
        )

        model_2 = smf.ols(
            formula_2, data=sample
        ).fit(
            cov_type="cluster",
            cov_kwds={"groups": sample["well_id"]},
        )

        model_3 = smf.ols(
            formula_3, data=sample
        ).fit(
            cov_type="cluster",
            cov_kwds={"groups": sample["well_id"]},
        )

        interaction_terms = [
            name for name in model_3.params.index
            if ":" in name and "year_cat" in name
        ]

        hypothesis = " = 0, ".join(interaction_terms) + " = 0"

        interaction_test = model_3.wald_test(
            hypothesis,
            scalar=True,
        )

        results.append({
            "scenario": scenario,
            "trim_fraction": trim_fraction,
            "n_observations": len(sample),
            "n_wells": sample["well_id"].nunique(),
            "R2_inputs_only": model_1.rsquared,
            "R2_inputs_year": model_2.rsquared,
            "R2_interactions": model_3.rsquared,
            "interaction_wald_statistic": float(
                interaction_test.statistic
            ),
            "interaction_p_value": float(
                interaction_test.pvalue
            ),
        })

        print(
            f"Interaction Wald statistic: "
            f"{float(interaction_test.statistic):.5f}"
        )

        print(
            f"Interaction p-value: "
            f"{float(interaction_test.pvalue):.8g}"
        )

    results_df = pd.DataFrame(results)

    results_df.to_csv(OUTPUT_FILE, index=False)

    print("\nFINAL SENSITIVITY COMPARISON")
    print("=" * 70)
    print(results_df.to_string(index=False))

    print(f"\nSaved results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
