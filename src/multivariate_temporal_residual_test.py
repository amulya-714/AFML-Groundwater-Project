from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data/processed/groundwater_rainfall_features_qc.csv"
PREDICTION_FILE = ROOT / "results/controlled_removal_predictions.csv"

SUMMARY_FILE = ROOT / "results/multivariate_temporal_model_summary.csv"
COEFFICIENT_FILE = ROOT / "results/multivariate_temporal_coefficients.csv"

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
    predictions = pd.read_csv(
        PREDICTION_FILE,
        parse_dates=["date"],
    )

    # Reconstruct the same well identifier used by the baseline.
    data["well_id"] = (
        data["Station"].astype(str) + "_" +
        data["Latitude"].astype(str) + "_" +
        data["Longitude"].astype(str)
    )

    # Reconstruct previous groundwater level chronologically.
    data = data.sort_values(
        ["well_id", "observation_date"]
    ).copy()

    data["previous_gwl"] = (
        data.groupby("well_id")[TARGET].shift(1)
    )

    # Reconstruct the cyclical month features.
    month = data["observation_date"].dt.month

    data["month_sin"] = np.sin(2 * np.pi * month / 12)
    data["month_cos"] = np.cos(2 * np.pi * month / 12)

    # Training-only statistics for standardization.
    training = data[
        data["observation_date"] < "2018-01-01"
    ].copy()

    # Retain familiar observations from the untouched test period.
    familiar = predictions[
        predictions["novelty_group"] == "Familiar"
    ].copy()

    # Prepare the input feature table.
    inputs = data[
        ["well_id", "observation_date"] + RAW_FEATURES
    ].copy()

    if inputs.duplicated(
        ["well_id", "observation_date"]
    ).any():
        raise ValueError("Duplicate well/date keys in dataset.")

    if familiar.duplicated(["well_id", "date"]).any():
        raise ValueError("Duplicate well/date keys in predictions.")

    merged = familiar.merge(
        inputs,
        left_on=["well_id", "date"],
        right_on=["well_id", "observation_date"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = (merged["_merge"] != "both").sum()

    print(f"Familiar test observations: {len(familiar):,}")
    print(f"Unmatched observations: {unmatched:,}")

    if unmatched:
        raise ValueError("Some predictions could not be matched.")

    merged = merged.drop(
        columns=["_merge", "observation_date"]
    )

    merged["year"] = merged["date"].dt.year
    merged["year_cat"] = pd.Categorical(
        merged["year"],
        categories=[2018, 2019, 2020],
    )

    # The residual is prediction minus actual.
    merged["residual_m"] = (
        merged["full_model_prediction_m"] -
        merged["actual_gwl_m"]
    )

    # Standardize each feature using training-period statistics only.
    for raw_feature in RAW_FEATURES:

        mean = training[raw_feature].mean()
        std = training[raw_feature].std()

        if pd.isna(std) or std == 0:
            raise ValueError(
                f"Invalid training standard deviation: {raw_feature}"
            )

        safe_name = {
            "Latitude": "latitude_z",
            "Longitude": "longitude_z",
            "rainfall_30d_mm": "rainfall_30d_z",
            "rainfall_90d_mm": "rainfall_90d_z",
            "previous_gwl": "previous_gwl_z",
            "month_sin": "month_sin_z",
            "month_cos": "month_cos_z",
        }[raw_feature]

        merged[safe_name] = (
            merged[raw_feature] - mean
        ) / std

    merged = merged.dropna(
        subset=MODEL_FEATURES + ["residual_m", "well_id", "year_cat"]
    ).copy()

    if merged["well_id"].nunique() < 2:
        raise ValueError("Not enough distinct wells for clustered inference.")

    print(f"Usable observations: {len(merged):,}")
    print(f"Distinct wells: {merged['well_id'].nunique():,}")
    print("\nObservations by year:")
    print(merged.groupby("year").size().to_string())

    feature_expression = " + ".join(MODEL_FEATURES)

    # Model 1: residuals explained by inputs alone.
    formula_1 = f"residual_m ~ {feature_expression}"

    # Model 2: add a year effect.
    formula_2 = (
        f"residual_m ~ {feature_expression} + C(year_cat)"
    )

    # Model 3: add year-by-feature interactions.
    formula_3 = (
        f"residual_m ~ C(year_cat) * ({feature_expression})"
    )

    print("\nFitting residual models with well-clustered standard errors...")

    model_1 = smf.ols(
        formula_1,
        data=merged,
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": merged["well_id"]},
    )

    model_2 = smf.ols(
        formula_2,
        data=merged,
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": merged["well_id"]},
    )

    model_3 = smf.ols(
        formula_3,
        data=merged,
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": merged["well_id"]},
    )

    # Joint test: do year indicators add explanatory information?
    year_terms = [
        name for name in model_2.params.index
        if name.startswith("C(year_cat)")
    ]

    year_hypothesis = " = 0, ".join(year_terms) + " = 0"
    year_test = model_2.wald_test(
        year_hypothesis,
        scalar=True,
    )

    # Joint test: do year-feature interactions add information?
    interaction_terms = [
        name for name in model_3.params.index
        if ":" in name and "year_cat" in name
    ]

    interaction_hypothesis = (
        " = 0, ".join(interaction_terms) + " = 0"
    )

    interaction_test = model_3.wald_test(
        interaction_hypothesis,
        scalar=True,
    )

    summary = pd.DataFrame([
        {
            "model": "Inputs only",
            "n_observations": int(model_1.nobs),
            "n_wells": merged["well_id"].nunique(),
            "R_squared": model_1.rsquared,
            "adjusted_R_squared": model_1.rsquared_adj,
            "AIC": model_1.aic,
        },
        {
            "model": "Inputs + year",
            "n_observations": int(model_2.nobs),
            "n_wells": merged["well_id"].nunique(),
            "R_squared": model_2.rsquared,
            "adjusted_R_squared": model_2.rsquared_adj,
            "AIC": model_2.aic,
        },
        {
            "model": "Inputs + year + interactions",
            "n_observations": int(model_3.nobs),
            "n_wells": merged["well_id"].nunique(),
            "R_squared": model_3.rsquared,
            "adjusted_R_squared": model_3.rsquared_adj,
            "AIC": model_3.aic,
        },
    ])

    coefficients = pd.concat(
        [
            model_1.summary2().tables[1].assign(model="Inputs only"),
            model_2.summary2().tables[1].assign(model="Inputs + year"),
            model_3.summary2().tables[1].assign(
                model="Inputs + year + interactions"
            ),
        ],
        axis=0,
    ).reset_index(names="term")

    summary.to_csv(SUMMARY_FILE, index=False)
    coefficients.to_csv(COEFFICIENT_FILE, index=False)

    print("\nMODEL COMPARISON")
    print("=" * 75)
    print(summary.round(5).to_string(index=False))

    print("\nJOINT TEST: YEAR EFFECT")
    print("=" * 75)
    print(f"Terms tested: {len(year_terms)}")
    print(f"Wald statistic: {float(year_test.statistic):.5f}")
    print(f"p-value: {float(year_test.pvalue):.8g}")

    print("\nJOINT TEST: YEAR-FEATURE INTERACTIONS")
    print("=" * 75)
    print(f"Terms tested: {len(interaction_terms)}")
    print(f"Wald statistic: {float(interaction_test.statistic):.5f}")
    print(f"p-value: {float(interaction_test.pvalue):.8g}")

    print("\nSaved:")
    print(SUMMARY_FILE)
    print(COEFFICIENT_FILE)


if __name__ == "__main__":
    main()
