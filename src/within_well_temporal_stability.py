"""Within-well temporal stability analysis.

Tests whether the relationship between familiar input conditions
and Random Forest prediction residuals changes across 2018-2020.

Uses well-level demeaning to control for time-invariant
differences between wells.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data/processed/groundwater_rainfall_features_qc.csv"
PREDICTION_FILE = ROOT / "results/controlled_removal_predictions.csv"

SUMMARY_FILE = ROOT / "results/within_well_temporal_summary.csv"
DETAIL_FILE = ROOT / "results/within_well_temporal_observations.csv"
COEFFICIENT_FILE = ROOT / "results/within_well_temporal_coefficients.csv"

TARGET = "Groundwater Level Quarterly Manual (meter)"

RAW_FEATURES = [
    "rainfall_30d_mm",
    "rainfall_90d_mm",
    "previous_gwl",
    "month_sin",
    "month_cos",
]

YEARS = [2018, 2019, 2020]


def main():

    print("Loading dataset and existing predictions...")

    data = pd.read_csv(
        DATA_FILE,
        parse_dates=["observation_date"],
    )

    predictions = pd.read_csv(
        PREDICTION_FILE,
        parse_dates=["date"],
    )

    # Reconstruct well identifiers.
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

    # Reconstruct seasonal features.
    month = data["observation_date"].dt.month

    data["month_sin"] = np.sin(2 * np.pi * month / 12)
    data["month_cos"] = np.cos(2 * np.pi * month / 12)

    # Training-period statistics only.
    training = data[
        data["observation_date"] < "2018-01-01"
    ].copy()

    # Keep familiar test observations.
    familiar = predictions[
        predictions["novelty_group"] == "Familiar"
    ].copy()

    inputs = data[
        ["well_id", "observation_date"] + RAW_FEATURES
    ].copy()

    # Validate merge keys.
    if inputs.duplicated(["well_id", "observation_date"]).any():
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

    print(f"Familiar predictions: {len(familiar):,}")
    print(f"Unmatched observations: {unmatched:,}")

    if unmatched:
        raise ValueError("Some predictions could not be matched.")

    merged = merged.drop(
        columns=["_merge", "observation_date"]
    )

    merged["year"] = merged["date"].dt.year

    merged = merged[
        merged["year"].isin(YEARS)
    ].copy()

    # Positive residual means overprediction.
    merged["residual_m"] = (
        merged["full_model_prediction_m"] -
        merged["actual_gwl_m"]
    )

    # Standardize using training-period statistics.
    for feature in RAW_FEATURES:

        mean = training[feature].mean()
        std = training[feature].std()

        if pd.isna(std) or std == 0:
            raise ValueError(f"Invalid standard deviation: {feature}")

        merged[f"{feature}_z"] = (
            merged[feature] - mean
        ) / std

    standardized_features = [
        f"{feature}_z" for feature in RAW_FEATURES
    ]

    required = (
        ["well_id", "year", "residual_m"] +
        standardized_features
    )

    merged = merged.dropna(subset=required).copy()

    # Retain only wells represented in ALL three years.
    years_per_well = (
        merged.groupby("well_id")["year"].nunique()
    )

    balanced_wells = years_per_well[
        years_per_well == len(YEARS)
    ].index

    merged = merged[
        merged["well_id"].isin(balanced_wells)
    ].copy()

    print("\nBALANCED WITHIN-WELL SAMPLE")
    print("=" * 55)
    print(f"Observations: {len(merged):,}")
    print(f"Wells represented in all three years: {len(balanced_wells):,}")
    print("\nObservations by year:")
    print(merged.groupby("year").size().to_string())

    if len(balanced_wells) < 30:
        raise ValueError("Too few balanced wells for this analysis.")

    # Create year indicators.
    year_dummies = pd.get_dummies(
        merged["year"],
        prefix="year",
        drop_first=True,
        dtype=float,
    )

    # Build the original standardized input matrix.
    X_inputs = merged[standardized_features].astype(float).copy()

    # Add year indicators.
    X_year = pd.concat(
        [X_inputs, year_dummies],
        axis=1,
    )

    # Add year-by-feature interactions.
    X_interactions = X_year.copy()

    for year_column in year_dummies.columns:

        for feature in standardized_features:

            name = f"{year_column}:{feature}"

            X_interactions[name] = (
                year_dummies[year_column] *
                merged[feature]
            )

    # Within transformation:
    # subtract each well's average from every variable.
    y = merged["residual_m"].astype(float)

    y_dm = y - y.groupby(merged["well_id"]).transform("mean")

    def demean(frame):
        return frame - frame.groupby(
            merged["well_id"]
        ).transform("mean")

    X1 = demean(X_inputs)
    X2 = demean(X_year)
    X3 = demean(X_interactions)

    print("\nFitting well-fixed-effects models...")
    print("Using well-clustered standard errors...")

    model1 = sm.OLS(y_dm, X1).fit(
        cov_type="cluster",
        cov_kwds={"groups": merged["well_id"]},
    )

    model2 = sm.OLS(y_dm, X2).fit(
        cov_type="cluster",
        cov_kwds={"groups": merged["well_id"]},
    )

    model3 = sm.OLS(y_dm, X3).fit(
        cov_type="cluster",
        cov_kwds={"groups": merged["well_id"]},
    )

    # Joint test of year indicators.
    year_terms = list(year_dummies.columns)

    year_test = model2.wald_test(
        np.eye(len(model2.params))[
            [model2.params.index.get_loc(term) for term in year_terms]
        ],
        scalar=True,
    )

    # Joint test of all year-feature interactions.
    interaction_terms = [
        term for term in X3.columns
        if ":" in term
    ]

    interaction_indices = [
        model3.params.index.get_loc(term)
        for term in interaction_terms
    ]

    interaction_test = model3.wald_test(
        np.eye(len(model3.params))[interaction_indices],
        scalar=True,
    )

    # Save model comparison.
    summary = pd.DataFrame([
        {
            "model": "Inputs only",
            "n_observations": int(model1.nobs),
            "n_wells": len(balanced_wells),
            "R_squared": model1.rsquared,
            "adjusted_R_squared": model1.rsquared_adj,
            "AIC": model1.aic,
        },
        {
            "model": "Inputs + year",
            "n_observations": int(model2.nobs),
            "n_wells": len(balanced_wells),
            "R_squared": model2.rsquared,
            "adjusted_R_squared": model2.rsquared_adj,
            "AIC": model2.aic,
        },
        {
            "model": "Inputs + year + interactions",
            "n_observations": int(model3.nobs),
            "n_wells": len(balanced_wells),
            "R_squared": model3.rsquared,
            "adjusted_R_squared": model3.rsquared_adj,
            "AIC": model3.aic,
        },
    ])

    coefficients = pd.concat(
        [
            model1.summary2().tables[1].assign(model="Inputs only"),
            model2.summary2().tables[1].assign(model="Inputs + year"),
            model3.summary2().tables[1].assign(
                model="Inputs + year + interactions"
            ),
        ]
    ).reset_index(names="term")

    summary.to_csv(SUMMARY_FILE, index=False)
    coefficients.to_csv(COEFFICIENT_FILE, index=False)
    model3.cov_params().to_csv(
    COEFFICIENT_FILE.parent / "within_well_temporal_covariance.csv")

    # Save the exact balanced observations used.
    merged.to_csv(DETAIL_FILE, index=False)

    print("\nMODEL COMPARISON")
    print("=" * 75)
    print(summary.round(5).to_string(index=False))

    print("\nJOINT TEST: YEAR EFFECTS")
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
    print(DETAIL_FILE)


if __name__ == "__main__":
    main()
