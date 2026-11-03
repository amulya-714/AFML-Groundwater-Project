from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

FIGURES.mkdir(parents=True, exist_ok=True)


def save_figure(filename):
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


# --------------------------------------------------
# FIGURE 1: BASELINE MODEL COMPARISON
# --------------------------------------------------

df = pd.read_csv(RESULTS / "baseline_metrics.csv")

plt.figure(figsize=(8, 5))

x = np.arange(len(df))
width = 0.35

plt.bar(x - width / 2, df["MAE_m"], width, label="MAE")
plt.bar(x + width / 2, df["RMSE_m"], width, label="RMSE")

plt.xticks(x, df["model"], rotation=10)
plt.ylabel("Error (metres)")
plt.title("Baseline Model Performance")
plt.legend()

save_figure("figure1_baseline_comparison.png")


# --------------------------------------------------
# FIGURE 2: INPUT NOVELTY VS PREDICTION ERROR
# --------------------------------------------------

df = pd.read_csv(RESULTS / "input_extrapolation_summary.csv")

df = df[df["novelty_type"] == "Full-feature"]

order = ["Familiar", "Poorly represented", "Extreme novelty"]

df["novelty_group"] = pd.Categorical(
    df["novelty_group"],
    categories=order,
    ordered=True
)

df = df.sort_values("novelty_group")

plt.figure(figsize=(8, 5))

plt.bar(
    df["novelty_group"].astype(str),
    df["MAE_m"]
)

plt.ylabel("Mean Absolute Error (metres)")
plt.xlabel("Input Novelty")
plt.title("Prediction Error Across Input Novelty Groups")

save_figure("figure2_input_novelty.png")


# --------------------------------------------------
# FIGURE 3: GEOGRAPHIC VS ENVIRONMENTAL NOVELTY
# --------------------------------------------------

df = pd.read_csv(RESULTS / "novelty_type_summary.csv")

plt.figure(figsize=(9, 5))

for novelty_type in df["novelty_type"].unique():

    subset = df[df["novelty_type"] == novelty_type].copy()

    subset = subset.sort_values("quartile")

    plt.plot(
        subset["quartile"].astype(str),
        subset["MAE_m"],
        marker="o",
        label=novelty_type.capitalize()
    )

plt.xlabel("Novelty Quartile")
plt.ylabel("Mean Absolute Error (metres)")
plt.title("Prediction Error Across Different Types of Novelty")
plt.legend()

save_figure("figure3_novelty_types.png")


# --------------------------------------------------
# FIGURE 4: TEMPORAL ERROR INCREASE
# --------------------------------------------------

df = pd.read_csv(RESULTS / "within_well_clustered_bootstrap.csv")

row = df.iloc[0]

labels = ["2018", "2020"]

values = [
    row["mean_MAE_2018_m"],
    row["mean_MAE_2020_m"]
]

plt.figure(figsize=(7, 5))

plt.bar(labels, values)

plt.ylabel("Mean Absolute Error (metres)")
plt.title("Prediction Error for the Same Groundwater Wells")

plt.text(
    0.5,
    max(values) * 0.9,
    f"Increase: {row['mean_error_increase_m']:.3f} m\n"
    f"95% CI: [{row['bootstrap_CI_lower_m']:.3f}, "
    f"{row['bootstrap_CI_upper_m']:.3f}]",
    ha="center"
)

save_figure("figure4_temporal_degradation.png")


# --------------------------------------------------
# FIGURE 5: FEATURE RELATIONSHIP INSTABILITY
# --------------------------------------------------

df = pd.read_csv(RESULTS / "within_well_temporal_coefficients.csv")

df = df[
    df["model"] == "Inputs + year + interactions"
].copy()

covariance = pd.read_csv(
    RESULTS / "within_well_temporal_covariance.csv",
    index_col=0
)

features = [
    "rainfall_30d_mm_z",
    "rainfall_90d_mm_z",
    "previous_gwl_z",
    "month_sin_z",
    "month_cos_z"
]

records = []

for feature in features:

    base = df[df["term"] == feature]

    if base.empty:
        continue

    base = base.iloc[0]

    base_coef = base["Coef."]

    # 2018 is the reference year.
    base_variance = covariance.loc[feature, feature]
    base_se = np.sqrt(max(base_variance, 0))

    records.append({
        "feature": feature,
        "year": "2018",
        "coefficient": base_coef,
        "lower": base_coef - 1.96 * base_se,
        "upper": base_coef + 1.96 * base_se
    })

    for year in ["2019", "2020"]:

        term = f"year_{year}:{feature}"

        interaction = df[df["term"] == term]

        if interaction.empty:
            continue

        interaction = interaction.iloc[0]

        interaction_coef = interaction["Coef."]

        # Combined coefficient for this year.
        coef = base_coef + interaction_coef

        # Correct variance, including covariance.
        variance = (
            covariance.loc[feature, feature]
            + covariance.loc[term, term]
            + 2 * covariance.loc[feature, term]
        )

        se = np.sqrt(max(variance, 0))

        records.append({
            "feature": feature,
            "year": year,
            "coefficient": coef,
            "lower": coef - 1.96 * se,
            "upper": coef + 1.96 * se
        })

plot_df = pd.DataFrame(records)

fig, axes = plt.subplots(
    len(features),
    1,
    figsize=(9, 14)
)

for ax, feature in zip(axes, features):

    subset = plot_df[
        plot_df["feature"] == feature
    ].copy()

    subset["year"] = pd.Categorical(
        subset["year"],
        categories=["2018", "2019", "2020"],
        ordered=True
    )

    subset = subset.sort_values("year")

    ax.errorbar(
        subset["year"].astype(str),
        subset["coefficient"],
        yerr=[
            subset["coefficient"] - subset["lower"],
            subset["upper"] - subset["coefficient"]
        ],
        marker="o",
        capsize=4
    )

    ax.axhline(0, linestyle="--", linewidth=0.8)
    ax.set_title(feature)
    ax.set_ylabel("Coefficient")

axes[-1].set_xlabel("Year")

fig.suptitle(
    "Changes in Feature Relationships Across Years",
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    FIGURES / "figure5_feature_relationships.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: figure5_feature_relationships.png")

print("\nALL FIGURES GENERATED SUCCESSFULLY.")
print(f"Location: {FIGURES}")
