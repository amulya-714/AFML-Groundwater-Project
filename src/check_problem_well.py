import pandas as pd

DATA_PATH = "data/processed/groundwater_rainfall_features_qc.csv"
TARGET = "Groundwater Level Quarterly Manual (meter)"

df = pd.read_csv(DATA_PATH, parse_dates=["observation_date"])

# Select the well with the largest observed prediction errors.
well = df[df["Station"] == "Govt L V Polytechnic"].copy()

well = well.sort_values("observation_date")

well["previous_gwl"] = well[TARGET].shift(1)
well["change_from_previous_m"] = (
    well[TARGET] - well["previous_gwl"]
)

print("\n=== GOVT L V POLYTECHNIC: GROUNDWATER HISTORY ===")
print(f"Number of observations: {len(well)}")

print("\nGroundwater level summary:")
print(well[TARGET].describe().round(3))

print("\nObservations around the extreme errors:")
columns = [
    "observation_date",
    "Station",
    "Latitude",
    "Longitude",
    TARGET,
    "previous_gwl",
    "change_from_previous_m",
]

print(
    well.loc[
        well["observation_date"].between(
            "2016-01-01", "2020-12-31"
        ),
        columns,
    ].to_string(index=False)
)

print("\nObservations before 2018 with groundwater level above 80 m:")
print(
    well.loc[
        (well["observation_date"] < "2018-01-01")
        & (well[TARGET] > 80),
        ["observation_date", TARGET],
    ].to_string(index=False)
)
