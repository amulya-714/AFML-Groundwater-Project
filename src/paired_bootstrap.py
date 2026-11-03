import numpy as np
import pandas as pd

df = pd.read_csv("results/baseline_predictions.csv", parse_dates=["date"])
features = pd.read_csv(
    "data/processed/groundwater_rainfall_features_qc.csv",
    parse_dates=["observation_date"]
)

features["well_id"] = (
    features["Station"].astype(str) + "_" +
    features["Latitude"].astype(str) + "_" +
    features["Longitude"].astype(str)
)
features = features[features["observation_date"] >= "2018-01-01"]
df = df.merge(
    features[["well_id", "observation_date", "rainfall_90d_mm"]],
    left_on=["well_id", "date"],
    right_on=["well_id", "observation_date"],
    validate="one_to_one"
)

train = pd.read_csv(
    "data/processed/groundwater_rainfall_features_qc.csv",
    parse_dates=["observation_date"]
)
cutoff = train.loc[
    train["observation_date"] < "2018-01-01", "rainfall_90d_mm"
].quantile(0.95)

df["error"] = (
    df["actual_gwl_m"] - df["random_forest_prediction_m"]
).abs()
df["extreme"] = df["rainfall_90d_mm"] >= cutoff

groups = df.groupby(["well_id", "extreme"])["error"].mean().unstack()
groups = groups.dropna(subset=[False, True])
differences = (groups[True] - groups[False]).to_numpy()

rng = np.random.default_rng(42)
boot = np.array([
    rng.choice(differences, size=len(differences), replace=True).mean()
    for _ in range(10000)
])

print("Wells with both groups:", len(differences))
print(f"Mean paired MAE difference (extreme - non-extreme): {differences.mean():.4f} m")
print(f"95% bootstrap CI: [{np.quantile(boot, .025):.4f}, {np.quantile(boot, .975):.4f}] m")
print("Negative means lower error during extreme rainfall.")
print("Saved result: results/paired_bootstrap.txt")

with open("results/paired_bootstrap.txt", "w") as f:
    f.write(f"Wells: {len(differences)}\n")
    f.write(f"Mean difference: {differences.mean():.6f} m\n")
    f.write(f"95% CI: [{np.quantile(boot, .025):.6f}, {np.quantile(boot, .975):.6f}] m\n")
