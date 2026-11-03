from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

path = Path("data/processed/groundwater_rainfall_features_qc.csv")
out = Path("results")
out.mkdir(exist_ok=True)

df = pd.read_csv(path, parse_dates=["observation_date"])
target = "Groundwater Level Quarterly Manual (meter)"

df["well_id"] = (
    df["Station"].astype(str) + "_" +
    df["Latitude"].astype(str) + "_" +
    df["Longitude"].astype(str)
)
df = df.sort_values(["well_id", "observation_date"]).copy()
df["previous_gwl"] = df.groupby("well_id")[target].shift(1)
month = df["observation_date"].dt.month
df["month_sin"] = np.sin(2 * np.pi * month / 12)
df["month_cos"] = np.cos(2 * np.pi * month / 12)

features = [
    "Latitude", "Longitude", "rainfall_30d_mm",
    "rainfall_90d_mm", "previous_gwl", "month_sin", "month_cos"
]
df = df.dropna(subset=features + [target])
train = df[df["observation_date"] < "2018-01-01"]
test = df[df["observation_date"] >= "2018-01-01"]

if train.empty or test.empty:
    raise ValueError("Training or test set is empty; check dates and missing values.")

model = RandomForestRegressor(
    n_estimators=300, min_samples_leaf=3,
    random_state=42, n_jobs=-1
)
model.fit(train[features], train[target])

predictions = model.predict(test[features])
persistence = test["previous_gwl"].to_numpy()
actual = test[target].to_numpy()

def metrics(name, pred):
    return {
        "model": name,
        "test_rows": len(actual),
        "MAE_m": mean_absolute_error(actual, pred),
        "RMSE_m": np.sqrt(mean_squared_error(actual, pred)),
        "R2": r2_score(actual, pred),
    }

results = pd.DataFrame([
    metrics("Persistence baseline", persistence),
    metrics("Random Forest", predictions),
])
results.to_csv(out / "baseline_metrics.csv", index=False)

pd.DataFrame({
    "date": test["observation_date"].to_numpy(),
    "well_id": test["well_id"].to_numpy(),
    "actual_gwl_m": actual,
    "persistence_prediction_m": persistence,
    "random_forest_prediction_m": predictions,
}).to_csv(out / "baseline_predictions.csv", index=False)

print("Training period:", train["observation_date"].min().date(),
      "to", train["observation_date"].max().date())
print("Testing period:", test["observation_date"].min().date(),
      "to", test["observation_date"].max().date())
print("Training rows:", len(train), "| Test rows:", len(test))
print("\n", results.to_string(index=False))
print("\nSaved results/baseline_metrics.csv")
print("Saved results/baseline_predictions.csv")
