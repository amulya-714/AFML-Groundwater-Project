# Feature Construction Documentation

## Dataset

**File:** `data/processed/groundwater_rainfall_features_qc.csv`
**Rows:** 104,781 | **Columns:** 30

## Target variable

`Groundwater Level Quarterly Manual (meter)` — CGWB manual groundwater-level measurement in metres.

## Model features (used by baseline_model.py)

| Feature | Construction | Units |
|---|---|---|
| `Latitude` | Station latitude from CGWB records | decimal degrees |
| `Longitude` | Station longitude from CGWB records | decimal degrees |
| `rainfall_30d_mm` | 30-day rolling cumulative rainfall (pre-computed in dataset) | mm |
| `rainfall_90d_mm` | 90-day rolling cumulative rainfall (pre-computed in dataset) | mm |
| `previous_gwl` | Lag-1 of target within well, sorted chronologically by `observation_date` | metres |
| `month_sin` | `sin(2π × month / 12)` computed from `observation_date` | dimensionless |
| `month_cos` | `cos(2π × month / 12)` computed from `observation_date` | dimensionless |

## Well identifier

`well_id = Station_Latitude_Longitude` (string concatenation)

## Feature construction verification (Person 3)

### previous_gwl

- **Verified:** sorted by `[well_id, observation_date]`, then `groupby("well_id")[target].shift(1)`.
- **Result:** PASS — for 5 sampled wells, all `previous_gwl` values match the preceding observation's target exactly.
- **No future leakage:** by construction, `previous_gwl` uses only prior observations.
- **Note:** in the rolling one-step-ahead prediction setting, the previous measurement is assumed available at prediction time.

### month_sin / month_cos

- **Construction:** `np.sin(2 * np.pi * month / 12)` and `np.cos(2 * np.pi * month / 12)` where `month = observation_date.dt.month`.
- **Status:** NOT in the processed dataset; constructed inside `baseline_model.py` and all analysis scripts that need them.

### rainfall_30d_mm / rainfall_90d_mm

- **Status:** Pre-computed in the processed dataset.
- **Structural check:** `rainfall_90d_mm >= rainfall_30d_mm` for 100% of rows. PASS.
- **Missing values:** 746 rows (0.71%) have NaN for both rainfall features.
- **Cannot independently verify:** window construction (inclusive/exclusive), missing-day treatment, prediction-cutoff alignment. See `data_provenance.md`.

## Chronological split

| Dataset | Condition |
|---|---|
| Training | `observation_date < 2018-01-01` |
| Test | `observation_date >= 2018-01-01` |

- Training: 87,426 rows (1996-05-15 to 2017-11-05)
- Test: 14,863 rows (2018-01-05 to 2020-11-30)
- **Verified:** zero training-test overlap (0 duplicate well-date keys).

## Prediction setting

Rolling one-step-ahead prediction: the previous groundwater measurement for each well is assumed to be available at the prediction cutoff. This is NOT a multi-quarter-ahead forecast.

## Rows lost to feature construction

After constructing `previous_gwl` (which requires at least 2 observations per well), the model-ready dataset contains 102,289 rows (2,492 rows lost to the lag).
