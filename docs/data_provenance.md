# Data Provenance

## Status: PARTIALLY RESOLVED

This document records what is known and unknown about the rainfall data provenance.

---

## Groundwater observations

- **Source:** Central Ground Water Board (CGWB), Karnataka
- **File:** `data/raw/gwl_manual_quarterly_cgwb_ka_1991_2020.csv`
- **Coverage:** 1991–2020, Karnataka state
- **Frequency:** Quarterly manual measurements
- **Target variable:** `Groundwater Level Quarterly Manual (meter)`
- **Status:** Raw file referenced in `README.md`; the file itself was not present at audit time in the thrupthi912 repository but exists in the amulya-714 source.

## Rainfall features

- **Processed columns:** `rainfall_30d_mm`, `rainfall_90d_mm`
- **QC columns:** `qc_suspicious_rainfall_30d`, `qc_suspicious_rainfall_90d`
- **Grid columns:** `rainfall_cell_id`, `rainfall_grid_latitude`, `rainfall_grid_longitude`

### What is known

1. Rainfall was matched to groundwater stations via a spatial grid (278 unique grid cells, 27 unique latitudes, 18 unique longitudes).
2. Two rolling-window aggregations were constructed: 30-day and 90-day cumulative rainfall in millimetres.
3. Quality-control flags were applied (`qc_suspicious_rainfall_*`), marking 8 and 16 observations respectively.
4. All QC-flagged observations fall in the training period (before 2018-01-01); zero in the test period.

### What is NOT known (UNRESOLVED)

| Item | Status |
|---|---|
| Rainfall data provider / agency | **Unknown** |
| Dataset / product name | **Unknown** |
| Original URL or archive reference | **Unknown** |
| Spatial resolution of original rainfall grid | **Inferred:** ~0.25° from grid lat/long spacing, but unconfirmed |
| Temporal resolution of original rainfall data | **Unknown** (daily assumed, not confirmed) |
| Units of original rainfall data | **Inferred:** millimetres, based on column names |
| Coverage dates of rainfall source | **Unknown** |
| Retrieval date | **Unknown** |
| Spatial matching procedure | **Inferred:** nearest-grid assignment using station lat/long to grid cell centroid |
| Coordinate reference system | **Unknown** (assumed WGS84) |
| Missing-data treatment in rainfall source | **Unknown** |
| Aggregation method (inclusive/exclusive window) | **Unknown** |
| Whether observation date is included in the window | **Unknown** |
| Whether missing rainfall days were treated as zero or missing | **Unknown** |

### Structural checks performed

- `rainfall_90d_mm >= rainfall_30d_mm` for 100% of rows (structural sanity confirmed)
- Rainfall values are non-negative (min = 0.00 mm for both features)
- 746 rows (0.71%) have missing rainfall values

### Action required

**Person 1 or Person 2 must provide:**

1. The original rainfall source filename and provider.
2. The download URL or archive reference.
3. The spatial resolution and temporal resolution.
4. The exact aggregation procedure (window construction, missing-data treatment).
5. The spatial matching procedure (coordinate reference system, matching algorithm).

Until these are provided, rainfall provenance remains **UNRESOLVED** and any claim about rainfall data quality must be qualified.
