# Data Provenance

## Status: PARTIALLY CONFIRMED — strong structural consistency with IMD 0.25° daily gridded rainfall, but source not definitively confirmed

The rainfall data exhibits **strong structural consistency** with the IMD 0.25° daily gridded rainfall dataset (Pai et al. 2014). The grid resolution, grid origin alignment, units, temporal coverage, and spatial coverage all match. However, **definitive source confirmation requires the original download/processing script or direct value comparison with the IMD source**, neither of which is present in the repository. The distinction between structural consistency and definitive confirmation is critical:

- **Structural consistency** means the dataset's grid, values, and coverage are consistent with IMD data — it *could* be from IMD.
- **Definitive confirmation** would require the original source files, processing code, or a statement from the person who performed the data extraction.

This document reports what can be verified from the available evidence and clearly separates it from what cannot.

---

## 1. Groundwater Observations

- **Source agency:** Central Ground Water Board (CGWB)
- **File:** `data/raw/gwl_manual_quarterly_cgwb_ka_1991_2020.csv`
- **Coverage:** Karnataka state, 1991–2020 (quarterly manual measurements)
- **Target variable:** `Groundwater Level Quarterly Manual (meter)`
- **Status:** Confirmed from README, REPRODUCIBILITY.md, and research_handoff.md
- **Raw file contains:** Station, Agency, State, District, coordinates, `Data Acquisition Time`, and GWL measurements. No rainfall columns in the raw file — rainfall was added externally during processing.

---

## 2. Rainfall Data

### 2.1 Processed columns

| Column | Description |
|---|---|
| `rainfall_30d_mm` | 30-day accumulated rainfall (mm) |
| `rainfall_90d_mm` | 90-day accumulated rainfall (mm) |
| `rainfall_cell_id` | Integer identifier for the assigned grid cell |
| `rainfall_grid_latitude` | Latitude of the assigned grid cell centre |
| `rainfall_grid_longitude` | Longitude of the assigned grid cell centre |
| `qc_suspicious_rainfall_30d` | Boolean QC flag for 30-day rainfall |
| `qc_suspicious_rainfall_90d` | Boolean QC flag for 90-day rainfall |

### 2.2 Evidence from repository documentation

- **README.md:** Lists "India Meteorological Department (IMD): rainfall data" as a **candidate source**, with the caveat: "Sources, licenses, units, temporal coverage, and data-processing decisions will be documented after verification." The word "candidate" indicates this was not confirmed at the time of writing.
- **report/research_handoff.md:** "The rainfall source and complete rainfall preprocessing methodology should be confirmed from the original data-processing documentation before the final report is submitted."
- **Git history:** The dataset was committed in `2c670bc` ("Add groundwater and rainfall feature datasets") with no accompanying metadata, download script, or source documentation.
- **No processing scripts:** No rainfall download, merge, grid-matching, or aggregation scripts exist in the repository. The rainfall columns appear to have been pre-computed externally.

### 2.3 Structural evidence: grid characteristics

| Property | Value in dataset | IMD 0.25° dataset (Pai et al. 2014) | Match? |
|---|---|---|---|
| Grid resolution | 0.25° × 0.25° | 0.25° × 0.25° | ✓ Exact |
| Grid latitude values | 11.75, 12.00, 12.25, ..., 18.25 (27 values) | 6.5, 6.75, ..., 38.5 (129 values) | ✓ Subset |
| Grid longitude values | 74.25, 74.50, 74.75, ..., 78.50 (18 values) | 66.5, 66.75, ..., 100.0 (135 values) | ✓ Subset |
| Grid spacing (latitude) | Uniform 0.25° | Uniform 0.25° | ✓ Exact |
| Grid spacing (longitude) | Uniform 0.25° | Uniform 0.25° | ✓ Exact |
| Grid origin alignment | 11.75 = 6.5 + 21 × 0.25; 74.25 = 66.5 + 31 × 0.25 | First point at (6.5°N, 66.5°E) | ✓ Aligned |
| Units | millimetres | millimetres | ✓ Consistent |
| Temporal coverage | 1996–2020 | 1901–present (updated annually) | ✓ Covers |
| Spatial coverage | Karnataka (11.6°–18.3°N, 74.1°–78.6°E) | All India (6.5°–38.5°N, 66.5°–100°E) | ✓ Contains |

All grid latitude and longitude values in the dataset are exact integer multiples of 0.25° starting from the IMD grid origin (6.5°N, 66.5°E). This is the defining characteristic of the IMD 0.25° grid and represents the strongest piece of structural evidence.

**Important distinction:** This structural consistency means the data *could* be from the IMD 0.25° dataset. It does not prove that it *is* from IMD, because other datasets could theoretically share the same grid (though none of the plausible alternatives do — see Section 2.5). Definitive confirmation requires the original source files or a statement from the data preparer.

### 2.4 Structural evidence: spatial matching

- **Assignment method:** Nearest-grid matching. All 2,434 unique stations are assigned to their closest grid cell centre.
- **Maximum station-to-grid distance:** 0.1768° (= √(0.125² + 0.125²), the half-diagonal of a 0.25° cell). All stations fall within this threshold, confirming nearest-cell assignment.
- **Multi-grid stations:** 44 stations (1.8%) are assigned to multiple grid cells — these are stations with the same name but different coordinates (different physical wells sharing a name, common in CGWB records).
- **Coordinate reference system:** Inferred as WGS84 (standard for both CGWB and IMD); not explicitly documented.

### 2.5 Comparison with alternative sources

| Dataset | Resolution | Coverage starts | Grid alignment | Assessment |
|---|---|---|---|---|
| **IMD 0.25°** (Pai et al. 2014) | 0.25° × 0.25° | 1901 | ✓ Exact match | **Most likely** |
| TRMM/GPM | 0.25° | 1998 | Partial | **Excluded** — our data covers 1996–1997 |
| ERA5 | 0.25° | 1979 | Different grid origin | **Unlikely** — different grid definition, requires unit conversion |
| CHIRPS | 0.05° | 1981 | Does not match | **Excluded** — wrong resolution |
| NCMRWF/IMD merged | 0.25° | 2015 | Matches | **Excluded** — starts too late |

The IMD 0.25° dataset is the only identified source that matches all structural characteristics. The README's explicit mention of IMD as a candidate source further supports this assessment.

### 2.6 Bibliographic reference (verified)

The IMD 0.25° daily gridded rainfall dataset is described in:

> Pai, D. S., Latha Sridhar, M. Rajeevan, O. P. Sreejith, N. S. Satbhai, and B. Mukhopadhyay, 2014: Development of a new high spatial resolution (0.25° × 0.25°) long period (1901–2010) daily gridded rainfall dataset over India. *Mausam*, 65(1), 1–18.

This reference is verified from multiple independent sources (ResearchGate, the IMD journal Mausam's website, and multiple research publications citing it). The dataset is available from:

> India Meteorological Department, Climate Prediction Group, Pune.
> https://www.imdpune.gov.in/Clim_Pred_LRF_New/Grided_Data_Download.php

**Note:** This citation describes the dataset that the structural evidence points to. It does not constitute confirmation that this specific dataset was the source used. If confirmed by the data preparer, this is the citation that should appear in the research paper.

---

## 3. Investigation of Maximum 90-Day Rainfall Value

### 3.1 The observation

The maximum `rainfall_90d_mm` value in the dataset is **7,042.44 mm**, recorded at:

| Field | Value |
|---|---|
| Station | Majali |
| District | Uttara Kannada |
| Grid cell | 149 (15.0°N, 74.25°E) |
| Observation date | 1997-08-25 |
| `rainfall_30d_mm` | 2,957.28 mm |
| `rainfall_90d_mm` | 7,042.44 mm |
| `qc_suspicious_rainfall_30d` | False |
| `qc_suspicious_rainfall_90d` | False |
| Groundwater level | 0.78 m |

### 3.2 Context

- **Uttara Kannada** is a coastal district along the Arabian Sea, adjacent to the Western Ghats. It receives some of the highest rainfall in India — annual totals of 3,000–5,000+ mm are typical.
- **August 25** falls within the southwest monsoon season (June–September), when daily rainfall can exceed 200–300 mm during active spells.
- The **30d/90d ratio** is 2.38, meaning the 30-day window captured ~42% of the 90-day total. This is consistent with monsoon concentration — the most recent 30 days received disproportionately more rain than the preceding 60 days.
- The **same grid cell** (id=149) has a mean 90d rainfall of 1,093 mm across 248 observations. The maximum of 7,042 mm is approximately 6.4× the grid-cell average.

### 3.3 Assessment

A 90-day total of 7,042 mm implies an average of ~78 mm/day over 90 days. For Uttara Kannada during an active monsoon, this is extreme but not physically impossible — daily rainfall exceeding 500 mm has been recorded in this region. However, it is at the upper boundary of plausibility.

**Without the original daily rainfall source data, it is not possible to determine whether this value is:**
1. A genuine extreme monsoon event
2. An artifact of the gridding process (a high-rainfall station disproportionately influencing the grid cell)
3. A data quality issue not caught by the QC flags

**What was done:** The observation was **retained without modification**. The QC flags do not flag it. It falls in the training period (1997, before the 2018 test boundary) and does not directly affect test-period evaluation metrics. Removing it would require evidence that it is erroneous, which is not available.

### 3.4 Top 10 highest 90-day values

The top 10 highest 90d values cluster in two regions/dates:
1. Uttara Kannada grid (14.5–15.0°N, 74.25–74.5°E) in August 1997
2. Shimoga/Udupi grid (13.5°N, 75.0°E) in August 2013 (6 stations sharing the same grid cell, all with 5,982 mm)

These are all in high-rainfall Western Ghats/Coastal Karnataka zones during peak monsoon. The clustering is consistent with the grid assignment — multiple stations in the same grid cell receive the same rainfall value.

---

## 4. Processing Methodology

### What is known

1. Rainfall was assigned to groundwater stations via **nearest-grid matching** from a 0.25° × 0.25° grid.
2. Two rolling-window accumulations were computed: **30-day** and **90-day** cumulative rainfall in mm.
3. Quality-control flags were applied (`qc_suspicious_rainfall_*`), marking 8 and 16 observations respectively.
4. All QC-flagged observations fall in the training period (before 2018-01-01); zero in the test period.

### What is NOT known (requires team confirmation)

| Item | Status | Impact |
|---|---|---|
| Whether the rainfall window includes or excludes the observation date | **UNRESOLVED** | Affects temporal leakage assessment |
| Whether the window is inclusive or exclusive at both endpoints | **UNRESOLVED** | Determines exact number of days aggregated |
| Whether missing daily rainfall values were treated as zero or as missing | **UNRESOLVED** | Affects cumulative totals for periods with missing days |
| The exact daily-to-rolling aggregation code used | **UNRESOLVED** | Cannot independently verify window construction |
| Whether the observation date is the window end or start | **Inferred** window end, not confirmed | Affects whether rainfall is available at prediction time |
| Coordinate reference system | **Inferred** WGS84 | Standard for both CGWB and IMD |
| The download date of the rainfall source | **UNRESOLVED** | Determines which version of the source was used |

### Rolling-window endpoint conventions (UNRESOLVED)

The following conventions remain unresolved and must be confirmed by the data preparer:

1. **Observation-date inclusion:** Does `rainfall_30d_mm` include rainfall on the observation date itself, or only the 30 days preceding it?
2. **Endpoint convention:** Is the window inclusive at both ends (e.g., 30 days including both endpoints = 30 daily values) or exclusive at one end (e.g., 30 days excluding the observation date = 30 daily values from the preceding 30 days)?
3. **Window direction:** Does the window look backward (ending at or before the observation date) or forward (starting at the observation date)? The rolling one-step-ahead prediction setting requires backward-looking windows, but this has not been explicitly confirmed.

These conventions affect whether the rainfall features contain information that would not yet be available at the prediction cutoff. If the observation date is included in the rainfall window and rainfall on that date would not be known at prediction time, this constitutes temporal leakage. However, since the observation date is the date of the groundwater measurement (not a future date), and rainfall data is typically available with a short delay, the practical impact is likely minimal — but it must be confirmed.

---

## 5. Temporal Verification

### 5.1 Groundwater temporal leakage checks

These checks verify that no future groundwater information enters the prediction features.

| Check | Method | Result |
|---|---|---|
| `previous_gwl` uses prior observations only | Sort by `[well_id, observation_date]`, compute lag-1 within well | ✓ PASS — 0 future dates detected |
| No future GWL values in any feature | `previous_gwl` is the only GWL-derived feature; verified as lag-1 | ✓ PASS |
| No training-test observation overlap | Compare well-date keys between train and test splits | ✓ PASS — 0 overlapping pairs |
| Training dates precede test dates | Training: <2018-01-01; Test: ≥2018-01-01 | ✓ PASS |

These checks confirm that groundwater information respects the chronological split and does not contain temporal leakage. They are verified directly from the dataset structure.

### 5.2 Rainfall temporal-window verification

These checks examine whether the rainfall features respect the prediction cutoff. **These checks are limited by the absence of the original daily rainfall source data.**

| Check | Method | Result |
|---|---|---|
| 30d ≤ 90d for all rows | Compare the two columns directly | ✓ PASS — 0 violations (confirms 30d is a sub-window of 90d) |
| Rainfall consistent for same well-date | Group by `[well_id, observation_date]`, check unique rainfall values | ✓ PASS — 0 inconsistencies |
| Rainfall values are non-negative | Check min ≥ 0 | ✓ PASS — min = 0.00 mm for both features |
| Rainfall values are plausible | Compare against known Karnataka rainfall ranges | ✓ PASS — extreme values are in high-rainfall Western Ghats/Coastal zones |
| Window does not include future dates | **Cannot verify** without the original daily rainfall source — requires confirmation of endpoint convention | **UNRESOLVED** |
| Observation-date inclusion | **Cannot verify** without the processing code — requires confirmation from data preparer | **UNRESOLVED** |

**Key distinction:** The groundwater temporal leakage checks (Section 5.1) are fully verified from the dataset structure. The rainfall temporal-window checks (Section 5.2) are only partially verified — structural properties (30d ≤ 90d, consistency, non-negativity) are confirmed, but the endpoint and observation-date inclusion conventions cannot be verified without the original daily rainfall data and processing code.

### 5.3 Missing data pattern

- **746 rows (0.71%)** have missing rainfall values.
- **All 746 missing rows are from a single date: 1996-01-05** (the earliest observation date in the dataset).
- This is consistent with insufficient preceding daily rainfall data for the 30-day and 90-day rolling windows on the first observation date.
- If the rainfall source covers from 1901 (as the IMD dataset does), the missing data is likely a processing boundary effect — the rainfall extraction may have started from late 1995 or 1996, leaving insufficient preceding days for the January 1996 windows.
- This affects 746 training-period observations only and does not impact test-period results.

---

## 6. Limitations

1. **Source not definitively confirmed.** The structural evidence is strongly consistent with IMD 0.25° data, but the original download/processing script is not in the repository. The data preparer (Person 1 or Person 2) must confirm the source.

2. **Aggregation details unknown.** Without the processing code, the exact window construction (inclusive/exclusive endpoints, missing-data treatment, observation-date inclusion) cannot be verified.

3. **Rolling-window endpoint convention unresolved.** Whether the observation date is included in the rainfall window, and whether the window is inclusive or exclusive at its endpoints, remains unknown. This must be confirmed to fully rule out rainfall temporal leakage.

4. **Extreme rainfall values warrant investigation.** The maximum 90d value (7,042 mm) is at the upper boundary of plausibility. Without the original daily source data, it cannot be determined whether this is a genuine extreme or a processing artifact. It was retained without modification.

5. **1996 boundary effect.** All missing rainfall values are in January 1996, likely due to insufficient preceding data for the rolling window. This affects 746 observations (< 1% of the dataset) and does not impact test-period results.

6. **Grid assignment is nearest-cell, not interpolation.** Rainfall values represent grid-cell-average rainfall, not station-specific rainfall. This introduces spatial representation error proportional to the station-to-grid distance (median: 0.10° ≈ 11 km).

7. **No data was modified.** All existing results are preserved. The rainfall dataset was inspected but not altered. No observations were added, removed, or corrected.

---

## 7. Action Required

**Person 1 or Person 2 must provide:**

1. Confirmation that the rainfall source is the IMD 0.25° daily gridded dataset (Pai et al. 2014).
2. The download date and URL used.
3. The exact aggregation code (daily → 30-day and 90-day cumulative).
4. Whether the observation date is included in the rainfall window.
5. Whether the window endpoints are inclusive or exclusive.
6. How missing daily rainfall values were handled.

Until these are provided, rainfall provenance remains **partially confirmed** and the report should qualify any claim about rainfall data quality accordingly.
