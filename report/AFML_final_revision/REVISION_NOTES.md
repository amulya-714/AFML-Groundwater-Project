# Revision notes

- Updated title to describe measured input novelty and temporal performance degradation without implying direct physical-extreme classification.
- Replaced extreme-removal terminology with top-1%-novelty removal terminology.
- Corrected rainfall provenance statement to match the supplied repository archive: its `data/raw/` directory contains the groundwater CSV, not rainfall NetCDF files.
- Added high-rainfall threshold and paired within-well methodology from `src/person3_grouped_evaluation.py` and the corresponding result artifacts.
- Documented within-well Hungarian one-to-one matching, standardised Euclidean distance and 1.0 caliper from `src/within_well_matched_input_analysis.py`.
- Corrected matched-input Figure 2 title/caption and regenerated chart from `results/within_well_matched_input_summary.csv`.
- Clarified single-seed Table II versus ten-seed Table IV.
- Clarified that rainfall-only results in Figure 1 are support groups, not quartiles; the archived `novelty_type_summary.csv` has no rainfall-only quartile values, so none were invented.

## Outstanding source discrepancy
The review states that rainfall NetCDF files `RF25_ind1996_rfp25.nc` through `RF25_ind2020_rfp25.nc` are present in `data/raw/`. They are not present in the repository ZIP supplied for this revision. If those files exist in a newer branch/repository, provide that archive and the provenance paragraph can be updated to reflect it.

## Verification caution
The supplied repository contains both `results/within_well_matched_input_summary.csv` (2,077 pairs; increase 0.506 m) and `results/matched_input_temporal_summary.csv` (17 pairs; inconsistent alternate output). This revision uses the former because its script and summary correspond to the paper's 2,077-pair result. The discrepancy should be resolved in the repository before submission.
