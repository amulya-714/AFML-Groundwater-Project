# Paper Directory — AFML Groundwater ML Project

This directory contains the complete IEEE conference paper for the AFML
Groundwater ML project, produced as Person 4's contribution.

---

## Directory Structure

```
paper/
├── main.tex             # Complete IEEE-format LaTeX paper (8 sections)
├── references.bib       # BibTeX bibliography (verified entries only)
├── figures/             # Copies of project figures (originals in results/figures/)
│   ├── figure1_baseline_comparison.png
│   ├── figure2_input_novelty.png
│   ├── figure3_novelty_types.png
│   ├── figure4_temporal_degradation.png
│   ├── figure5_feature_relationships.png
│   ├── figure6_validated_novelty.png
│   ├── figure7_repeated_removal.png
│   ├── figure8_familiar_input_temporal_comparison.png
│   ├── figure9_temporal_residuals.png
│   ├── baseline_comparison.png       (Person 3)
│   ├── familiar_unfamiliar_comparison.png  (Person 3)
│   ├── rainfall_comparison.png       (Person 3)
│   ├── same_well_comparison.png      (Person 3)
│   ├── bootstrap_results.png         (Person 3)
│   └── coverage_comparison.png       (Person 3)
├── tables/              # Reserved for standalone table files (currently empty)
├── README.md            # This file
└── VERIFICATION_REPORT.md   # Audit trail documenting sources and gaps
```

> **Important:** The `figures/` directory contains copies only.  
> Originals are at `results/figures/` and `results/figures/person3/`.  
> Never delete or move the originals.

---

## How to Compile

### Recommended: latexmk (handles multiple passes automatically)

```bash
cd /Users/Mikey/GitHub/AFML-Groundwater-Project/paper
latexmk -pdf main.tex
```

Clean up auxiliary files after compilation:

```bash
latexmk -c
```

### Alternative: manual pdflatex + bibtex

```bash
cd /Users/Mikey/GitHub/AFML-Groundwater-Project/paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Three pdflatex passes are needed to resolve all cross-references and
citation numbers.

---

## Required LaTeX Packages

The paper uses the following packages (all part of standard TeX
distributions such as TeX Live 2022+ or MiKTeX):

| Package      | Purpose                                      |
|--------------|----------------------------------------------|
| `IEEEtran`   | IEEE conference paper document class         |
| `cite`       | IEEE-style numbered citations                |
| `amsmath`    | Mathematical typesetting                     |
| `amssymb`    | Additional math symbols                      |
| `graphicx`   | Figure inclusion (`\includegraphics`)         |
| `booktabs`   | Professional table rules (`\toprule`, etc.)  |
| `array`      | Extended table column specifications         |
| `url`        | URL formatting                               |
| `hyperref`   | Clickable cross-references and links         |
| `xspace`     | Smart spacing after macros                   |
| `multirow`   | Multi-row table cells                        |
| `siunitx`    | SI unit typesetting                          |

Install TeX Live on macOS:

```bash
brew install --cask mactex
```

Or install the smaller BasicTeX + required packages:

```bash
brew install --cask basictex
sudo tlmgr update --self
sudo tlmgr install IEEEtran booktabs siunitx multirow hyperref
```

---

## Paper Contents

The paper is structured as a standard IEEE conference paper:

1. **Abstract** — Research question, methods, key verified results
2. **Introduction** — Groundwater importance, ML challenges, contributions
3. **Related Work** — ML groundwater prediction, distribution shift,
   concept drift, robustness evaluation
4. **Dataset and Study Design** — CGWB Karnataka dataset description
5. **Methodology** — Baseline models, novelty analysis, temporal analyses,
   controlled removal, Person 2 gap documentation, Person 3 evaluation
6. **Experimental Results** — 11 tables, 9 figures, all with real numbers
7. **Discussion** — Interpretation of findings, between-well confound,
   practical implications
8. **Limitations** — 13 documented limitations
9. **Conclusion and Future Work**

---

## Notes on Data Sources

All numerical values in tables are extracted directly from CSV files in
`results/` and `results/person3/`.
No values are fabricated or estimated.

The planned reliability-aware modelling (Person 2) is documented as absent
because the corresponding result files do not exist in the repository.

---

## Compilation Status

LaTeX (pdflatex/latexmk) was not found on the build machine at the time
of creation. See `VERIFICATION_REPORT.md` for the full compilation status
and any actions needed.
