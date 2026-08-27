# GWTC-5.0 full-catalog generalization extension

This directory contains a post-lock exploratory/generalization analysis of the
complete 104-event GWTC-5.0 XPHM-XPNR catalog universe. It preserves the
original 18-event confirmation and nine-event prospective replication as
separate locked analyses.

## Final state (2026-08-26)

- all 104 expected catalog files are present and verified against the official
  Zenodo MD5 manifest
- 93 model pairs pass the automated strict matched-settings rules
- all 104 pairs contain the posterior fields required for the H3 analysis
- the primary non-development population contains 78 strict events from the
  fixed 87-event non-development frame
- primary H1 result: Spearman rho = 0.5025, bootstrap 95% interval 0.3092 to
  0.6584, Holm-adjusted one-sided permutation p = 2.00e-5
- primary H3 ordering result: 75/78 positive deltas (96.2%), with median delta
  0.05250 and bootstrap 95% interval 0.04115 to 0.05802
- the screening proxy and complete-posterior screening endpoint have Spearman
  rho = 0.9788 in the primary population
- strict matched-settings result: 89/93 positive deltas (95.7%)
- all-usable-pairs descriptive result: 99/104 positive deltas (95.2%)
- the strict-population median delta is 0.05598 (bootstrap 95% interval:
  0.04606 to 0.06210)
- all 27 locked confirmation/replication delta signs reproduce, with a maximum
  absolute numerical difference of approximately 1.39e-17

The full-catalog result is final for this version of the analysis. It is a
post-lock generalization analysis, not a replacement for the original frozen
confirmation or prospective replication.

The five pilot events are labeled `development_pilot`. They were informally
chosen from the pool during method development and are not described as
randomly or prospectively selected.

## Directory guide

- `protocol/`: the analysis rules and interpretation boundaries fixed before
  completing the 104-event outcome analysis
- `design/official_zenodo_md5sums.txt`: the official release checksum manifest
- `design/frozen_inputs/`: compact ranking, development, confirmation, and
  replication design inputs copied into this extension for self-contained
  reproduction
- `reference_results/`: frozen confirmatory and replication parameter metrics
  used only for locked-result regression checks
- `analysis/build_full_catalog_inventory.py`: exact-filename and checksum
  inventory; reads no HDF5 data
- `analysis/GWTC5_full_catalog_metadata_audit.py`: outcome-blind matched-settings
  audit; reads posterior schema and row counts but no posterior values
- `analysis/GWTC5_full_catalog_generalization_analysis.py`: NW1 calculations,
  independent two-parameter census, subgroups, and locked-result consistency
  check
- `analysis/full_catalog_generalization_analysis.py`: four-parameter H1,
  H3, screening-proxy, robustness, finite-sample, and heterogeneity analyses
- `analysis/full_catalog_mechanism_generalization.py`: exploratory joint
  `(q, chi_eff)` and `chi_p` generalization analyses
- `analysis/verify_full_catalog_outputs.py`: independent structural and
  consistency checks on the completed output tables
- `analysis/make_full_catalog_h3_figure.js`: generates the final sorted-delta
  figure in PNG and SVG formats
- `results/analysis/`: final event-level and population-level result tables
- `results/generalization_full/`: final four-parameter generalization,
  robustness, and heterogeneity tables
- `results/mechanism_full/`: final exploratory mechanism tables and manifest
- `results/figures/` and `results/figures_full/`: final catalog-wide figures
- `results/FULL_CATALOG_GENERALIZATION_FINAL_SUMMARY.md`: concise scientific
  interpretation, exceptions, and claim boundaries
- `REPRODUCTION_COMMANDS.md`: exact run order using only the compact package
  plus the separately downloaded official HDF5 files

## Required Python packages

- Python 3.14
- NumPy 2.5.1
- SciPy 1.17.0
- h5py 3.16.0

The analysis records the full runtime versions and hashes in its JSON
manifest.

## Reproduction order

1. Run `build_full_catalog_inventory.py` without `--skip-md5` and confirm that
   the summary reports exactly 104 verified files.
2. Run `GWTC5_full_catalog_metadata_audit.py --overwrite`.
3. Review every recorded non-Tier-A difference without looking at its outcome.
4. Run `GWTC5_full_catalog_generalization_analysis.py --overwrite` without
   `--allow-incomplete`.
5. Run `verify_full_catalog_outputs.py` and require every check to pass.
6. Run `full_catalog_generalization_analysis.py` with the verified inventory,
   outcome-blind metadata audit, frozen ranking table, and frozen metric core.
7. Run `full_catalog_mechanism_generalization.py` with those same audited
   inputs.
8. Run `make_full_catalog_h3_figure.js` and
   `figures/make_full_catalog_figures.py` to regenerate the figures.

The final scientific interpretation must lead with the strict matched-settings
population. The all-usable-pairs population is broader but descriptive because
non-waveform settings can contribute to its differences.
