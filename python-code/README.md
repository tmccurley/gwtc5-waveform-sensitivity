# Earlier and site-maintenance Python code

This directory preserves every Python script and notebook from the earlier figures-and-code collection, plus newer repository versions that were not byte-identical to an archived copy. The later 104-event extension is a checksum-preserved capsule under [`../reproducibility/full-catalog-generalization/`](../reproducibility/full-catalog-generalization/) because moving its internal scripts would break their relative-path assumptions and frozen provenance.

| Directory | Purpose |
| --- | --- |
| `current-analysis/` | Latest repository versions of the confirmatory, robustness, and paper-sensitivity analyses |
| `frozen_reproducible_pipeline/` | Frozen public preparation, metadata audit, confirmatory, robustness, test, and verification pipeline; its internal layout is intentionally preserved |
| `main_commented_analyses/` | Commented analysis versions from the supplied archive |
| `prospective_replication_executed/` | Executed prospective-replication analysis and tests |
| `prospective_replication_github_upload/` | Publication-ready replication code and tests |
| `exploratory_joint_mechanism/` | Exploratory joint-geometry and spin-sensitivity analyses |
| `figure_generation_and_validation/` | Figure creation, correction, rendering, and validation tools |
| `site-tools/` | Local launcher, both website-data builders, static-site verifier, and verified distribution-archive builder |

Several archived scripts use their location to find sibling files or project directories. Their subdirectory structure has therefore been retained instead of flattening files with duplicate names.

This directory contains **57 Python files** and **12 Jupyter notebooks**. The frozen full-catalog capsule contributes another **15 Python files**, and `reproducibility/download_full_catalog.py` supplies one portable acquisition helper, bringing the repository total to **73 Python files**. The Windows launcher is retained as an additional `.bat` script. The distribution builder checks every one of these source files against the generated archive.
