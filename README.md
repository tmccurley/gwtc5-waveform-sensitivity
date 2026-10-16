# GWTC-5.0 waveform-configuration sensitivity

This repository contains the public website, frozen analyses, generated tables, figures, and the code needed to reproduce a study of how binary-black-hole parameter estimates change between IMRPhenomXPHM-SpinTaylor and IMRPhenomXPNR waveform configurations.

The evidence is deliberately separated by design stage. The original study used an 18-event frozen confirmation sample. A nine-event prospective sample supplied a locked replication of the spin-versus-distance ordering. The post-lock extension then audited all 104 available XPHM-XPNR event pairs and defined 78 strict non-development events as its primary generalization population.

These analyses measure sensitivity to a configuration choice. They do not establish which waveform is more accurate, because the true source parameters are unknown.

## Repository map

| Path | Contents |
| --- | --- |
| [`website/`](website/) | Canonical GitHub Pages source and browser-ready data assets |
| [`python-code/`](python-code/) | Earlier confirmatory, replication, robustness, figure, test, and site-maintenance code |
| [`reproducibility/`](reproducibility/) | Frozen 104-event generalization capsule plus catalog-acquisition support |
| [`results/`](results/) | Generated outputs used by the selected-sample website views |
| [`figures/`](figures/) | Manuscript, supplementary, archived, and regenerated figures |
| [`preview-assets/`](preview-assets/) | Supporting audit tables and retained historical assets |
| [`waveform-comparison/`](waveform-comparison/) | Redirect retained for older dashboard links |
| [`downloads/`](downloads/) | Rebuilt all-code and all-figure distribution archive |

The root [`index.html`](index.html) and [`about.html`](about.html) are small GitHub Pages redirects. Website implementation remains inside `website/`, separate from research code.

## View the complete website

Run:

```powershell
python -B python-code/site-tools/launch_dashboard.py
```

The launcher uses cached confirmatory results when available, refreshes both browser data assets, serves the repository locally, and opens `http://127.0.0.1:8765/website/`. Add `--force` to rerun the confirmatory analysis before launch. The full-catalog website asset is rebuilt from the checksum-preserved result tables, so visitors never need to import data.

To refresh the full-catalog asset without starting the server:

```powershell
python -B python-code/site-tools/build_full_catalog_site_data.py
```

## Reproduce the 104-event extension

The self-contained analysis capsule is [`reproducibility/full-catalog-generalization/`](reproducibility/full-catalog-generalization/). Its internal paths and original files are intentionally unchanged so the supplied 75-entry SHA-256 manifest remains verifiable.

Raw GWTC-5.0 HDF5 releases are too large to duplicate in Git. The repository instead records all 104 official filenames, download URLs, byte sizes, and MD5 values. From an empty directory, acquire and verify them with:

```powershell
python -B reproducibility/download_full_catalog.py --download-dir PATH_TO_HDF5_FILES
```

Then create an isolated Python environment and install the pinned analysis requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r reproducibility/full-catalog-generalization/requirements-analysis.txt
```

Follow [`REPRODUCTION_COMMANDS.md`](reproducibility/full-catalog-generalization/REPRODUCTION_COMMANDS.md) from the capsule directory, substituting the verified HDF5 directory. Final structural and numerical checks are provided by `analysis/verify_full_catalog_outputs.py` and the scripts under `tests/`.

Figure regeneration uses the separately pinned Pillow requirements. The sorted H3 PNG/SVG renderer also needs Node.js and the exact `sharp` dependency declared in [`package.json`](package.json).

## Verify packaged outputs

From the full-catalog capsule:

```powershell
python -B analysis/verify_full_catalog_outputs.py
```

The immutable output hashes are listed in [`FINAL_OUTPUT_SHA256SUMS.txt`](reproducibility/full-catalog-generalization/FINAL_OUTPUT_SHA256SUMS.txt). Archived JSON manifests retain absolute paths from the original execution machine as provenance; reproduction commands use relative paths and do not depend on those locations.

The static-site contract has its own dependency-free check for local links, duplicate IDs, embedded D3, generated source hashes, and finalized row counts:

```powershell
python -B python-code/site-tools/verify_static_site.py
```

## Rebuild the complete ZIP

Run:

```powershell
python -B python-code/site-tools/build_distribution.py
```

This creates `downloads/GWTC5_all_figures_and_code_only.zip`, writes a SHA-256 manifest into the archive, emits a neighboring `.zip.sha256` checksum, and checks every repository `.py`, `.ipynb`, and `.bat` file against its archived copy. The website, both analysis collections, final tables, protocols, figures, requirements, and reproduction records are included.

## GitHub Pages

Publish from the repository branch root. The root redirect opens `website/`, and `.nojekyll` prevents GitHub Pages from treating research paths beginning with underscores as Jekyll internals.

For study context, authorship, acknowledgments, and contact information, see the website [About page](website/about.html).

## Supplied reproduction update

The complete supplied `GWTC5_REPRO.zip` is available in [`downloads/`](downloads/GWTC5_REPRO.zip), with its SHA-256 checksum. Its extracted, manifest-verified contents are preserved in [`reproducibility/GWTC5_REPRO/`](reproducibility/GWTC5_REPRO/README.md). This adds joint-distribution and reduced population code, retained figure generators, and the final figure collection. Browse all supplied files and figures on the website [Code & figures page](website/materials.html). This code-and-figure package does not supply new numerical tables or population caches; existing frozen results retain their original scope.
