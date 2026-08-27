# Reproducibility materials

This directory separates the post-lock 104-event extension from the earlier analysis collection.

## Frozen capsule

[`full-catalog-generalization/`](full-catalog-generalization/) is the supplied final capsule dated 26 August 2026. Its 76 original files are retained at their original relative paths. Do not reorganize or edit files inside this directory without deliberately issuing a new analysis version, because [`FINAL_OUTPUT_SHA256SUMS.txt`](full-catalog-generalization/FINAL_OUTPUT_SHA256SUMS.txt) authenticates 75 packaged artifacts.

The capsule includes acquisition records, fixed design inputs, locked protocols, analysis and test code, pinned Python requirements, event-level tables, population summaries, manifests, figures, and the final scientific interpretation. It does not duplicate the very large public HDF5 releases.

Some archived manifests contain absolute paths from the machine that produced the final run. Those paths are provenance records, not runtime requirements. The commands in [`REPRODUCTION_COMMANDS.md`](full-catalog-generalization/REPRODUCTION_COMMANDS.md) use portable relative paths.

The outcome-blind C01 label correction for GW240925_005809 and other implementation notes are recorded in [`PROTOCOL_DEVIATIONS.md`](full-catalog-generalization/PROTOCOL_DEVIATIONS.md), rather than being silently folded into the final tables.

## Acquire the public inputs

[`download_full_catalog.py`](download_full_catalog.py) reads the frozen 104-row inventory, downloads each official Zenodo file into a directory chosen by the user, resumes `.part` files when the server permits, and verifies the recorded byte size and MD5 before accepting a file. It uses only the Python standard library. The complete input set is 55,134,074,468 bytes, about 51.35 GiB, so check available disk space before starting.

```powershell
python -B reproducibility/download_full_catalog.py --download-dir D:\GWTC5-HDF5
```

Use `--verify-only` to audit an existing directory without network access. Verification records are written to `<download-dir>/_gwtc5_verification/` by default.

## Run and check the analysis

Install [`requirements-analysis.txt`](full-catalog-generalization/requirements-analysis.txt), enter the capsule directory, and follow its numbered reproduction commands. The final output verifier is dependency-light:

```powershell
cd reproducibility\full-catalog-generalization
python -B analysis\verify_full_catalog_outputs.py
```

Figure generation has a separate pinned Pillow requirement. The Node renderer uses the `sharp` version declared in the repository root [`package.json`](../package.json).
