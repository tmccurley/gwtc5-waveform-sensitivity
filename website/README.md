# Website source

`index.html` and `about.html` are the canonical website pages. `styles.css` supplies the shared black-hole visual system, while `site-shell.js` adds reduced-motion-aware page transitions and scroll feedback. The main visualization uses D3, explanatory text, keyboard-readable marks, and static data assets, so it works on GitHub Pages without a database or a user import step.

## Browser data

| Asset | Source | Builder |
| --- | --- | --- |
| `assets/waveform-comparison-data.js` | Selected-sample confirmatory CSV tables | `python-code/site-tools/launch_dashboard.py` |
| `assets/full-catalog-generalization-data.js` | Frozen 78-event primary, 93-event strict, and 104-event census tables | `python-code/site-tools/build_full_catalog_site_data.py` |
| Embedded `GWTC5_PREVIEW_TABLES` block in `index.html` | Retained robustness, audit, and supporting tables | Archived project outputs |

The generated full-catalog asset records SHA-256 values for each source table. Its builder refuses population counts that differ from the finalized 78 primary, 93 strict, and 104 census rows. D3 7.9.0 is embedded in `index.html`, so chart rendering does not depend on a third-party content-delivery network.

## Local preview

From the repository root:

```powershell
python -B python-code/site-tools/launch_dashboard.py
```

The local server must use the repository root, not this directory alone, because the research-material links intentionally point to protocols and checksums outside `website/`. The root GitHub Pages entry point follows the same path layout.

Before publishing, run `python -B python-code/site-tools/verify_static_site.py` from the repository root. It checks local links, duplicate identifiers, generated asset formats, authenticated full-catalog sources, and the finalized row counts.
