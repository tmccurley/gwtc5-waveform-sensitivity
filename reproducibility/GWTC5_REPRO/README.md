# GWTC5 reproduction code and final figures

Folders:
- events/: original confirmation, robustness, sensitivity and input preparation.
- replication/: prospective nine-event replication.
- catalog/: full-catalog calculations and plotting.
- joint/: exploratory joint-distribution calculations.
- population/: reduced population scripts and their src/tests dependencies.
- plotting/: current and retained original figure generators.
- figures/: 12 final figures, with available PDF/PNG/SVG versions.

Redundant code/work and dated project-name wrappers have been removed.
The src/scripts/tests and original event-stage relationships are retained
where programs use them. Scientific source and figure contents are unchanged.
MANIFEST.csv maps every new path to its location in the preceding package.

This remains a code-and-figure distribution. Input datasets, frozen tables,
result tables and saved population caches/evidence are supplied separately.
Some original scripts contain paths or source bindings referring to the
original research workspace; renaming archive folders does not rewrite those
references. Use the recorded inputs and project-root arguments, or restore
the original locations from MANIFEST.csv when needed. No input checks have
been bypassed. This package alone is not a full data-to-paper environment.

Original requirement files are inside their project folders. The population
work used its recorded Ubuntu Python 3.12 runtime; the catalog analysis used
a separate recorded Python 3.14 runtime.

For Windows, extract to a short location such as C:/GWTC5 if the destination
path is already deeply nested. The ZIP filename and internal paths have
both been shortened. Do not run analyses into the preserved original outputs.
