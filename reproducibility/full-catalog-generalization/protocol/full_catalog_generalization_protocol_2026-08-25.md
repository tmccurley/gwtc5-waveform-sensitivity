# Post-lock full-catalog generalization protocol

## Status and purpose

This analysis is a post-lock exploratory/generalization extension of the
frozen GWTC-5.0 confirmatory study and its prospective nine-event H3
replication. It does not replace, enlarge, or retroactively redefine either
locked sample. Its purpose is to determine how broadly the observed
within-event ordering generalizes across the public catalog.

The analysis code may be corrected for an identified implementation error,
but any correction must be documented and applied to every event. Events may
not be included, excluded, or reclassified because of a posterior-disagreement
result.

## Catalog universe

The fixed universe is the 104 events in the archived
`catalog_event_manifest.csv`, each of which was recorded as containing both
of the following released model families:

- `IMRPhenomXPHM-SpinTaylor`
- `IMRPhenomXPNR`

The paired analyses must have a unique common event-specific configuration
tag. Most events use `C00`; GW240925_005809 uses `C01` for both model families.
This catalog-label exception was discovered during the outcome-blind metadata
audit before its posterior values were read. The model-family rule, rather
than a literal `C00` requirement, matches the archived 104-event catalog
screening frame.

The analysis uses version 2 of the GWTC-5.0 parameter-estimation release
(Zenodo records 20348005 and 20348006). Expected filenames come from the
archived catalog manifest. Each local input must match the official MD5 value
in the release's `md5sums.txt` before its posterior values are used.

The 104-event universe is a census of the released O4b candidates meeting the
catalog release criterion, not a random sample of all binary-black-hole
mergers. Any inference beyond this release must therefore remain cautious.

## Prior-role labels

Prior role is recorded only for transparent subgroup summaries; it never
changes eligibility or the metric calculation.

- Five pilot events were informally chosen from the available pool while the
  analysis process was being developed. They had no formal selection
  procedure. They are labeled `development_pilot` and are not represented as
  randomly or prospectively selected.
- The remaining 12 frozen development events are labeled
  `development_diagnostic`.
- The 18 events in the final strict confirmatory sample are labeled
  `confirmatory_strict`.
- The nine events in the final prospectively locked replication sample are
  labeled `prospective_replication_strict`.
- All other catalog events are labeled `previously_unselected`.

Every available event is audited and, when outcome-usable, analyzed. The
labels are not grounds for exclusion.

## Input and failure rules

1. Only an exact expected filename is accepted for an event.
2. A missing, zero-byte, checksum-mismatched, corrupt, or incomplete file is
   recorded as such. It is not replaced by another event.
3. The complete analysis is not declared final unless all 104 expected files
   are present and checksum-verified.
4. No posterior parameter is imputed.
5. If explicit nonuniform posterior-weight fields are found, analysis of that
   event pauses until a documented weighting rule is added.
6. Posterior arrays are read as the equal-weight empirical representation
   released by LVK. Storage checks do not establish sampler convergence.

## Metadata eligibility audit

Before posterior values are analyzed, the two target analyses are compared
using the same strict matched-settings logic used for the confirmatory and
replication studies. A strict Tier A pair requires matching detector data and
set, PSDs and channels, frequency settings, likelihood and marginalization
settings, priors (apart from negligible numeric serialization), sampler
settings and seeds, reference frame, and cosmology. Only the waveform model,
model-specific waveform arguments, human-readable description, and
run-result fields such as sample count or evidence may differ.

The audit reads posterior schema and row count but not posterior values. A
pair with any nonpermitted metadata difference is not called Tier A.

## Endpoints

The main endpoint uses the same H3 parameters and normalized
Wasserstein-1 definition as the locked study:

`NW1 = W1 / mean(XPHM 90% width, XPNR 90% width)`

where each 90% width is the 95th percentile minus the 5th percentile of the
released posterior samples. All finite released rows are used; no random
row cap or resampling is applied. This avoids adding a new sampling seed and
makes the result an exact summary of the stored empirical arrays. Because the
locked analyses used compact arrays capped at 100,000 rows, agreement with
the locked results must be checked on their events and any small numerical
difference attributed to sample representation rather than a change in the
metric definition.

For each event:

- calculate `chi_eff` NW1;
- calculate `luminosity_distance` NW1;
- calculate `delta = chi_eff NW1 - luminosity_distance NW1`;
- record whether delta is positive, zero, or negative.

## Two analysis populations

Two populations answer different questions and must never be conflated.

1. **Strict matched-settings population (main):** every event whose pair
   passes the Tier A metadata audit and contains both required posterior
   fields. This is the appropriate population for interpreting the contrast
   as primarily a waveform-configuration comparison.
2. **All outcome-usable pairs (secondary descriptive):** every event with
   verified files, both target posterior arrays, the required fields, no
   explicit weights, and enough finite rows, whether or not other settings
   match. This shows the released pipeline-pair ordering across as much of the
   catalog as possible, but non-waveform differences can contribute.

## Summaries and tests

For each population, report:

- event count and catalog coverage;
- positive, zero, and negative delta counts;
- positive fraction;
- median delta and interquartile range;
- a one-sided exact sign test for a positive probability greater than 0.5;
- a one-sided Wilcoxon signed-rank test for compatibility with the locked H3
  analysis;
- a fixed-seed event bootstrap percentile interval for the median delta
  (100,000 resamples, seed 20260825).

These p-values are post-lock descriptive/generalization statistics, not new
confirmatory tests. Because the 104 events form a fixed release census, event
counts and effect-size distributions carry more interpretive weight than the
p-values.

## Prespecified subgroup summaries

Report the same counts and median delta without subgroup hypothesis testing
for:

- prior-role label;
- catalog screening-score tertile;
- catalog SNR tertile.

These summaries check whether the ordering is confined to the samples or
strata emphasized earlier. They are descriptive and do not establish a
mechanism.

## Interpretation boundary

The analysis measures disagreement/sensitivity, not accuracy. Observed-event
posteriors have no known true source parameters, so the analysis cannot show
that either waveform configuration is more accurate. It also cannot by
itself establish why the disagreement occurs. Known-truth injections and/or
controlled waveform-component changes would be required for those claims.
