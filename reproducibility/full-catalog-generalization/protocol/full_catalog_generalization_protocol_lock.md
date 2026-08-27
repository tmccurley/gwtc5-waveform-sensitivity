# Locked protocol: GWTC-5 full-catalog generalization analysis

## Lock record

- **Status:** locked before acquisition and analysis of the remaining canonical event files
- **Lock timestamp:** 2026-08-25T14:04:25.9897175-07:00
- **User decision:** use all available events to improve scientific precision, breadth, and the ability to detect exceptions
- **Parent stages:** the frozen 18-event confirmatory analysis and the prospectively locked nine-event H3 replication
- **Scope:** a post-confirmatory census/generalization stage; it does not alter the hypotheses, samples, results, or status of either parent stage

At the time of this lock, the parent-study outcomes were already known for the
18 confirmatory events and nine replication events. Exploratory joint-posterior
and `chi_p` results were also known for those 27 events. Several metadata-failed
events had been inspected descriptively. Therefore, this stage is not described
as blinded, independent, or prospectively replicated.

The rules below are frozen before the remaining 56 canonical files in the
87-event non-development frame are acquired and before any full-frame result is
calculated. Any later amendment must be dated, justified, hashed, and made
before the affected outcome is calculated. Implementation-error corrections
must be documented and applied uniformly.

## Scientific purpose

This stage asks whether the patterns found under the balanced 18-event design
and nine-event replication generalize across essentially the usable GWTC-5
release for the same configuration pair. It is designed to:

1. estimate the frequency and distribution of the `chi_eff`-greater-than-
   luminosity-distance ordering without screening-based subsampling;
2. sharpen the estimated association between SNR and four-parameter waveform
   disagreement;
3. identify exceptions and event-level heterogeneity; and
4. test whether the exploratory joint `(q, chi_eff)` and `chi_p` patterns extend
   beyond the original 27 selected events.

It does not test which waveform configuration is more accurate and does not
identify a causal waveform component.

## Populations and reporting order

### Population A: full non-development frame

Population A contains all 87 events remaining after the frozen 17-event
development set was removed from the 104 events carrying both target catalog
labels. It includes the 18 confirmatory and nine replication events because the
goal is a census of the complete non-development frame, not a third independent
sample.

Every Population A event passing the frozen metadata and data-integrity rules is
included. There is no random sampling, balancing, replacement, or outcome-based
removal.

Population A is the primary generalization population.

### Population B: complete paired-label catalog

Population B adds the 17 development events to Population A, giving all 104
events carrying both target labels before metadata exclusions. Its results are
reported separately as a complete-catalog descriptive sensitivity analysis.
Development events never count as independent confirmation.

### Required reporting

Results must state, for each population:

- the number of catalog events considered;
- the number passing the Tier A recorded-settings audit;
- every excluded event and the exact exclusion reason;
- parameter-specific missingness, if any; and
- the exact denominator used for every fraction.

The balanced 18-event and prospective nine-event results remain visible as
historical stages and are not pooled with Population A to create a false claim
of independent replication.

## Frozen configuration pair

- XPHM configuration: `C00:IMRPhenomXPHM-SpinTaylor`
- XPNR configuration: `C00:IMRPhenomXPNR`
- official release: GWTC-5 parameter-estimation release version 2
- expected filename prefix: `IGWN-GWTC5p0-29ebe06b7_25-`

## Eligibility and metadata audit

The same Tier A rule used in the parent analyses applies to every event:

- detector data, detector set, channels, and PSDs match;
- detector and waveform frequency settings match;
- duration and sampling settings match;
- priors match apart from documented negligible serialization differences;
- likelihood, marginalization, calibration, reference-frame, and cosmology
  settings match;
- sampler, sampler arguments, and recorded seeds match; and
- only waveform approximant and model-specific waveform arguments may differ.

Canonical key ordering and numerically equivalent floating-point
representations receive the same normalization used in the parent audit.

The audit is metadata-only. It may verify file integrity, configuration labels,
dataset shape, field names, and finite sample counts, but it may not calculate
posterior summaries or cross-configuration distances.

There are no replacements. An event failing Tier A remains in an exclusion
table and is not replaced. An event lacking one of the four core parameters is
excluded from the four-parameter core endpoints but retained for any
parameter-specific analysis it can support. Missing `chi_p` does not exclude an
event from the core analysis.

## Posterior representation

The analysis uses the official `posterior_samples` datasets as released. Every
stored row receives equal empirical weight. If an explicit nonuniform posterior
weight field is found, analysis pauses for a protocol amendment before that
event's outcomes are calculated.

All finite released rows are used. Unequal sample counts are permitted by the
empirical Wasserstein calculation and are recorded. A matched-size
cross-configuration subsampling check is prespecified below.

## Core parameters and metric

Core scientific parameters:

- detector-frame `chirp_mass`;
- `mass_ratio`;
- `chi_eff`; and
- `luminosity_distance`.

`chirp_mass_source` is additionally retained for screening-score validation.

For posterior arrays `x` and `y`, the primary metric remains

`NW1(x, y) = W1(x, y) / mean(width90(x), width90(y))`,

where `width90` is the 95th percentile minus the 5th percentile. No definition
or constant from the parent analysis is changed.

The parent implementations of 60-bin base-2 Jensen-Shannon divergence,
standardized median displacement, central-90% interval overlap, and log
interval-width ratio are retained as secondary metrics.

## Locked generalization endpoints

### G1: SNR association

For each Tier A event, calculate the median NW1 across the four core
parameters. Test its positive association with catalog network matched-filter
SNR using the parent one-sided Spearman permutation test with 100,000
permutations.

Report Spearman rho, the plus-one-corrected permutation p-value, a two-sided
95% event-bootstrap interval for rho, and a scatterplot containing every
eligible event.

### G2: effective-spin versus luminosity-distance ordering

For each Tier A event, calculate

`delta_i = NW1_i(chi_eff) - NW1_i(luminosity_distance)`.

Report every event-level delta, the positive/zero/negative counts, the exact
positive fraction, median delta, interquartile range, and a two-sided 95%
event-bootstrap interval for the median delta.

The inherited directional test is a one-sided Wilcoxon signed-rank test with
`alternative="greater"`, `zero_method="wilcox"`, and `method="auto"`. An exact
one-sided sign test against positive probability 0.5 is the prespecified
distribution-free sensitivity test.

### G3: catalog-screening validation

Test the positive Spearman association between catalog
`maximum_screening_shift` and the maximum complete-posterior NW1 across
`chirp_mass_source`, `chi_eff`, and `luminosity_distance`, using 100,000
one-sided event-label permutations.

### Multiplicity

G1, G2, and G3 form one locked three-test generalization family. Their primary
p-values are adjusted together using Holm's method at familywise alpha 0.05.
The sign test is a sensitivity test and is not substituted for G2 after results
are known. Emphasis remains on effect sizes, event counts, and intervals.

## Prespecified robustness analyses

For G2, repeat the ordering using:

- geometric-mean width normalization;
- larger-width normalization;
- `log(luminosity_distance)`;
- two-sample Kolmogorov-Smirnov distance;
- NW1 divided by the pair-specific split-posterior reference; and
- 100 deterministic matched-size cross-configuration subsamples per
  event-parameter pair.

The finite-sample reference repeats the parent procedure: 100 random
non-overlapping half-splits per event, configuration, and parameter, with the
larger configuration-specific 95th percentile as the pair reference.

The robustness table reports sign counts and median paired differences under
every representation. No representation may replace the primary NW1 result.

## Prespecified mechanism generalization

These analyses remain post-confirmatory and exploratory:

1. standardized sliced-Wasserstein distance for joint `(q, chi_eff)` using the
   existing 360 fixed directions `theta_k = k*pi/360`, `k=0,...,359`;
2. correlation of joint distance with marginal `chi_eff` NW1;
3. the standardized `chi_eff` share of the joint median-shift magnitude;
4. alignment of the median-shift vector with the averaged within-posterior
   first principal axis; and
5. paired `chi_eff` versus `chi_p` disagreement under primary NW1, the two
   alternate width normalizations, standardized median displacement, and KS
   distance.

These results may narrow or contradict the earlier mechanism interpretation.
They do not receive new confirmatory labels.

## Prespecified exploratory heterogeneity analysis

Using catalog quantities fixed independently of this full-posterior analysis,
calculate Spearman associations between each of the following predictors:

- log network matched-filter SNR;
- total source-frame mass;
- source-frame chirp mass;
- absolute catalog `chi_eff`;
- luminosity distance; and
- maximum screening shift;

and each of the following outcomes:

- four-parameter event-median NW1;
- `chi_eff` NW1; and
- G2 delta.

All available predictor-outcome correlations are reported. Benjamini-Hochberg
false-discovery-rate adjustment is applied once across this complete grid.
Scatterplots or case studies may be chosen after calculation only as
illustrations; they cannot be presented as prespecified tests.

## Random seeds and deterministic settings

- finite-sample half-splits: `20260802` (inherited)
- inherited permutation tests: `20260803`
- event bootstrap and matched-size subsampling: `20260825`
- sliced-Wasserstein directions: deterministic 360-direction grid

Every stochastic result must be reproducible from these values. If separate
random-number streams are needed, they are derived deterministically from the
listed seed and a stable event/parameter identifier, not from Python's
process-randomized `hash()`.

## Missing data, stopping, and deviations

- No posterior value is imputed.
- No event is removed because its disagreement is small, large, or contrary to
  an earlier result.
- No analysis stops early because the ordering is preserved or broken.
- All Tier A decisions are completed before newly acquired posterior outcomes
  are calculated.
- Download or storage failures are reported and retried; they are not
  scientific exclusions.
- Any exhausted or permanently unavailable official file remains listed as
  unavailable, with the corresponding denominator reduced transparently.
- All deviations are recorded before the affected result is interpreted.

## Required reproducibility outputs

The stage must preserve:

- this protocol and its pre-execution SHA-256 digest;
- the 104-event inventory and 87-event Population A frame;
- official record IDs, filenames, file sizes, and published checksums;
- local download verification records;
- the complete metadata-comparison table and Tier A decisions;
- parameter-level metrics and event-level endpoints;
- split-reference and matched-size sensitivity outputs;
- generalization, robustness, mechanism, and heterogeneity result tables;
- figures generated directly from those tables;
- software versions, seeds, execution logs, and output hashes; and
- a protocol-deviation log, even if it contains no deviations.

The large official HDF5 files are referenced by Zenodo record and checksum and
are not duplicated in the public project repository.
