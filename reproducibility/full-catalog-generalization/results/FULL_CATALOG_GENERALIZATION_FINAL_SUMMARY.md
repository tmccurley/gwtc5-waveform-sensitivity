# Final full-catalog generalization result

## Primary non-development result

The prespecified primary population is the full 87-event non-development
frame, after removing the 17 development events from the 104-event catalog.
Seventy-eight of those 87 events pass the automated strict matched-settings
audit. No sampling, balancing, or replacement was applied within that frame.

The completed analysis changes the scientific picture in a meaningful way:

- **SNR association:** Spearman rho = 0.5025 between catalog network
  matched-filter SNR and median NW1 across the four primary parameters. The
  two-sided event-bootstrap 95% interval is 0.3092 to 0.6584, and the
  Holm-adjusted one-sided permutation p-value is 2.00e-5 (100,000
  permutations). The earlier 18-event null result therefore does not
  generalize to the full non-development frame.
- **Effective-spin/distance ordering:** 75 of 78 events (96.2%) have
  `chi_eff NW1 > luminosity_distance NW1`. The median delta is 0.05250, its
  bootstrap 95% interval is 0.04115 to 0.05802, the Holm-adjusted one-sided
  Wilcoxon p-value is 6.28e-13, and the exact sign-test p-value is 2.62e-19.
- **Screening-proxy validation:** Spearman rho = 0.9788 between the frozen
  screening score and the complete-posterior screening endpoint, with a
  Holm-adjusted one-sided permutation p-value of 2.00e-5.

The permutation p-values reach the resolution floor imposed by 100,000
permutations; `2.00e-5` is the Holm-adjusted result, not evidence that the true
p-value has been measured more precisely than the Monte Carlo design allows.

## Exploratory mechanism generalization

All 78 primary events were eligible for both the joint `(q, chi_eff)` analysis
and the `chi_p` comparison.

- Joint sliced-Wasserstein disagreement and one-dimensional `chi_eff` NW1
  have Spearman rho = 0.9412.
- The median joint sliced-Wasserstein distance is 0.04539.
- The effective-spin component accounts for a median 0.7489 of the absolute
  two-coordinate median-shift magnitude.
- `chi_eff` NW1 exceeds `chi_p` NW1 in 70/78 events, with median paired
  difference 0.04455. The same 70/78 direction appears under geometric-width,
  larger-width, and KS representations; standardized median displacement is
  less uniform at 61/78.

These results show that the one-dimensional effective-spin signal closely
tracks broader joint movement and is usually larger than the precession-spin
disagreement. They narrow the empirical mechanism boundary but do not identify
a causal waveform component.

## Broader catalog H3 census

The previously observed H3 ordering generalizes to nearly the entire released
GWTC-5.0 catalog, but it is not universal. These broader denominators include
development events and therefore remain descriptive sensitivity summaries.

- **Strict matched-settings population:** 89 of 93 events (95.7%) have
  `chi_eff NW1 > luminosity_distance NW1`.
- **All outcome-usable pairs:** 99 of 104 events (95.2%) have the same ordering.
- **Strict median delta:** 0.05598, where
  `delta = chi_eff NW1 - luminosity_distance NW1`.
- **Strict bootstrap 95% interval for the median:** 0.04606 to 0.06210
  (100,000 resamples; seed 20260825).
- **Strict one-sided exact sign-test p-value:** 3.08e-22.
- **Strict one-sided Wilcoxon p-value:** 6.50e-16.

These are post-lock generalization results. The 18-event frozen confirmation
and nine-event prospective replication retain their original roles and are not
redefined by this analysis.

## What changed scientifically

The original selected samples showed the ordering in 18/18 confirmation events
and 9/9 replication events. The completed primary frame establishes that the
pattern is not confined to those selected samples: it occurs in 75/78 strict
non-development events. The broader 93-event strict catalog gives 89/93.

The full-frame H1 result is also scientifically important. The frozen
18-event sample was too small and deliberately balanced on screening features;
its weak association should not be generalized into a catalog-wide null. The
78-event analysis instead estimates a moderate positive SNR association with
a substantially narrower uncertainty interval.

The five negative-delta events are scientifically important. They replace an
overly simple claim of a universal ordering with a stronger and more accurate
claim: the ordering is highly prevalent across the released catalog, with a
small, identifiable set of exceptions.

## Negative-delta exceptions

| Event | Prior role | Strict matched settings? | Configuration | Delta NW1 |
|---|---|---:|---:|---:|
| GW241116_151753 | development diagnostic | no | C00 | -0.12050 |
| GW241114_024711 | previously unselected | yes | C00 | -0.06689 |
| GW240930_234614 | previously unselected | yes | C00 | -0.02432 |
| GW240925_005809 | previously unselected | yes | C01 | -0.02337 |
| GW241225_082815 | development diagnostic | yes | C00 | -0.00295 |

GW241116_151753 is excluded from the strict population because its paired runs
have multiple non-waveform metadata differences. The other four exceptions
remain after applying the strict matched-settings rules.

## Matched-settings audit

The outcome-blind audit classified 93 pairs as strict Tier A and 11 as having
at least one nonpermitted metadata difference. All 104 pairs nevertheless had
the posterior fields required to calculate the H3 metric, so the broader
104-event result is retained as descriptive evidence.

One event, GW240925_005809, stores the paired models under `C01` rather than
`C00`. Before posterior outcomes were read, the implementation was corrected
to discover the unique event-specific same-prefix XPHM-SpinTaylor/XPNR pair.
That correction and its evidence are documented in
`protocol/implementation_correction_2026-08-26.md`.

## Reproducibility checks

- All 104 inputs match the official Zenodo MD5 checksum manifest.
- All finite released posterior rows were used; no posterior-row cap or random
  subsampling was applied.
- All 27 locked confirmation and replication signs were preserved.
- The largest absolute difference between the recomputed and locked deltas was
  approximately 1.39e-17, which is floating-point rounding scale.
- The automated final-output verifier passed every required check.
- The frozen metric core reproduced all 1,530 archived metric values across
  the 90 original confirmatory event-parameter rows.
- The comprehensive and independent H3 paths agree exactly for all 93 strict
  events (maximum absolute delta difference 0.0).
- All 78 primary events and all 93 strict-catalog events were eligible for the
  joint and `chi_p` mechanism analyses.

The five pilot events remain labeled `development_pilot`. They were informally
chosen during method development, without a formal selection procedure, and
are not described as random or prospective.

## Claim boundaries

This analysis supports claims about **configuration sensitivity, its
catalog-wide prevalence, and a moderate catalog-wide association with SNR**.
It does not establish that either waveform
configuration is more accurate, because the true source parameters of the
observed events are unknown. It also does not by itself establish the physical
mechanism producing the ordering. Known-truth injections or controlled
waveform-component studies would be required for those stronger claims.

The strict 78-event non-development population is the primary generalization
summary. The 93-event strict population is a complete-catalog sensitivity
analysis that includes development events. The 104-event population is
broader still and must be labeled descriptive because non-waveform settings
differ for 11 pairs.
