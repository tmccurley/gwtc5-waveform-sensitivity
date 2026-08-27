# Outcome-blind implementation correction: event-specific configuration tag

## Discovery

During the complete metadata-only audit on 2026-08-26,
GW240925_005809 was found to store its released XPHM-XPNR pair as:

- `C01:IMRPhenomXPHM-SpinTaylor`
- `C01:IMRPhenomXPNR`

The initial extension code looked only for literal `C00` labels. The archived
catalog screening manifest, however, had correctly included this event among
the 104 events with both model families and recorded its release tag as C01.

## Timing and outcome boundary

The issue was identified in the outcome-blind metadata stage. No posterior
value or disagreement metric for GW240925_005809 had been loaded or
calculated. The correction therefore cannot depend on its H3 direction or
magnitude.

## Correction

For each event, the audit now requires exactly one posterior-bearing group
ending in each target model-family name and requires the two groups to share
the same configuration prefix. The discovered labels and prefix are written
to the metadata summary. The outcome script reads those audited labels.

This changes no metric, statistical test, threshold, seed, prior-role label,
or eligibility rule. It prevents a metadata-label artifact from excluding one
of the 104 fixed catalog events.

