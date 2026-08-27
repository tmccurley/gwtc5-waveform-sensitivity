# Protocol deviations

No outcome-driven scientific or statistical deviations have been recorded.

One outcome-blind implementation correction was required. GW240925_005809
stores the intended XPHM-SpinTaylor/XPNR pair under the event-specific `C01`
prefix rather than the assumed literal `C00` prefix. The audit discovered this
before reading posterior values, and the analysis now uses the unique
same-prefix pair recorded for each event. The metric, tests, seeds,
eligibility rules, and interpretation boundaries are unchanged. See
`protocol/implementation_correction_2026-08-26.md` for the full record.
