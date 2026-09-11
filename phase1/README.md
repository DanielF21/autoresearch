# Phase 1: the measurement record

This directory holds the scripts that produced every number in `artifacts/measurements.md`
and `artifacts/repo.md`. They are kept exactly as they ran and are excluded from the lint,
type and test checks. Nothing in the harness imports them.

- `scripts/` ran on the laptop and drove sailboxes.
- `inbox/` was uploaded into sailboxes and ran there.

Raw outputs live in `runs/`, which is not committed. The plain English findings are in
`.claude/measurement_*_findings.md`.

The harness reuses two of these files by copying, not importing: `inbox/provenance.py` became
`src/autoresearch/guest/provenance.py` and `inbox/workload.py` became
`src/autoresearch/guest/canary.py`.
