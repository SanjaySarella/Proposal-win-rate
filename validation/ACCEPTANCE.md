# Additional acceptance review

PASS for snapshot arithmetic: all 72 agency/type combinations reconcile to an independent CSV/Decimal calculation within half a cent. All 969 identifiers are unique. This verifies the stored snapshot, not freshness, population coverage, or the correctness of the upstream source.

Reproduce with `python validation/acceptance.py`. Results are in `validation/results.json`.

Status: local validation only. Container startup, deployed UI behavior, deployment revision mapping, and live source-link checks remain outstanding. No live service has been updated.
