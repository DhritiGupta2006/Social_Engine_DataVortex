# Phase 2 Validation Report

All checks are computed independently in pandas and compared against the SQLite query results in `data/datavortex.db`.

| Check | Status | Actual | Expected | Note |
|---|---|---|---|---|
| E3.1 NULL-platform rows excluded | ✅ PASS | 1784 | 1784 | dataset should have 1,784 NULL-platform posts |
| E3.2 Total post count in SQL matches pandas | ✅ PASS | 10216 | 10216 |  |
| E3.3 SQL returns 5 platform rows | ✅ PASS | 5 | 5 |  |
| E3.4 Per-platform avg_total_engagement matches pandas (tol 0.01) | ✅ PASS | all match | all match |  |
| E3.5 Platform engagement spread is narrow (< 10%) — consistent with Phase 1 Insight #2 | ✅ PASS | 2.96% | < 10% | Phase 1 reported ~3% spread |
| M1.1 Location count matches pandas | ✅ PASS | 33 | 33 |  |
| M1.2 Total post count across all locations is 12,000 | ✅ PASS | 12000 | 12000 |  |
| M1.3 Top-3 locations match pandas independently | ✅ PASS | ['Los Angeles, USA', 'Munich, Germany', 'Shanghai, China'] | ['Los Angeles, USA', 'Munich, Germany', 'Shanghai, China'] |  |
| M1.4 Per-location total_engagement matches pandas (tol 1) | ✅ PASS | all match | all match |  |
| H2.1 No duplicate (user_id, location) pairs in result | ✅ PASS | 0 | 0 |  |
| H2.2 Within each location, rows are ordered DESC by total_engagement | ✅ PASS | none | none |  |
| H2.3 All returned location_rank values are 1, 2, or 3 | ✅ PASS | 0 | 0 |  |
| H2.4 Locations with fewer than 3 active users | ✅ PASS | none found (all locations have ≥ 3 active users) | any value |  |
| H2.5 Rank-3 tie check: 0 location(s) have >1 user at rank 3 | ✅ PASS | 0 location(s) | any (ties allowed) | RANK() used, not ROW_NUMBER() |
| H2.6 Top-1 user per location matches pandas independently | ✅ PASS | 0 mismatch(es) | 0 mismatches |  |

## Summary

- **PASS:** 15
- **FAIL:** 0
- **Total checks:** 15

