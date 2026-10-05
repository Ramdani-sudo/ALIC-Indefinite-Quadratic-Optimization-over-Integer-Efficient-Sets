# Results

The complete computational campaign contains 1440 paired instances (2880 method executions).

This repository provides the exact implementation, deterministic generator, experimental protocol, and compact archived summaries needed to reproduce and inspect the study. Wall-clock times are machine- and solver-version-dependent, so new runs will not reproduce the exact timing values byte-for-byte.

## Main findings

- ALIC certified 1438/1440 instances.
- The Prerna--Sharma reconstruction certified 1421/1440 instances.
- Both methods were optimal on 1421 instances and agreed on all 1421 optimal objective values.
- The median explicit solver-call count was 2 for ALIC and 25 for the reference.
- The median common-optimal runtime was 0.433 s for ALIC and 0.168 s for the reference.
- The mean common-optimal runtime was 1.314 s for ALIC and 2.969 s for the reference.
- The PAR-2 mean was 2.587 s for ALIC and 10.846 s for the reference.

See `final_overall_summary.csv` and `docs/RESULTS_SUMMARY.md` for the statistical summary.

To reproduce the full raw campaign, run:

```bash
python run.py benchmark --config config/extended_1440_300s.json --root .
```

The benchmark is resumable and stores `results.csv`, `pair_checks.csv`, and `results.xlsx` in the configured result directory.
