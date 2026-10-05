# Benchmark instances

This directory contains the exact **1440 NPZ instance files** used by the final extended computational campaign, together with `benchmark_manifest.csv`.

The final study uses:

- `p in {5,10,20,50}`;
- `n in {10,20,50}`;
- `m in {10,20,30,50}`;
- 30 replications for each of the 48 configurations.

The 480-instance paper-size protocol corresponds to the first 10 replications of each configuration. Its manifest is stored under `paper_protocol_480/`; the duplicated NPZ files are intentionally not committed a second time because they are already present in `extended_1440/`.

Each NPZ file has a canonical SHA-256 hash recorded in the manifest and in the archived final result data. The full repository check is:

```bash
python tools/verify_full_repository.py
```

The complete set can also be regenerated deterministically:

```bash
python run.py generate --config config/extended_1440_300s.json --root .
```

The exact seed rule, random-draw order, rejection filters, and coefficient ranges are documented in `docs/PROTOCOL.md`.
