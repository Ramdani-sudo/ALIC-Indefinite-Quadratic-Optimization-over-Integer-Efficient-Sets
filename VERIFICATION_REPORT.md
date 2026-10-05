# Recovery and verification report

Date: 2026-10-05

This repository is a recovered, independently rebuilt copy of the Python benchmark package used for the final ALIC Revised vs Prerna--Sharma computational study.

## Recovered benchmark protocol

- Objectives: `p in {5,10,20,50}`.
- Variables: `n in {10,20,50}`.
- Constraints: `m in {10,20,30,50}`.
- Extended campaign: 30 replications per configuration = 1440 paired instances.
- Paper-size campaign: 10 replications per configuration = 480 paired instances.
- Method time limit: 300 wall-clock seconds per method and per instance.
- Execution order alternates deterministically between ALIC Revised and the Prerna--Sharma reconstruction.
- Explicit ILP/MILP subproblems are solved through `scipy.optimize.milp` with the HiGHS backend.

## Exact recovery of the 1440 generated instances

The recovered generator was verified against the SHA-256 values stored in the archived final result file.

Verification output:

```text
Archived result instances: 1440
Generated manifest instances: 1440
Missing: 0
Extra: 0
Manifest hash mismatches: 0
NPZ content hash mismatches: 0
Recovered campaign instances exactly match the hashes of the final results.
```

The recovery therefore reproduces the exact 1440 accepted instance contents, not merely the same dimensions or coefficient distributions.

The accepted-instance generator uses the deterministic seed

```text
seed = 100000000 + 1000000*p + 10000*n + 100*m + rep
```

and the complete candidate draw order `A`, `b`, `C`, `Q`, `d`. The final extended campaign required 7614 rejected complete draws before the 1440 accepted instances were obtained; the largest accepted-attempt index was 112.

## Reference example

Fresh execution of `python examples/reference_example.py` gives:

- brute-force optimum: `x* = (0,10)`, objective `820`;
- ALIC Revised: `OPTIMAL`, objective `820`, 7 master MILPs + 6 primary ILPs = 13 explicit optimization calls, 0 secondary solves, 6/6 uniqueness certificates;
- Prerna--Sharma reconstruction: `OPTIMAL`, objective `820`, 50 ILPs + 6 MILP efficiency tests = 56 explicit optimization calls.

## Automated tests

Fresh test command:

```text
python -m pytest -q
```

Result at recovery verification: 20 tests passed.

## Pilot and resume verification

The three-instance pilot was generated and solved with both methods. All completed pairs returned matching certified optimal objective values. Re-running the same pilot did not duplicate method-instance rows.

## Archived final campaign results

The final 1440-instance result files used for the article are distributed with the reproducibility package and can be regenerated from the same deterministic benchmark protocol.

SHA-256 checksums of the local recovery package were:

```text
281266613ce7bb52803236cfaef9d37b5af0e2f0fe45ac524a3b04041917ffc0  results.csv
a8e41e64280b52be632693581041469b93941b4d1d90a15b247111029fd54509  results.xlsx
3799725b2977837c80e617c3d90a518cca8f5ad6e8ecf1d25704cbc20e192d58  pair_checks.csv
```

The archived campaign contains 1440 paired instances. Both methods certify optimality on 1421 instances and agree on the optimal objective value in all 1421 common certified cases. ALIC certifies 1438/1440 instances and the reference reconstruction 1421/1440.

## Reproducibility limitation

Exact wall-clock times are machine- and solver-version-dependent. The original Windows campaign was run in an Anaconda/Python environment on an 11th-generation Intel Core i7 computer with 16 GB RAM, but the exact Python/SciPy/HiGHS patch versions were not recorded in the result files.

The reference implementation is a Python reconstruction of Prerna--Sharma (2024), not the authors' original implementation.
