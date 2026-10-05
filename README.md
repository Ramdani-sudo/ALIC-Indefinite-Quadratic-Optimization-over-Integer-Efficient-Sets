# ALIC Revised vs Prerna--Sharma 2024 Reconstruction

This repository contains the Python implementation, exact deterministic instance generator, benchmark protocol, reproducibility documentation, and archived result summaries used for the computational study comparing **ALIC Revised** with a Python reconstruction of **Prerna--Sharma (2024)**. The full binary instance set can be regenerated exactly from the published seed rule and generator.

## Authors and affiliations

This research software accompanies the manuscript **“Optimization of an Indefinite Quadratic Function over Integer Efficient Sets.”**

1. **Zoubir Ramdani** — Corresponding author  
   Mathematical Analysis and Applications Laboratory, Faculty of Mathematics and Computer Science, University Mohamed El Bachir El Ibrahimi of Bordj Bou Arreridj, El Anasser 34030, Bordj Bou Arreridj, Algeria.  
   Contact: `z.ramdani@univ-bba.dz`

2. **Boualem Brahmi**  
   Faculty of Mathematics and Computer Science, University Mohamed El Bachir El Ibrahimi of Bordj Bou Arreridj, El Anasser 34030, Bordj Bou Arreridj, Algeria.  
   Contact: `b.brahmi@univ-bba.dz`

3. **Leila Younsi Abbaci**  
   Department of Electrical Engineering, Faculty of Technology, Research Unit LAMOS, University of Bejaia, Targa Ouzemour 06000, Bejaia, Algeria.  
   Contact: `leila.abbaci@univ-bejaia.dz`

4. **El-Ghani Iftissen**  
   LIM Laboratory, Faculty of Exact Sciences, University of Bouira, Bouira 10000, Algeria.  
   Contact: `e.iftissen@univ-bouira.dz`

## Important reproducibility note

The coefficient ranges are taken from the computational protocol reported by Prerna--Sharma: constraint coefficients in `[0,20]`, right-hand-side values in `[0,100]`, and multiobjective/quadratic preference coefficients in `[-100,100]`.

The paper does not specify every random-generation detail. The following are reproducibility choices of this repository:

- NumPy's `default_rng` with a deterministic seed formula;
- symmetric quadratic matrix generation;
- strict-indefiniteness requirement;
- complete-draw rejection of obvious trivial instances;
- arithmetic integer upper bounds derived from the nonnegative constraints.

A candidate is accepted only if it has at least one nonzero feasible unit vector, a nonconstant preference on the cheap probe `{0} + feasible unit vectors`, finite arithmetic bounds, and a strictly indefinite quadratic matrix. A rejected candidate is discarded completely and a fresh full candidate is drawn.

## Benchmark grid

- objectives: `p in {5,10,20,50}`;
- variables: `n in {10,20,50}`;
- constraints: `m in {10,20,30,50}`.

Two complete protocols are included:

- `paper_protocol_480_300s.json`: 10 replications per configuration, 480 paired instances;
- `extended_1440_300s.json`: 30 replications per configuration, 1440 paired instances.

Each method receives **300 wall-clock seconds per instance**. Method-specific preprocessing is included in the method time. Instance generation is common setup and is not charged to either method.

The execution order alternates deterministically between the two methods to reduce order effects.

## Solver backend

All explicit ILP/MILP subproblems are solved with `scipy.optimize.milp`, using the HiGHS backend distributed through SciPy. Internal LP relaxations solved by HiGHS are not counted as separate algorithm-level subproblems.

For ALIC Revised:

- the exact linearized master is a MILP;
- the primary completion / efficiency problem is a pure ILP in the original decision variables;
- the secondary completion is skipped when uniqueness is certified;
- otherwise it is an ILP if the quadratic preference is certified affine on the primary level, and an exact MILP otherwise.

For the Prerna--Sharma reconstruction:

- the `V_j` upper-estimator preprocessing consists of ILPs;
- ranked linear problems and explicit tied-level enumeration are ILPs;
- efficiency tests are MILPs because they combine integer decisions with continuous improvement variables.

The reference implementation is explicitly a **reconstruction** of the published method, not the authors' original source code.

## Windows quick start

1. Install Anaconda/Python, or run `SETUP_ENV.bat` to create `.venv`.
2. Double-click `RUN_REFERENCE_EXAMPLE.bat`.
3. Double-click `RUN_PILOT_300S.bat`.
4. For the full study, double-click `RUN_EXTENDED_1440_300S.bat`.

The benchmark is resumable. `results.csv` is the authoritative append-safe result store. Re-running the same launcher skips completed `(instance, method)` pairs.

## Main commands

```bash
python run.py generate --config config/extended_1440_300s.json --root .
python run.py benchmark --config config/extended_1440_300s.json --root .
python -m pytest -q
```

## Reference example

The illustrative Prerna--Sharma instance has the certified optimum

`x* = (0,10)`, objective `820`.

ALIC Revised solves the example with 13 explicit optimization calls: 7 master MILPs and 6 primary ILPs. The secondary pass is skipped in all six completion iterations by the uniqueness certificate.

## Result interpretation

For agreement between exact methods, the primary criterion is equality of the certified optimal objective value. Two methods may legitimately return different decision vectors when multiple efficient decisions attain the same global optimum.

## Recovery verification

The recovered generator was validated locally against the SHA-256 values of all 1440 accepted instances from the archived final campaign. The GitHub repository keeps the deterministic generator and protocol; users can regenerate the full NPZ instance set locally.

Run the automated tests with:

```bash
python -m pytest -q
```

See `VERIFICATION_REPORT.md` for the recorded 1440-instance hash-recovery evidence and final-campaign checks.

## Rerunning without overwriting the archived results

To perform a new full experiment in a separate result directory, use:

```text
RUN_EXTENDED_1440_300S_RERUN.bat
```

This launcher uses the same exact 1440 instance files but writes new outputs to a separate results directory.
