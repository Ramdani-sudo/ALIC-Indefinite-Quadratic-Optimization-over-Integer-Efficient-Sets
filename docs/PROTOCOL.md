# Experimental protocol recovered from the final 1440-instance campaign

## Grid

- `p = 5, 10, 20, 50`
- `n = 10, 20, 50`
- `m = 10, 20, 30, 50`
- 30 replications per configuration
- 48 configurations
- 1440 paired instances
- 2880 method executions

The strict paper-size variant uses 10 replications, for 480 paired instances.

## Deterministic seed

For `(p,n,m,rep)`:

```text
seed = 100000000 + 1000000*p + 10000*n + 100*m + rep
```

Example: `(5,10,10,1) -> 105101001`.

## Complete candidate draw

For each candidate attempt, the RNG is consumed in this exact order:

1. `A` uniformly in integer `[0,20]`;
2. `b` uniformly in integer `[0,100]`;
3. `C` uniformly in integer `[-100,100]`;
4. a symmetric strictly-indefinite `Q` from integer `[-100,100]`;
5. `d` uniformly in integer `[-100,100]`.

The accepted candidate is not modified after generation.

## Rejection filter

A complete draw is rejected when it has no nonzero feasible unit vector, or when the scalar preference is constant on the cheap probe composed of the origin and all feasible unit vectors. The quadratic matrix is required to be strictly indefinite. Finite upper bounds are computed arithmetically from the nonnegative constraints.

For the final 1440-instance campaign:

- total rejected complete draws: 7614;
- maximum accepted-attempt number: 112.

## Fair paired execution

- time limit: 300 wall-clock seconds per method and instance;
- same immutable instance for both methods;
- SHA-256 checked before execution;
- execution order alternates between ALIC-first and Prerna-first across instances;
- method-specific preprocessing is included in method time;
- dataset generation is excluded from both method times;
- internal HiGHS LP relaxations are not counted as explicit algorithmic subproblems.

## Solver

All explicit ILP/MILP models are submitted through `scipy.optimize.milp`, using the HiGHS backend included in SciPy.
