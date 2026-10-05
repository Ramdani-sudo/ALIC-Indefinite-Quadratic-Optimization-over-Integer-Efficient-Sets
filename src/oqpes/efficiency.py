from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .instance import Instance
from .milp import LinearModel, SolveResult


@dataclass
class EfficiencyResult:
    status: str
    optimal: bool
    efficient: bool | None
    theta: int | None
    witness: np.ndarray | None
    solve: SolveResult


def add_original_feasible_set(model: LinearModel, inst: Instance) -> list[int]:
    u = inst.ensure_upper_bounds()
    x_idx = model.add_vars(inst.n, lb=0, ub=u, integer=True, prefix="x")
    for i in range(inst.m):
        coeff = {x_idx[j]: float(inst.A[i, j]) for j in range(inst.n) if inst.A[i, j] != 0}
        model.add_constraint(coeff, ub=float(inst.b[i]), name=f"Aub_{i}")
    return x_idx


def solve_efficiency(inst: Instance, seed: np.ndarray, time_limit: float | None = None) -> EfficiencyResult:
    seed = np.asarray(seed, dtype=np.int64).reshape(-1)
    model = LinearModel()
    x = add_original_feasible_set(model, inst)
    psi = model.add_vars(inst.p, lb=0.0, ub=np.inf, integer=False, prefix="psi")
    cy = inst.C @ seed
    for i in range(inst.p):
        coeff = {x[j]: float(inst.C[i, j]) for j in range(inst.n) if inst.C[i, j] != 0}
        coeff[psi[i]] = -1.0
        model.add_eq(coeff, float(cy[i]), name=f"eff_{i}")
    objective = {psi[i]: 1.0 for i in range(inst.p)}
    sol = model.solve(objective, maximize=True, time_limit=time_limit)
    if not sol.optimal or sol.x is None:
        return EfficiencyResult(sol.status, sol.optimal, None, None, None, sol)
    theta_f = float(sum(sol.x[j] for j in psi))
    theta = int(round(theta_f))
    witness = np.rint(sol.x[x]).astype(np.int64)
    return EfficiencyResult(sol.status, True, theta == 0, theta, witness, sol)
