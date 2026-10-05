from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp

from .efficiency import add_original_feasible_set
from .instance import Instance
from .milp import LinearModel


@dataclass
class AffineLevelCertificate:
    is_affine: bool
    gradient: np.ndarray | None
    details: dict[str, object]


def build_primary_completion(inst: Instance, x_bar: np.ndarray):
    """Build the pure ILP primary completion in the original x variables."""
    x_bar = np.asarray(x_bar, dtype=np.int64)
    model = LinearModel()
    x = add_original_feasible_set(model, inst)
    cbar = inst.C @ x_bar
    for i in range(inst.p):
        row = {x[j]: float(inst.C[i, j]) for j in range(inst.n) if inst.C[i, j] != 0}
        model.add_constraint(row, lb=float(cbar[i]), name=f"primary_dom_{i}")
    gamma = np.sum(inst.C, axis=0).astype(np.int64)
    obj = {x[j]: float(gamma[j]) for j in range(inst.n) if gamma[j] != 0}
    return model, x, obj, gamma


def certify_primary_uniqueness(inst: Instance, x_bar: np.ndarray, theta: int) -> tuple[bool, dict[str, object]]:
    """Sufficient exact uniqueness certificate on the primary level."""
    x_bar = np.asarray(x_bar, dtype=np.int64)
    p, n = inst.C.shape
    if theta <= 0:
        return True, {"reason": "zero_primary_improvement", "unique_x": x_bar.astype(int).tolist()}
    if p < n:
        return False, {"reason": "rank_impossible_p_lt_n"}

    C = sp.Matrix(inst.C.tolist())
    _, piv = C.T.rref()
    rows = list(piv)
    if len(rows) < n:
        return False, {"reason": "rank_deficient"}
    rows = rows[:n]
    Csq = C.extract(rows, list(range(n)))
    try:
        inv = Csq.inv()
    except Exception:
        return False, {"reason": "left_inverse_failed"}

    L = sp.zeros(n, p)
    for local_col, original_row in enumerate(rows):
        for j in range(n):
            L[j, original_row] = inv[j, local_col]

    unique: list[int] = []
    intervals: list[tuple[str, str]] = []
    th = sp.Integer(int(theta))
    for j in range(n):
        vals = [sp.Rational(L[j, i]) for i in range(p)]
        lo = sp.Integer(int(x_bar[j])) + th * min(vals)
        hi = sp.Integer(int(x_bar[j])) + th * max(vals)
        lo_i = int(sp.ceiling(lo))
        hi_i = int(sp.floor(hi))
        intervals.append((str(lo), str(hi)))
        if lo_i != hi_i:
            return False, {"reason": "interval_not_singleton", "coordinate": j, "intervals": intervals}
        unique.append(lo_i)
    return True, {"reason": "singleton_integer_intervals", "unique_x": unique, "pivot_rows": rows, "intervals": intervals}


def _integer_nullspace_one_row(gamma: np.ndarray) -> np.ndarray:
    gamma = np.asarray(gamma, dtype=np.int64).reshape(-1)
    n = len(gamma)
    nz = np.flatnonzero(gamma)
    if len(nz) == 0:
        return np.eye(n, dtype=np.int64)
    pivot = int(nz[0])
    gp = int(gamma[pivot])
    cols: list[np.ndarray] = []
    for j in range(n):
        if j == pivot:
            continue
        v = np.zeros(n, dtype=np.int64)
        v[j] = gp
        v[pivot] = -int(gamma[j])
        cols.append(v)
    if not cols:
        return np.zeros((n, 0), dtype=np.int64)
    return np.column_stack(cols)


def certify_affine_on_primary_level(inst: Instance, x_primary: np.ndarray, gamma: np.ndarray) -> AffineLevelCertificate:
    """Exact affine-level test for gamma^T x = tau."""
    x_primary = np.asarray(x_primary, dtype=np.int64)
    gamma = np.asarray(gamma, dtype=np.int64)
    N = _integer_nullspace_one_row(gamma)
    Q = np.asarray(inst.Q, dtype=object)
    if N.shape[1] == 0:
        affine = True
        reduced = np.zeros((0, 0), dtype=object)
    else:
        reduced = np.asarray(N, dtype=object).T @ Q @ np.asarray(N, dtype=object)
        affine = bool(np.all(reduced == 0))
    if not affine:
        return AffineLevelCertificate(False, None, {"nullity": int(N.shape[1])})
    gradient = ((inst.Q + inst.Q.T) @ x_primary + inst.d).astype(np.int64)
    return AffineLevelCertificate(True, gradient, {"nullity": int(N.shape[1])})


def build_affine_secondary(inst: Instance, x_bar: np.ndarray, tau: int, gradient: np.ndarray):
    model = LinearModel()
    x = add_original_feasible_set(model, inst)
    cbar = inst.C @ np.asarray(x_bar, dtype=np.int64)
    for i in range(inst.p):
        row = {x[j]: float(inst.C[i, j]) for j in range(inst.n) if inst.C[i, j] != 0}
        model.add_constraint(row, lb=float(cbar[i]), name=f"secondary_dom_{i}")
    gamma = np.sum(inst.C, axis=0).astype(np.int64)
    grow = {x[j]: float(gamma[j]) for j in range(inst.n) if gamma[j] != 0}
    model.add_eq(grow, float(tau), name="primary_level")
    obj = {x[j]: float(gradient[j]) for j in range(inst.n) if gradient[j] != 0}
    return model, x, obj
