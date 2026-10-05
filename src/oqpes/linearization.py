from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from math import ceil, gcd, log2
from functools import reduce
from typing import Iterable

import numpy as np
import sympy as sp

from .instance import Instance, standard_form
from .milp import LinearModel


def _lcm(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return abs(a // gcd(a, b) * b)


def _frac(x: sp.Rational | int | float | Fraction) -> Fraction:
    if isinstance(x, Fraction):
        return x
    if isinstance(x, sp.Rational):
        return Fraction(int(x.p), int(x.q))
    if isinstance(x, (int, np.integer)):
        return Fraction(int(x), 1)
    return Fraction(x)


def _box_linear_bounds(coeff: Iterable[Fraction], u: np.ndarray) -> tuple[Fraction, Fraction]:
    lo = Fraction(0, 1)
    hi = Fraction(0, 1)
    for c, uj in zip(coeff, u):
        val = c * int(uj)
        if val < 0:
            lo += val
        else:
            hi += val
    return lo, hi


@dataclass
class Projection:
    a: np.ndarray
    d: list[Fraction]
    t_lower: int
    t_upper: int
    v_lower: Fraction
    v_upper: Fraction

    @property
    def width(self) -> int:
        return int(self.t_upper - self.t_lower)

    @property
    def bits(self) -> int:
        w = self.width
        return 0 if w <= 0 else int(ceil(log2(w + 1)))


@dataclass
class LinearizationPlan:
    instance: Instance
    Aeq: np.ndarray
    beq: np.ndarray
    u_ext: np.ndarray
    C_ext: np.ndarray
    Q_ext: np.ndarray
    d_ext: np.ndarray
    linear_coeff: list[Fraction]
    projections: list[Projection]
    mode: str
    representative: str
    aggregate_rank: int
    bit_count: int
    notes: dict[str, object] = field(default_factory=dict)


@dataclass
class BuiltLinearization:
    model: LinearModel
    x_ext: list[int]
    objective: dict[int, float]
    original_n: int
    plan: LinearizationPlan

    def original_x(self, solution: np.ndarray) -> np.ndarray:
        vals = solution[self.x_ext[: self.original_n]]
        return np.rint(vals).astype(np.int64)


def _representatives(B: np.ndarray) -> list[tuple[str, np.ndarray]]:
    B = np.asarray(B, dtype=np.int64)
    n = B.shape[0]
    upper = np.zeros_like(B)
    lower = np.zeros_like(B)
    for i in range(n):
        upper[i, i] = B[i, i]
        lower[i, i] = B[i, i]
        for j in range(i + 1, n):
            s = int(B[i, j] + B[j, i])
            upper[i, j] = s
            lower[j, i] = s
    out = [("K=0", B.copy()), ("upper-triangular", upper), ("lower-triangular", lower)]
    unique: list[tuple[str, np.ndarray]] = []
    seen: set[bytes] = set()
    for name, M in out:
        key = M.tobytes()
        if key not in seen:
            seen.add(key)
            unique.append((name, M))
    return unique


def _aggregate_factorization(Aeq: np.ndarray, Be: np.ndarray, d_ext: np.ndarray, beq: np.ndarray,
                             u_ext: np.ndarray, representative: str) -> LinearizationPlan | None:
    N = Aeq.shape[1]
    A_sp = sp.Matrix(Aeq.tolist())
    B_sp = sp.Matrix(Be.tolist())
    AT = A_sp.T

    _, piv_a = AT.rref()
    piv_a = list(piv_a)
    ATi = AT[:, piv_a] if piv_a else sp.zeros(N, 0)
    b_i = [int(beq[i]) for i in piv_a]
    rank_a = len(piv_a)

    G = ATi.row_join(B_sp)
    _, piv_g = G.rref()
    d_col_indices = [j - rank_a for j in piv_g if j >= rank_a]
    D = B_sp[:, d_col_indices] if d_col_indices else sp.zeros(N, 0)
    r = len(d_col_indices)
    M = ATi.row_join(D)
    q = M.shape[1]

    if q == 0:
        if any(Be.ravel()):
            return None
        coeff = sp.zeros(0, N)
    else:
        _, piv_rows = M.T.rref()
        piv_rows = list(piv_rows)
        if len(piv_rows) != q:
            return None
        Msub = M.extract(piv_rows, list(range(q)))
        Bsub = B_sp.extract(piv_rows, list(range(N)))
        try:
            coeff = Msub.inv() * Bsub
        except Exception:
            return None
        if M * coeff != B_sp:
            return None

    Rind = coeff[:rank_a, :] if rank_a else sp.zeros(0, N)
    H = coeff[rank_a:, :] if r else sp.zeros(0, N)

    linear = [Fraction(int(d_ext[j]), 1) for j in range(N)]
    for j in range(N):
        add = Fraction(0, 1)
        for i in range(rank_a):
            add += _frac(Rind[i, j]) * b_i[i]
        linear[j] += add

    projections: list[Projection] = []
    for k in range(r):
        row = [_frac(H[k, j]) for j in range(N)]
        den_lcm = 1
        for h in row:
            den_lcm = _lcm(den_lcm, h.denominator)
        a = np.array([int(h * den_lcm) for h in row], dtype=object)
        dvec = [_frac(D[j, k]) / den_lcm for j in range(N)]

        nz = [abs(int(v)) for v in a if int(v) != 0]
        if not nz:
            continue
        g = reduce(gcd, nz)
        if g > 1:
            a = np.array([int(v) // g for v in a], dtype=object)
            dvec = [v * g for v in dvec]
        first = next(int(v) for v in a if int(v) != 0)
        if first < 0:
            a = -a
            dvec = [-v for v in dvec]
        a64 = np.asarray([int(v) for v in a], dtype=np.int64)

        t_lo_f, t_hi_f = _box_linear_bounds([Fraction(int(v), 1) for v in a64], u_ext)
        if t_lo_f.denominator != 1 or t_hi_f.denominator != 1:
            return None
        t_lo, t_hi = int(t_lo_f), int(t_hi_f)
        v_lo, v_hi = _box_linear_bounds(dvec, u_ext)

        if t_lo != 0:
            for j in range(N):
                linear[j] += Fraction(t_lo, 1) * dvec[j]
        if t_hi > t_lo:
            projections.append(Projection(a64, dvec, t_lo, t_hi, v_lo, v_hi))

    bit_count = sum(p.bits for p in projections)
    return LinearizationPlan(
        instance=None,
        Aeq=np.asarray(Aeq, dtype=np.int64), beq=np.asarray(beq, dtype=np.int64),
        u_ext=np.asarray(u_ext, dtype=np.int64), C_ext=np.empty((0, N), dtype=np.int64),
        Q_ext=np.asarray(Be, dtype=np.int64), d_ext=np.asarray(d_ext, dtype=np.int64),
        linear_coeff=linear, projections=projections, mode="aggregate",
        representative=representative, aggregate_rank=r, bit_count=bit_count,
        notes={"rank_A": rank_a, "selected_B_columns": d_col_indices},
    )


def _column_plan(inst: Instance, sf: dict[str, np.ndarray]) -> LinearizationPlan:
    B = sf["Q_ext"]
    N = B.shape[0]
    linear = [Fraction(int(v), 1) for v in sf["d_ext"]]
    projs: list[Projection] = []
    for j in range(N):
        col = B[:, j]
        if not np.any(col):
            continue
        a = np.zeros(N, dtype=np.int64)
        a[j] = 1
        dvec = [Fraction(int(v), 1) for v in col]
        t_lo, t_hi = 0, int(sf["u_ext"][j])
        v_lo, v_hi = _box_linear_bounds(dvec, sf["u_ext"])
        if t_hi > t_lo:
            projs.append(Projection(a, dvec, t_lo, t_hi, v_lo, v_hi))
    return LinearizationPlan(
        instance=inst, Aeq=sf["Aeq"], beq=sf["beq"], u_ext=sf["u_ext"],
        C_ext=sf["C_ext"], Q_ext=sf["Q_ext"], d_ext=sf["d_ext"],
        linear_coeff=linear, projections=projs, mode="column",
        representative="uncompressed-column", aggregate_rank=len(projs),
        bit_count=sum(p.bits for p in projs), notes={},
    )


def prepare_linearization(inst: Instance, mode: str = "auto", max_aggregate_dimension: int = 80, *,
                          extra_eq_rows: np.ndarray | None = None,
                          extra_eq_rhs: np.ndarray | None = None) -> LinearizationPlan:
    sf = standard_form(inst)
    if extra_eq_rows is not None:
        E = np.asarray(extra_eq_rows, dtype=np.int64)
        if E.ndim == 1:
            E = E.reshape(1, -1)
        if E.shape[1] != inst.n:
            raise ValueError("extra_eq_rows must have n columns")
        rhs = np.asarray(extra_eq_rhs, dtype=np.int64).reshape(-1)
        if len(rhs) != E.shape[0]:
            raise ValueError("extra_eq_rhs length mismatch")
        ext = np.hstack([E, np.zeros((E.shape[0], inst.m), dtype=np.int64)])
        sf = dict(sf)
        sf["Aeq"] = np.vstack([sf["Aeq"], ext])
        sf["beq"] = np.concatenate([sf["beq"], rhs])
    fallback = _column_plan(inst, sf)
    if mode == "column":
        return fallback
    if mode not in {"aggregate", "auto"}:
        raise ValueError(f"Unknown linearization mode {mode}")

    N = sf["Aeq"].shape[1]
    if mode == "auto" and N > max_aggregate_dimension:
        fallback.notes["auto_reason"] = f"extended dimension {N} exceeds aggregate threshold {max_aggregate_dimension}"
        return fallback

    candidates: list[LinearizationPlan] = []
    for rep_name, Be in _representatives(sf["Q_ext"]):
        try:
            plan = _aggregate_factorization(sf["Aeq"], Be, sf["d_ext"], sf["beq"], sf["u_ext"], rep_name)
        except Exception:
            plan = None
        if plan is not None:
            plan.instance = inst
            plan.C_ext = sf["C_ext"]
            candidates.append(plan)

    if not candidates:
        if mode == "aggregate":
            raise RuntimeError("Exact aggregate factorization failed for all representatives")
        fallback.notes["auto_reason"] = "aggregate factorization failed; used exact column fallback"
        return fallback

    if mode == "aggregate":
        return min(candidates, key=lambda q: (q.bit_count, q.aggregate_rank, q.representative))

    all_plans = candidates + [fallback]
    best = min(all_plans, key=lambda q: (q.bit_count, q.aggregate_rank, 0 if q.mode == "aggregate" else 1))
    best.notes["candidate_bit_counts"] = {
        f"{q.mode}:{q.representative}": q.bit_count for q in all_plans
    }
    return best


def instantiate_linearization(plan: LinearizationPlan) -> BuiltLinearization:
    model = LinearModel()
    N = len(plan.u_ext)
    x_ext = model.add_vars(N, lb=0, ub=plan.u_ext, integer=True, prefix="z")

    for i in range(plan.Aeq.shape[0]):
        coeff = {x_ext[j]: float(plan.Aeq[i, j]) for j in range(N) if plan.Aeq[i, j] != 0}
        model.add_eq(coeff, float(plan.beq[i]), name=f"Aeq_{i}")

    obj: dict[int, float] = {}
    for j, c in enumerate(plan.linear_coeff):
        if c:
            obj[x_ext[j]] = float(c)

    for k, pr in enumerate(plan.projections):
        W = pr.width
        p = pr.bits
        if W <= 0 or p == 0:
            continue
        bits = model.add_vars(p, lb=0, ub=1, integer=True, prefix=f"zbit{k}")
        coeff = {x_ext[j]: float(pr.a[j]) for j in range(N) if pr.a[j] != 0}
        for q, bj in enumerate(bits):
            coeff[bj] = coeff.get(bj, 0.0) - float(2 ** q)
        model.add_eq(coeff, float(pr.t_lower), name=f"proj_{k}")
        model.add_constraint({bj: float(2 ** q) for q, bj in enumerate(bits)}, ub=float(W), name=f"range_{k}")

        v_idx = model.add_var(float(pr.v_lower), float(pr.v_upper), integer=False, name=f"v_{k}")
        v_eq = {v_idx: 1.0}
        for j, dj in enumerate(pr.d):
            if dj:
                v_eq[x_ext[j]] = v_eq.get(x_ext[j], 0.0) - float(dj)
        model.add_eq(v_eq, 0.0, name=f"vdef_{k}")

        vlo = float(pr.v_lower)
        vhi = float(pr.v_upper)
        wlo = min(0.0, vlo, vhi)
        whi = max(0.0, vlo, vhi)
        for q, bj in enumerate(bits):
            w = model.add_var(wlo, whi, integer=False, name=f"w_{k}_{q}")
            model.add_constraint({w: 1.0, bj: -vlo}, lb=0.0, name=f"mcc1_{k}_{q}")
            model.add_constraint({w: 1.0, bj: -vhi}, ub=0.0, name=f"mcc2_{k}_{q}")
            model.add_constraint({w: 1.0, v_idx: -1.0, bj: -vhi}, lb=-vhi, name=f"mcc3_{k}_{q}")
            model.add_constraint({w: 1.0, v_idx: -1.0, bj: -vlo}, ub=-vlo, name=f"mcc4_{k}_{q}")
            obj[w] = obj.get(w, 0.0) + float(2 ** q)

    return BuiltLinearization(model=model, x_ext=x_ext, objective=obj, original_n=plan.instance.n, plan=plan)


def exact_shifted_objective(inst: Instance, x: np.ndarray) -> int:
    return int(inst.objective(x) - inst.alpha)
