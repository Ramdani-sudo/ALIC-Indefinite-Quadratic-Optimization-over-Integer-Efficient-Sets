from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Dict, Iterable, Mapping

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


@dataclass
class SolveResult:
    status: str
    success: bool
    optimal: bool
    infeasible: bool
    x: np.ndarray | None
    objective: float | None
    wall_time: float
    message: str
    mip_node_count: int | None = None
    mip_dual_bound: float | None = None
    mip_gap: float | None = None
    raw_status: int | None = None


class LinearModel:
    """Small sparse MILP builder around scipy.optimize.milp."""

    def __init__(self) -> None:
        self.var_lb: list[float] = []
        self.var_ub: list[float] = []
        self.integrality: list[int] = []
        self.names: list[str] = []
        self.rows: list[Dict[int, float]] = []
        self.row_lb: list[float] = []
        self.row_ub: list[float] = []
        self.row_names: list[str] = []

    @property
    def nvars(self) -> int:
        return len(self.var_lb)

    @property
    def ncons(self) -> int:
        return len(self.rows)

    def add_var(self, lb: float = 0.0, ub: float = np.inf, integer: bool = False, name: str = "") -> int:
        idx = self.nvars
        self.var_lb.append(float(lb))
        self.var_ub.append(float(ub))
        self.integrality.append(1 if integer else 0)
        self.names.append(name or f"x{idx}")
        return idx

    def add_vars(self, count: int, lb: float | Iterable[float] = 0.0, ub: float | Iterable[float] = np.inf,
                 integer: bool = False, prefix: str = "x") -> list[int]:
        lbs = [float(lb)] * count if np.isscalar(lb) else list(lb)
        ubs = [float(ub)] * count if np.isscalar(ub) else list(ub)
        return [self.add_var(lbs[i], ubs[i], integer=integer, name=f"{prefix}_{i}") for i in range(count)]

    def add_constraint(self, coeffs: Mapping[int, float], lb: float = -np.inf, ub: float = np.inf,
                       name: str = "") -> int:
        row = {int(k): float(v) for k, v in coeffs.items() if abs(float(v)) > 0.0}
        self.rows.append(row)
        self.row_lb.append(float(lb))
        self.row_ub.append(float(ub))
        self.row_names.append(name or f"c{len(self.rows)-1}")
        return len(self.rows) - 1

    def add_eq(self, coeffs: Mapping[int, float], rhs: float, name: str = "") -> int:
        return self.add_constraint(coeffs, rhs, rhs, name)

    def solve(self, objective: Mapping[int, float] | np.ndarray, *, maximize: bool = True,
              time_limit: float | None = None, presolve: bool = True,
              mip_rel_gap: float = 0.0, node_limit: int | None = None) -> SolveResult:
        n = self.nvars
        if isinstance(objective, np.ndarray):
            c = np.asarray(objective, dtype=float).reshape(-1).copy()
            if c.shape != (n,):
                raise ValueError(f"Objective has {c.shape[0]} coefficients for {n} variables")
        else:
            c = np.zeros(n, dtype=float)
            for j, v in objective.items():
                c[int(j)] = float(v)
        if maximize:
            c = -c

        if self.rows:
            rr, cc, dd = [], [], []
            for i, row in enumerate(self.rows):
                for j, v in row.items():
                    rr.append(i); cc.append(j); dd.append(v)
            A = coo_matrix((dd, (rr, cc)), shape=(len(self.rows), n)).tocsr()
            cons = LinearConstraint(A, np.asarray(self.row_lb), np.asarray(self.row_ub))
        else:
            cons = None

        options: dict[str, object] = {"presolve": bool(presolve), "mip_rel_gap": float(mip_rel_gap), "disp": False}
        if time_limit is not None:
            options["time_limit"] = max(1e-6, float(time_limit))
        if node_limit is not None:
            options["node_limit"] = int(node_limit)

        t0 = time.perf_counter()
        try:
            res = milp(c=c, integrality=np.asarray(self.integrality, dtype=np.int8),
                       bounds=Bounds(np.asarray(self.var_lb), np.asarray(self.var_ub)),
                       constraints=cons, options=options)
            elapsed = time.perf_counter() - t0
        except Exception as exc:
            return SolveResult(status="ERROR", success=False, optimal=False, infeasible=False, x=None,
                               objective=None, wall_time=time.perf_counter()-t0,
                               message=f"milp exception: {exc}", raw_status=None)

        status_code = int(getattr(res, "status", -1))
        message = str(getattr(res, "message", ""))
        x = None if getattr(res, "x", None) is None else np.asarray(res.x, dtype=float)
        fun = getattr(res, "fun", None)
        objective_value = None if fun is None else float(-fun if maximize else fun)

        if status_code == 0:
            status, optimal, infeasible = "OPTIMAL", True, False
        elif status_code == 2:
            status, optimal, infeasible = "INFEASIBLE", False, True
        elif status_code == 3:
            status, optimal, infeasible = "UNBOUNDED", False, False
        elif status_code == 1:
            status, optimal, infeasible = "TIME_OR_LIMIT", False, False
        else:
            status, optimal, infeasible = "OTHER", False, False

        dual = getattr(res, "mip_dual_bound", None)
        if dual is not None:
            dual = float(-dual if maximize else dual)
        nodes = getattr(res, "mip_node_count", None)
        gap = getattr(res, "mip_gap", None)
        return SolveResult(status=status, success=bool(getattr(res, "success", False)), optimal=optimal,
                           infeasible=infeasible, x=x, objective=objective_value, wall_time=elapsed,
                           message=message, mip_node_count=None if nodes is None else int(nodes),
                           mip_dual_bound=dual, mip_gap=None if gap is None else float(gap),
                           raw_status=status_code)


def remaining_time(start: float, total_limit: float | None) -> float | None:
    if total_limit is None:
        return None
    return max(0.0, float(total_limit) - (time.perf_counter() - start))
