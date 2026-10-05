from __future__ import annotations

import time
from typing import Any

import numpy as np

from .alic import MethodResult
from .efficiency import add_original_feasible_set, solve_efficiency
from .instance import Instance
from .milp import LinearModel, remaining_time


def _add_linear_bound(model: LinearModel, x_idx: list[int], coeff: np.ndarray,
                      ub: int | None = None, eq: int | None = None) -> None:
    row = {x_idx[j]: float(coeff[j]) for j in range(len(x_idx)) if coeff[j] != 0}
    if eq is not None:
        model.add_eq(row, float(eq), name="g_level")
    elif ub is not None:
        model.add_constraint(row, ub=float(ub), name="g_upper")


def _add_nogood(model: LinearModel, x_idx: list[int], point: np.ndarray,
                u: np.ndarray, tag: str) -> None:
    """Exclude exactly one bounded integer point using a disjunction."""
    active: list[int] = []
    for j, p in enumerate(point.astype(int)):
        if p > 0:
            b = model.add_var(0, 1, integer=True, name=f"ng_{tag}_{j}_lo")
            M = int(u[j] - p + 1)
            model.add_constraint({x_idx[j]: 1.0, b: float(M)}, ub=float(u[j]), name=f"nglo_{tag}_{j}")
            active.append(b)
        if p < int(u[j]):
            b = model.add_var(0, 1, integer=True, name=f"ng_{tag}_{j}_hi")
            M = int(p + 1)
            model.add_constraint({x_idx[j]: 1.0, b: -float(M)}, lb=0.0, name=f"nghi_{tag}_{j}")
            active.append(b)
    if active:
        model.add_constraint({b: 1.0 for b in active}, lb=1.0, name=f"ng_or_{tag}")
    else:
        model.add_constraint({}, lb=1.0, ub=np.inf, name=f"ng_impossible_{tag}")


def _solve_linear_rank(inst: Instance, gcoef: np.ndarray, g_upper: int | None,
                       time_limit: float | None):
    model = LinearModel()
    x = add_original_feasible_set(model, inst)
    _add_linear_bound(model, x, gcoef, ub=g_upper)
    sol = model.solve({x[j]: float(gcoef[j]) for j in range(inst.n) if gcoef[j] != 0},
                      maximize=True, time_limit=time_limit)
    point = None if sol.x is None else np.rint(sol.x[x]).astype(np.int64)
    return sol, point, model.nvars, model.ncons


def _solve_tie(inst: Instance, gcoef: np.ndarray, level: int,
               excluded: list[np.ndarray], time_limit: float | None):
    model = LinearModel()
    x = add_original_feasible_set(model, inst)
    _add_linear_bound(model, x, gcoef, eq=level)
    for k, point in enumerate(excluded):
        _add_nogood(model, x, point, inst.ensure_upper_bounds(), f"{k}")
    sol = model.solve({}, maximize=True, time_limit=time_limit)
    point = None if sol.x is None else np.rint(sol.x[x]).astype(np.int64)
    return sol, point, model.nvars, model.ncons


def solve_prerna_sharma_2024(inst: Instance, *, time_limit: float | None = 300.0) -> MethodResult:
    """Exact Python reconstruction of the Prerna--Sharma ranking principle."""
    wall0 = time.perf_counter()
    cpu0 = time.process_time()
    u = inst.ensure_upper_bounds()
    n = inst.n
    ilp_calls = 0
    milp_calls = 0
    preprocessing_calls = 0
    efficiency_tests = 0
    rank_levels = 0
    tie_solutions = 0
    nodes = 0
    max_vars = 0
    max_cons = 0

    V = np.zeros(n, dtype=np.int64)
    for j in range(n):
        rem = remaining_time(wall0, time_limit)
        if rem is not None and rem <= 1e-6:
            return MethodResult(
                method="PRERNA_SHARMA_2024_RECONSTRUCTION", status="TIME_LIMIT", objective=None, x=None,
                wall_time=time.perf_counter()-wall0, cpu_time=time.process_time()-cpu0,
                ilp_calls=ilp_calls, milp_calls=milp_calls, preprocessing_calls=preprocessing_calls,
                preprocessing_wall_time=time.perf_counter()-wall0,
                message="time limit during V_j preprocessing",
            )
        model = LinearModel()
        x = add_original_feasible_set(model, inst)
        col = inst.Q[:, j]
        sol = model.solve({x[k]: float(col[k]) for k in range(n) if col[k] != 0},
                          maximize=True, time_limit=rem)
        ilp_calls += 1
        preprocessing_calls += 1
        nodes += sol.mip_node_count or 0
        max_vars = max(max_vars, model.nvars)
        max_cons = max(max_cons, model.ncons)
        if not sol.optimal or sol.x is None:
            status = "TIME_LIMIT" if sol.status == "TIME_OR_LIMIT" else "UNRESOLVED_PREPROCESSING"
            return MethodResult(
                method="PRERNA_SHARMA_2024_RECONSTRUCTION", status=status, objective=None, x=None,
                wall_time=time.perf_counter()-wall0, cpu_time=time.process_time()-cpu0,
                ilp_calls=ilp_calls, milp_calls=milp_calls, preprocessing_calls=preprocessing_calls,
                nodes=nodes, max_model_vars=max_vars, max_model_cons=max_cons,
                preprocessing_wall_time=time.perf_counter()-wall0,
                message=f"V_{j} preprocessing not solved optimally: {sol.message}",
            )
        xx = np.rint(sol.x[x]).astype(np.int64)
        V[j] = int(col @ xx)

    preprocessing_wall_time = time.perf_counter() - wall0
    gcoef = V + inst.d

    pool: dict[tuple[int, ...], dict[str, Any]] = {}
    g_upper: int | None = None
    remaining_g_ub: float = np.inf

    def add_point(pt: np.ndarray, efficient: bool | None = None) -> None:
        key = tuple(int(v) for v in pt)
        if key not in pool:
            pool[key] = {"x": pt.copy(), "h": float(inst.objective(pt)), "efficient": efficient}
        elif efficient is True:
            pool[key]["efficient"] = True

    def process_certified_pool() -> tuple[np.ndarray | None, str | None]:
        nonlocal milp_calls, efficiency_tests, nodes, max_vars, max_cons
        while pool:
            H = max(float(v["h"]) for v in pool.values())
            if H < remaining_g_ub:
                return None, None
            level_keys = [k for k, v in pool.items() if float(v["h"]) == H]
            for k in level_keys:
                if pool[k]["efficient"] is True:
                    return np.asarray(pool[k]["x"], dtype=np.int64), "OPTIMAL"

            for k in level_keys:
                if pool[k]["efficient"] is not None:
                    continue
                rem = remaining_time(wall0, time_limit)
                if rem is not None and rem <= 1e-6:
                    return None, "TIME_LIMIT"
                er = solve_efficiency(inst, np.asarray(pool[k]["x"], dtype=np.int64), time_limit=rem)
                efficiency_tests += 1
                milp_calls += 1
                nodes += er.solve.mip_node_count or 0
                max_vars = max(max_vars, inst.n + inst.p)
                max_cons = max(max_cons, inst.m + inst.p)
                if not er.optimal or er.efficient is None:
                    return None, "TIME_LIMIT" if er.solve.status == "TIME_OR_LIMIT" else "UNRESOLVED_EFFICIENCY"
                pool[k]["efficient"] = bool(er.efficient)
                if er.efficient:
                    return np.asarray(pool[k]["x"], dtype=np.int64), "OPTIMAL"
                if er.witness is not None:
                    add_point(er.witness, efficient=True)
                    hw = float(inst.objective(er.witness))
                    if hw >= H and hw >= remaining_g_ub:
                        return er.witness.copy(), "OPTIMAL"

            if H > remaining_g_ub:
                for k in level_keys:
                    pool.pop(k, None)
                continue
            return None, None
        return None, None

    while True:
        rem = remaining_time(wall0, time_limit)
        if rem is not None and rem <= 1e-6:
            status = "TIME_LIMIT"
            incumbent = None
            break

        rank_sol, first, nv, nc = _solve_linear_rank(inst, gcoef, g_upper, rem)
        ilp_calls += 1
        nodes += rank_sol.mip_node_count or 0
        max_vars = max(max_vars, nv)
        max_cons = max(max_cons, nc)
        if rank_sol.infeasible:
            remaining_g_ub = -np.inf
            incumbent, pstatus = process_certified_pool()
            if incumbent is not None:
                status = "OPTIMAL"
            elif pstatus is not None:
                status = pstatus
            else:
                status = "UNRESOLVED_EMPTY_POOL"
            break
        if not rank_sol.optimal or first is None:
            status = "TIME_LIMIT" if rank_sol.status == "TIME_OR_LIMIT" else "UNRESOLVED_RANKING"
            incumbent = None
            break

        level = int(gcoef @ first)
        rank_levels += 1
        level_points: list[np.ndarray] = [first.copy()]
        add_point(first)
        tie_solutions += 1

        while True:
            rem = remaining_time(wall0, time_limit)
            if rem is not None and rem <= 1e-6:
                status = "TIME_LIMIT"
                incumbent = None
                break
            tie_sol, pt, nv, nc = _solve_tie(inst, gcoef, level, level_points, rem)
            ilp_calls += 1
            nodes += tie_sol.mip_node_count or 0
            max_vars = max(max_vars, nv)
            max_cons = max(max_cons, nc)
            if tie_sol.infeasible:
                break
            if not tie_sol.optimal or pt is None:
                status = "TIME_LIMIT" if tie_sol.status == "TIME_OR_LIMIT" else "UNRESOLVED_TIE"
                incumbent = None
                break
            level_points.append(pt.copy())
            add_point(pt)
            tie_solutions += 1
        else:
            pass
        if 'status' in locals() and status in {"TIME_LIMIT", "UNRESOLVED_TIE"} and incumbent is None:
            break

        g_upper = level - 1
        remaining_g_ub = float(g_upper + inst.alpha)
        incumbent, pstatus = process_certified_pool()
        if incumbent is not None:
            status = "OPTIMAL"
            break
        if pstatus is not None:
            status = pstatus
            incumbent = None
            break

    wall = time.perf_counter() - wall0
    cpu = time.process_time() - cpu0
    obj = None if incumbent is None else float(inst.objective(incumbent))
    return MethodResult(
        method="PRERNA_SHARMA_2024_RECONSTRUCTION", status=status, objective=obj,
        x=None if incumbent is None else incumbent.astype(int).tolist(),
        wall_time=wall, cpu_time=cpu,
        ilp_calls=ilp_calls, milp_calls=milp_calls,
        efficiency_tests=efficiency_tests, rank_levels=rank_levels,
        tie_solutions=tie_solutions, preprocessing_calls=preprocessing_calls,
        nodes=nodes, max_model_vars=max_vars, max_model_cons=max_cons,
        message="exact ranking reconstruction with explicit tied-level enumeration",
        extra={"V": V.tolist(), "g_coeff": gcoef.tolist()},
        preprocessing_wall_time=preprocessing_wall_time,
    )
