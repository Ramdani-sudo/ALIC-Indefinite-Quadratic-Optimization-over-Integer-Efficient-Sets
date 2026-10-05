from __future__ import annotations

from dataclasses import dataclass, asdict
import time
from typing import Any

import numpy as np

from .completion import (
    build_primary_completion,
    certify_primary_uniqueness,
    certify_affine_on_primary_level,
    build_affine_secondary,
)
from .instance import Instance
from .linearization import prepare_linearization, instantiate_linearization
from .milp import remaining_time


@dataclass
class MethodResult:
    method: str
    status: str
    objective: float | None
    x: list[int] | None
    wall_time: float
    cpu_time: float
    lp_calls: int = 0
    ilp_calls: int = 0
    milp_calls: int = 0
    master_calls: int = 0
    completion_calls: int = 0
    completion_pass1_calls: int = 0
    completion_pass2_calls: int = 0
    efficiency_tests: int = 0
    rank_levels: int = 0
    tie_solutions: int = 0
    preprocessing_calls: int = 0
    cuts: int = 0
    nodes: int = 0
    dual_bound: float | None = None
    gap: float | None = None
    linearization_mode: str = ""
    representative: str = ""
    products: int = 0
    bits: int = 0
    max_model_vars: int = 0
    max_model_cons: int = 0
    message: str = ""
    extra: dict[str, Any] | None = None
    iterations: int = 0
    primary_pass_calls: int = 0
    secondary_pass_calls: int = 0
    secondary_ilp_calls: int = 0
    secondary_milp_calls: int = 0
    uniqueness_checks: int = 0
    uniqueness_certificates: int = 0
    affine_checks: int = 0
    affine_certificates: int = 0
    master_max_vars: int = 0
    master_max_cons: int = 0
    primary_max_vars: int = 0
    primary_max_cons: int = 0
    secondary_max_vars: int = 0
    secondary_max_cons: int = 0
    preprocessing_wall_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.x is not None:
            d["x"] = "[" + ",".join(map(str, self.x)) + "]"
        if self.extra is not None:
            import json
            d["extra"] = json.dumps(self.extra, sort_keys=True)
        return d


def _time_ok(start: float, limit: float | None) -> bool:
    r = remaining_time(start, limit)
    return r is None or r > 1e-6


def _add_criteria_lower_constraints(built, inst: Instance, lower: np.ndarray) -> None:
    for i in range(inst.p):
        coeff = {built.x_ext[j]: float(inst.C[i, j]) for j in range(inst.n) if inst.C[i, j] != 0}
        built.model.add_constraint(coeff, lb=float(lower[i]), name=f"dom_{i}")


def _criterion_box_lowers(inst: Instance) -> np.ndarray:
    u = inst.ensure_upper_bounds()
    return np.sum(np.minimum(inst.C * u[np.newaxis, :], 0), axis=1).astype(np.int64)


def _add_image_cut(built, inst: Instance, y: np.ndarray, criterion_lowers: np.ndarray, cut_id: int) -> None:
    cy = inst.C @ y
    deltas = built.model.add_vars(inst.p, lb=0, ub=1, integer=True, prefix=f"cut{cut_id}_d")
    for i in range(inst.p):
        M = max(0, int(cy[i]) + 1 - int(criterion_lowers[i]))
        coeff = {built.x_ext[j]: float(inst.C[i, j]) for j in range(inst.n) if inst.C[i, j] != 0}
        coeff[deltas[i]] = coeff.get(deltas[i], 0.0) - float(M)
        built.model.add_constraint(coeff, lb=float(int(cy[i]) + 1 - M), name=f"imgcut_{cut_id}_{i}")
    built.model.add_constraint({d: 1.0 for d in deltas}, lb=1.0, name=f"imgcut_or_{cut_id}")


def _add_incumbent_cutoff(built, inst: Instance, lb_value: float) -> None:
    rhs = float(lb_value - inst.alpha + 1)
    built.model.add_constraint(built.objective, lb=rhs, name="incumbent_cutoff")


def solve_alic_revised(
    inst: Instance,
    *,
    time_limit: float | None = 300.0,
    linearization_mode: str = "auto",
    max_aggregate_dimension: int = 80,
    max_iterations: int | None = None,
) -> MethodResult:
    wall0 = time.perf_counter()
    cpu0 = time.process_time()

    master_calls = primary_calls = 0
    secondary_calls = secondary_ilp = secondary_milp = 0
    uniqueness_checks = uniqueness_certs = 0
    affine_checks = affine_certs = 0
    ilp_calls = milp_calls = 0
    nodes = 0
    cuts: list[np.ndarray] = []
    incumbent: np.ndarray | None = None
    LB = -np.inf
    last_dual = None
    last_gap = None
    master_max_vars = master_max_cons = 0
    primary_max_vars = primary_max_cons = 0
    secondary_max_vars = secondary_max_cons = 0

    try:
        master_plan = prepare_linearization(
            inst, mode=linearization_mode, max_aggregate_dimension=max_aggregate_dimension
        )
        criterion_lowers = _criterion_box_lowers(inst)
    except Exception as exc:
        return MethodResult(
            method="ALIC_REVISED", status="PREPROCESS_ERROR", objective=None, x=None,
            wall_time=time.perf_counter()-wall0, cpu_time=time.process_time()-cpu0,
            message=str(exc), linearization_mode=linearization_mode,
        )
    preprocessing_wall_time = time.perf_counter() - wall0

    iteration = 0
    status = "UNRESOLVED"
    while True:
        iteration += 1
        if max_iterations is not None and iteration > max_iterations:
            status = "ITERATION_LIMIT"
            break
        if not _time_ok(wall0, time_limit):
            status = "TIME_LIMIT"
            break

        master = instantiate_linearization(master_plan)
        for cid, ycut in enumerate(cuts):
            _add_image_cut(master, inst, ycut, criterion_lowers, cid)
        if incumbent is not None:
            _add_incumbent_cutoff(master, inst, LB)
        master_max_vars = max(master_max_vars, master.model.nvars)
        master_max_cons = max(master_max_cons, master.model.ncons)
        master_calls += 1
        milp_calls += 1
        sm = master.model.solve(
            master.objective, maximize=True,
            time_limit=remaining_time(wall0, time_limit),
        )
        nodes += sm.mip_node_count or 0
        last_gap = sm.mip_gap
        if sm.mip_dual_bound is not None:
            last_dual = sm.mip_dual_bound + float(inst.alpha)

        if sm.infeasible:
            status = "INFEASIBLE" if incumbent is None else "OPTIMAL"
            break
        if not sm.optimal or sm.x is None:
            status = "TIME_LIMIT" if sm.status == "TIME_OR_LIMIT" else "UNRESOLVED"
            break

        x_bar = master.original_x(sm.x)
        if not inst.is_feasible(x_bar):
            status = "NUMERICAL_ERROR"
            break
        beta = float(inst.objective(x_bar))

        if not _time_ok(wall0, time_limit):
            status = "TIME_LIMIT"
            break
        pm, px, pobj, gamma = build_primary_completion(inst, x_bar)
        primary_max_vars = max(primary_max_vars, pm.nvars)
        primary_max_cons = max(primary_max_cons, pm.ncons)
        primary_calls += 1
        ilp_calls += 1
        sp = pm.solve(pobj, maximize=True, time_limit=remaining_time(wall0, time_limit))
        nodes += sp.mip_node_count or 0
        if not sp.optimal or sp.x is None:
            status = "TIME_LIMIT" if sp.status == "TIME_OR_LIMIT" else "UNRESOLVED_COMPLETION"
            break
        x_primary = np.rint(sp.x[px]).astype(np.int64)
        theta = int(gamma @ (x_primary - x_bar))
        if theta < 0:
            status = "NUMERICAL_ERROR"
            break

        if theta == 0:
            incumbent = x_bar.copy()
            LB = beta
            status = "OPTIMAL"
            break

        tau = int(gamma @ x_primary)
        uniqueness_checks += 1
        is_unique, unique_details = certify_primary_uniqueness(inst, x_bar, theta)
        if is_unique:
            uniqueness_certs += 1
            y = x_primary.copy()
        else:
            affine_checks += 1
            ac = certify_affine_on_primary_level(inst, x_primary, gamma)
            if ac.is_affine and ac.gradient is not None:
                affine_certs += 1
                secondary_calls += 1
                secondary_ilp += 1
                ilp_calls += 1
                secm, secx, secobj = build_affine_secondary(inst, x_bar, tau, ac.gradient)
                secondary_max_vars = max(secondary_max_vars, secm.nvars)
                secondary_max_cons = max(secondary_max_cons, secm.ncons)
                ss = secm.solve(secobj, maximize=True, time_limit=remaining_time(wall0, time_limit))
                nodes += ss.mip_node_count or 0
                if not ss.optimal or ss.x is None:
                    status = "TIME_LIMIT" if ss.status == "TIME_OR_LIMIT" else "UNRESOLVED_COMPLETION"
                    break
                y = np.rint(ss.x[secx]).astype(np.int64)
            else:
                secondary_calls += 1
                secondary_milp += 1
                milp_calls += 1
                try:
                    sec_plan = prepare_linearization(
                        inst,
                        mode=linearization_mode,
                        max_aggregate_dimension=max_aggregate_dimension,
                        extra_eq_rows=gamma.reshape(1, -1),
                        extra_eq_rhs=np.asarray([tau], dtype=np.int64),
                    )
                except Exception:
                    sec_plan = prepare_linearization(
                        inst,
                        mode="column",
                        max_aggregate_dimension=max_aggregate_dimension,
                        extra_eq_rows=gamma.reshape(1, -1),
                        extra_eq_rhs=np.asarray([tau], dtype=np.int64),
                    )
                sec = instantiate_linearization(sec_plan)
                _add_criteria_lower_constraints(sec, inst, inst.C @ x_bar)
                secondary_max_vars = max(secondary_max_vars, sec.model.nvars)
                secondary_max_cons = max(secondary_max_cons, sec.model.ncons)
                ss = sec.model.solve(
                    sec.objective, maximize=True,
                    time_limit=remaining_time(wall0, time_limit),
                )
                nodes += ss.mip_node_count or 0
                if not ss.optimal or ss.x is None:
                    status = "TIME_LIMIT" if ss.status == "TIME_OR_LIMIT" else "UNRESOLVED_COMPLETION"
                    break
                y = sec.original_x(ss.x)

        if not inst.is_feasible(y):
            status = "NUMERICAL_ERROR"
            break
        hy = float(inst.objective(y))
        if hy > LB:
            incumbent = y.copy()
            LB = hy

        if LB >= beta:
            status = "OPTIMAL"
            break
        cuts.append(y.copy())

    wall = time.perf_counter() - wall0
    cpu = time.process_time() - cpu0
    obj = None if incumbent is None else float(inst.objective(incumbent))
    if status == "OPTIMAL" and obj is not None:
        last_dual = obj
        last_gap = 0.0
    return MethodResult(
        method="ALIC_REVISED", status=status, objective=obj,
        x=None if incumbent is None else incumbent.astype(int).tolist(),
        wall_time=wall, cpu_time=cpu,
        lp_calls=0, ilp_calls=ilp_calls, milp_calls=milp_calls,
        master_calls=master_calls,
        completion_calls=primary_calls,
        completion_pass1_calls=primary_calls,
        completion_pass2_calls=secondary_calls,
        cuts=len(cuts), nodes=nodes,
        dual_bound=last_dual, gap=last_gap,
        linearization_mode=master_plan.mode,
        representative=master_plan.representative,
        products=len(master_plan.projections), bits=master_plan.bit_count,
        max_model_vars=max(master_max_vars, primary_max_vars, secondary_max_vars),
        max_model_cons=max(master_max_cons, primary_max_cons, secondary_max_cons),
        message=f"iterations={iteration}; revised certified completion",
        extra={"master_template_id": master_plan.notes.get("template_id"), "secondary_template_id": None},
        iterations=iteration,
        primary_pass_calls=primary_calls,
        secondary_pass_calls=secondary_calls,
        secondary_ilp_calls=secondary_ilp,
        secondary_milp_calls=secondary_milp,
        uniqueness_checks=uniqueness_checks,
        uniqueness_certificates=uniqueness_certs,
        affine_checks=affine_checks,
        affine_certificates=affine_certs,
        master_max_vars=master_max_vars,
        master_max_cons=master_max_cons,
        primary_max_vars=primary_max_vars,
        primary_max_cons=primary_max_cons,
        secondary_max_vars=secondary_max_vars,
        secondary_max_cons=secondary_max_cons,
        preprocessing_wall_time=preprocessing_wall_time,
    )


solve_alic = solve_alic_revised
