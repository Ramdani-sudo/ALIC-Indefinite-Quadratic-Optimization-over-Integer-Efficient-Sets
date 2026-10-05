from __future__ import annotations

from itertools import product
from math import prod
from typing import Iterable

import numpy as np

from .instance import Instance


def enumerate_feasible(inst: Instance, max_box_points: int = 500_000) -> list[np.ndarray]:
    u = inst.ensure_upper_bounds()
    box = prod(int(v) + 1 for v in u)
    if box > max_box_points:
        raise ValueError(f"Enumeration box has {box} points > limit {max_box_points}")
    out: list[np.ndarray] = []
    for tup in product(*[range(int(v) + 1) for v in u]):
        x = np.asarray(tup, dtype=np.int64)
        if np.all(inst.A @ x <= inst.b):
            out.append(x)
    return out


def efficient_points(inst: Instance, points: Iterable[np.ndarray]) -> list[np.ndarray]:
    pts = list(points)
    if not pts:
        return []
    Cvals = np.asarray([inst.C @ x for x in pts], dtype=np.int64)
    eff: list[np.ndarray] = []
    for i, y in enumerate(Cvals):
        dominates = np.all(Cvals >= y, axis=1) & np.any(Cvals > y, axis=1)
        if not np.any(dominates):
            eff.append(pts[i])
    return eff


def brute_force_optimum(inst: Instance, max_box_points: int = 500_000):
    feas = enumerate_feasible(inst, max_box_points=max_box_points)
    eff = efficient_points(inst, feas)
    if not eff:
        return None, None, 0, len(feas)
    vals = [inst.objective(x) for x in eff]
    idx = int(np.argmax(vals))
    return eff[idx], vals[idx], len(eff), len(feas)
