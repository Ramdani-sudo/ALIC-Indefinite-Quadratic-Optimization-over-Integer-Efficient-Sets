from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .instance import Instance, compute_upper_bounds


P_GRID = (5, 10, 20, 50)
N_GRID = (10, 20, 50)
M_GRID = (10, 20, 30, 50)


def benchmark_seed(p: int, n: int, m: int, rep: int) -> int:
    return int(100_000_000 + 1_000_000 * p + 10_000 * n + 100 * m + rep)


@dataclass
class GeneratorSettings:
    mode: str = "paper_range_complete_draw_rejection"
    min_unit_points: int = 1
    max_attempts: int = 50_000
    indef_tol_factor: float = 1e-10


def _strictly_indefinite(Q: np.ndarray, tol_factor: float) -> bool:
    ev = np.linalg.eigvalsh(Q.astype(float))
    scale = max(1.0, float(np.linalg.norm(Q, 2)))
    tol = tol_factor * scale
    return bool(ev[0] < -tol and ev[-1] > tol)


def _draw_symmetric_indefinite(rng: np.random.Generator, n: int, tol_factor: float) -> tuple[np.ndarray, int]:
    attempts = 0
    while True:
        attempts += 1
        upper = rng.integers(-100, 101, size=(n, n), dtype=np.int64)
        upper = np.triu(upper)
        Q = upper + np.triu(upper, 1).T
        if _strictly_indefinite(Q, tol_factor):
            return Q, attempts


def _nonzero_columns(A: np.ndarray) -> bool:
    return bool(np.all(np.any(A != 0, axis=0)))


def _unit_feasible_count(A: np.ndarray, b: np.ndarray) -> int:
    return int(sum(np.all(A[:, j] <= b) for j in range(A.shape[1])))


def _preference_probe_values(A: np.ndarray, b: np.ndarray, Q: np.ndarray, d: np.ndarray) -> list[int]:
    vals = [0]
    n = A.shape[1]
    for j in range(n):
        if np.all(A[:, j] <= b):
            e = np.zeros(n, dtype=np.int64)
            e[j] = 1
            vals.append(int(e @ Q @ e + d @ e))
    return vals


def generate_instance(p: int, n: int, m: int, rep: int,
                      settings: GeneratorSettings | None = None) -> Instance:
    """Generate one deterministic benchmark instance.

    Paper-range reconstruction:
      A in [0,20], b in [0,100], C,Q,d in [-100,100], alpha=0.

    The production campaign uses paper_range_complete_draw_rejection.
    Every candidate is drawn completely in the fixed order A, b, C, Q, d.
    """
    settings = settings or GeneratorSettings()
    seed = benchmark_seed(p, n, m, rep)
    rng = np.random.default_rng(seed)
    rejection_reasons: Counter[str] = Counter()

    if settings.mode == "conditioned_nontrivial":
        target_units = min(n, max(2, int(settings.min_unit_points)))
    else:
        target_units = min(n, max(1, int(settings.min_unit_points)))

    for attempt in range(1, settings.max_attempts + 1):
        A = rng.integers(0, 21, size=(m, n), dtype=np.int64)

        if settings.mode == "conditioned_nontrivial":
            selected = rng.choice(n, size=target_units, replace=False)
            floor_b = np.max(A[:, selected], axis=1)
            b = np.array([rng.integers(int(lo), 101) for lo in floor_b], dtype=np.int64)
            C = rng.integers(-100, 101, size=(p, n), dtype=np.int64)
            Q, q_attempts = _draw_symmetric_indefinite(rng, n, settings.indef_tol_factor)
            d = rng.integers(-100, 101, size=n, dtype=np.int64)
        elif settings.mode in {"paper_rejection", "paper_range_complete_draw_rejection"}:
            b = rng.integers(0, 101, size=m, dtype=np.int64)
            C = rng.integers(-100, 101, size=(p, n), dtype=np.int64)
            Q, q_attempts = _draw_symmetric_indefinite(rng, n, settings.indef_tol_factor)
            d = rng.integers(-100, 101, size=n, dtype=np.int64)
        else:
            raise ValueError(f"Unknown generation mode: {settings.mode}")

        reasons: list[str] = []
        unit_count = _unit_feasible_count(A, b)
        if unit_count < target_units:
            reasons.append("no_nonzero_unit_feasible_point")
        probe = _preference_probe_values(A, b, Q, d)
        if len(set(probe)) < 2:
            reasons.append("constant_preference_probe")

        try:
            u = compute_upper_bounds(A, b)
        except ValueError:
            reasons.append("unbounded_or_invalid_box")
            u = None

        if reasons:
            rejection_reasons.update(reasons)
            continue

        meta = {
            "p": int(p), "n": int(n), "m": int(m), "replicate": int(rep),
            "seed": int(seed),
            "accepted_attempt": int(attempt),
            "rejected_total": int(attempt - 1),
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
            "generation_mode": (
                "paper_range_complete_draw_rejection"
                if settings.mode in {"paper_rejection", "paper_range_complete_draw_rejection"}
                else settings.mode
            ),
            "coefficient_ranges": {
                "A": [0, 20], "b": [0, 100], "C": [-100, 100],
                "Q": [-100, 100], "d": [-100, 100], "alpha": 0,
            },
            "strict_indefinite": True,
            "q_generation_attempts": int(q_attempts),
            "min_unit_points": int(target_units),
            "unit_feasible_count": int(unit_count),
            "nontriviality_filter": "nonzero feasible unit point + nonconstant preference probe",
        }
        inst = Instance(A=A, b=b, C=C, Q=Q, d=d, alpha=0, u=u, meta=meta)
        inst.meta["sha256"] = inst.canonical_hash()
        return inst

    raise RuntimeError(
        f"Could not generate accepted instance p={p}, n={n}, m={m}, rep={rep} "
        f"after {settings.max_attempts} attempts in mode={settings.mode}"
    )


def iter_grid(p_values: Iterable[int] = P_GRID, n_values: Iterable[int] = N_GRID,
              m_values: Iterable[int] = M_GRID, replications: int = 30):
    for p in p_values:
        for n in n_values:
            for m in m_values:
                for rep in range(1, replications + 1):
                    yield int(p), int(n), int(m), int(rep)


def instance_filename(p: int, n: int, m: int, rep: int, seed: int) -> str:
    return f"P{p:02d}_N{n:03d}_M{m:03d}_R{rep:02d}_seed{seed}.npz"


def generate_dataset(out_dir: str | Path, *, p_values: Iterable[int] = P_GRID,
                     n_values: Iterable[int] = N_GRID, m_values: Iterable[int] = M_GRID,
                     replications: int = 30, settings: GeneratorSettings | None = None,
                     overwrite: bool = False, max_instances: int | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "benchmark_manifest.csv"
    rows: list[dict[str, object]] = []
    count = 0
    for p, n, m, rep in iter_grid(p_values, n_values, m_values, replications):
        if max_instances is not None and count >= max_instances:
            break
        seed = benchmark_seed(p, n, m, rep)
        fname = instance_filename(p, n, m, rep, seed)
        path = out_dir / fname
        if path.exists() and not overwrite:
            inst = Instance.load_npz(path)
        else:
            inst = generate_instance(p, n, m, rep, settings=settings)
            inst.save_npz(path)
        rows.append({
            "instance_id": path.stem,
            "p": p, "n": n, "m": m, "replicate": rep, "seed": seed,
            "generation_mode": inst.meta.get("generation_mode", ""),
            "accepted_attempt": inst.meta.get("accepted_attempt", ""),
            "rejected_total": inst.meta.get("rejected_total", ""),
            "rejection_reasons": json.dumps(inst.meta.get("rejection_reasons", {}), sort_keys=True),
            "unit_feasible_count": inst.meta.get("unit_feasible_count", ""),
            "sha256": inst.canonical_hash(),
            "instance_file": fname,
        })
        count += 1

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "instance_id", "p", "n", "m", "replicate", "seed",
            "generation_mode", "accepted_attempt", "rejected_total",
            "rejection_reasons", "unit_feasible_count", "sha256", "instance_file",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return manifest_path
