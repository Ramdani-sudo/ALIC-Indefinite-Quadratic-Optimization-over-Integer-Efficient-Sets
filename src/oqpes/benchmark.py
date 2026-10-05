from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .alic import solve_alic_revised
from .generator import GeneratorSettings, generate_dataset
from .instance import Instance
from .prerna_sharma_2024 import solve_prerna_sharma_2024
from .reporting import append_csv, completed_keys, write_pair_csv, write_excel, ALIC_LABEL, REF_LABEL


RESULT_FIELDS = [
    "instance_id", "instance_file", "sha256", "p", "n", "m", "replicate", "seed",
    "generation_mode", "accepted_attempt", "rejected_total", "rejection_reasons", "unit_feasible_count",
    "method", "status", "objective", "x", "wall_time", "cpu_time",
    "lp_calls", "ilp_calls", "milp_calls", "master_calls", "completion_calls",
    "completion_pass1_calls", "completion_pass2_calls", "efficiency_tests",
    "rank_levels", "tie_solutions", "preprocessing_calls", "cuts", "nodes",
    "dual_bound", "gap", "linearization_mode", "representative", "products", "bits",
    "max_model_vars", "max_model_cons", "iterations", "primary_pass_calls",
    "secondary_pass_calls", "secondary_ilp_calls", "secondary_milp_calls",
    "uniqueness_checks", "uniqueness_certificates", "affine_checks", "affine_certificates",
    "master_max_vars", "master_max_cons", "primary_max_vars", "primary_max_cons",
    "secondary_max_vars", "secondary_max_cons", "preprocessing_wall_time", "message", "extra",
]


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dataset(root: Path, cfg: dict[str, Any]) -> Path:
    inst_dir = root / cfg.get("instances_dir", "instances")
    gen = cfg.get("generation", {})
    settings = GeneratorSettings(
        mode=gen.get("mode", "paper_range_complete_draw_rejection"),
        min_unit_points=int(gen.get("min_unit_points", 1)),
        max_attempts=int(gen.get("max_attempts", 50000)),
    )
    return generate_dataset(
        inst_dir,
        p_values=cfg["grid"]["p"], n_values=cfg["grid"]["n"], m_values=cfg["grid"]["m"],
        replications=int(cfg["grid"]["replications"]), settings=settings,
        overwrite=bool(gen.get("overwrite", False)),
        max_instances=cfg.get("max_instances"),
    )


def run_benchmark(root: str | Path, config_path: str | Path) -> tuple[Path, Path, Path]:
    root = Path(root)
    cfg = load_config(config_path)
    manifest = ensure_dataset(root, cfg)
    results_dir = root / cfg.get("results_dir", "results")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_csv = results_dir / cfg.get("results_csv", "results.csv")
    pair_csv = results_dir / cfg.get("pair_csv", "pair_checks.csv")
    excel_path = results_dir / cfg.get("excel_file", "results.xlsx")
    done = completed_keys(results_csv) if cfg.get("resume", True) else set()
    time_limit = float(cfg.get("time_limit_seconds", 300))
    linear_mode = cfg.get("alic", {}).get("linearization_mode", "auto")
    max_agg_dim = int(cfg.get("alic", {}).get("max_aggregate_dimension", 80))

    with manifest.open("r", newline="", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))
    total = len(manifest_rows)

    for idx, mr in enumerate(manifest_rows):
        path = root / cfg.get("instances_dir", "instances") / mr["instance_file"]
        order = [ALIC_LABEL, REF_LABEL] if idx % 2 == 0 else [REF_LABEL, ALIC_LABEL]
        for method in order:
            key = (mr["instance_id"], method)
            if key in done:
                continue
            inst = Instance.load_npz(path)
            if inst.canonical_hash() != mr["sha256"]:
                raise RuntimeError(f"Hash mismatch before execution: {path.name}")
            if method == ALIC_LABEL:
                res = solve_alic_revised(
                    inst, time_limit=time_limit, linearization_mode=linear_mode,
                    max_aggregate_dimension=max_agg_dim,
                )
            else:
                res = solve_prerna_sharma_2024(inst, time_limit=time_limit)
            row = {**mr, **res.to_dict()}
            row["method"] = res.method
            append_csv(results_csv, row, RESULT_FIELDS)
            done.add((mr["instance_id"], res.method))
            calls = res.lp_calls + res.ilp_calls + res.milp_calls
            print(
                f"[{idx+1}/{total}] {mr['instance_id']} | {res.method} | {res.status} | "
                f"objective={res.objective} | wall={res.wall_time:.3f}s | calls={calls}",
                flush=True,
            )
            if Instance.load_npz(path).canonical_hash() != mr["sha256"]:
                raise RuntimeError(f"Instance mutated during execution: {path.name}")
        write_pair_csv(results_csv, pair_csv)

    write_pair_csv(results_csv, pair_csv)
    try:
        write_excel(results_csv, pair_csv, excel_path)
    except PermissionError:
        print(f"Warning: could not update {excel_path.name}; close the workbook and rerun. CSV progress is safe.", flush=True)
    return results_csv, pair_csv, excel_path
