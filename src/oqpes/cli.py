from __future__ import annotations

import argparse
from pathlib import Path

from .alic import solve_alic_revised
from .benchmark import load_config, ensure_dataset, run_benchmark
from .instance import Instance
from .prerna_sharma_2024 import solve_prerna_sharma_2024
from .validate import brute_force_optimum


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ALIC Revised vs Prerna-Sharma 2024 reconstruction benchmark")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pbench = sub.add_parser("benchmark", help="Generate/load instances and run the resumable paired benchmark")
    pbench.add_argument("--config", required=True, help="Path to JSON configuration")
    pbench.add_argument("--root", default=str(_project_root()), help="Project root")

    pgen = sub.add_parser("generate", help="Generate the dataset defined by a benchmark configuration")
    pgen.add_argument("--config", required=True, help="Path to JSON configuration")
    pgen.add_argument("--root", default=str(_project_root()), help="Project root")

    psolve = sub.add_parser("solve", help="Solve one .npz instance")
    psolve.add_argument("instance")
    psolve.add_argument("--method", choices=["alic", "prerna", "both"], default="both")
    psolve.add_argument("--time-limit", type=float, default=300.0)
    psolve.add_argument("--linearization", choices=["auto", "aggregate", "column"], default="auto")
    psolve.add_argument("--bruteforce", action="store_true")

    args = parser.parse_args(argv)
    if args.cmd == "benchmark":
        rcsv, pcsv, xlsx = run_benchmark(args.root, args.config)
        print(f"Results CSV: {rcsv}")
        print(f"Pair checks: {pcsv}")
        print(f"Excel: {xlsx}")
        return 0
    if args.cmd == "generate":
        cfg = load_config(args.config)
        path = ensure_dataset(Path(args.root), cfg)
        print(f"Manifest: {path}")
        return 0
    if args.cmd == "solve":
        inst = Instance.load_npz(args.instance)
        if args.bruteforce:
            x, v, ne, nf = brute_force_optimum(inst)
            print("BRUTE", v, x, "efficient", ne, "feasible", nf)
        if args.method in ("alic", "both"):
            print(solve_alic_revised(inst, time_limit=args.time_limit, linearization_mode=args.linearization))
        if args.method in ("prerna", "both"):
            print(solve_prerna_sharma_2024(inst, time_limit=args.time_limit))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
