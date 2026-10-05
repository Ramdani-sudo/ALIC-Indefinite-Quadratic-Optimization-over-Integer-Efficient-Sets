from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

ALIC_LABEL = "ALIC_REVISED"
REF_LABEL = "PRERNA_SHARMA_2024_RECONSTRUCTION"


def append_csv(path: str | Path, row: dict[str, object], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)
        f.flush()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def completed_keys(path: str | Path) -> set[tuple[str, str]]:
    return {(r.get("instance_id", ""), r.get("method", "")) for r in read_csv(path)}


def _num(row: dict[str, str], key: str):
    v = row.get(key)
    if v in (None, "", "None"):
        return None
    return float(v)


def _calls(row: dict[str, str]) -> int:
    return sum(int(float(row.get(k) or 0)) for k in ("lp_calls", "ilp_calls", "milp_calls"))


def build_pair_checks(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_inst: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for r in rows:
        by_inst[r["instance_id"]][r["method"]] = r
    out: list[dict[str, object]] = []
    for iid, mm in sorted(by_inst.items()):
        a = mm.get(ALIC_LABEL)
        p = mm.get(REF_LABEL)
        if not a or not p:
            continue
        aopt = a.get("status") == "OPTIMAL"
        popt = p.get("status") == "OPTIMAL"
        both = aopt and popt
        av, pv = _num(a, "objective"), _num(p, "objective")
        objective_agreement = bool(both and av is not None and pv is not None and abs(av-pv) <= 1e-9)
        decision_agreement = bool(both and a.get("x", "") == p.get("x", ""))
        if both and objective_agreement and decision_agreement:
            outcome = "AGREEMENT"
        elif both and objective_agreement:
            outcome = "MULTIPLE_OPTIMAL_DECISIONS"
        elif both:
            outcome = "OBJECTIVE_DISAGREEMENT"
        elif aopt and not popt:
            outcome = "ALIC_ONLY_OPTIMAL"
        elif popt and not aopt:
            outcome = "REFERENCE_ONLY_OPTIMAL"
        else:
            outcome = "NEITHER_OPTIMAL"
        out.append({
            "instance_id": iid,
            "both_present": True,
            "both_optimal": both,
            "objective_agreement": objective_agreement,
            "decision_agreement": decision_agreement,
            "outcome": outcome,
            "alic_status": a.get("status", ""),
            "prerna_status": p.get("status", ""),
            "alic_objective": av,
            "prerna_objective": pv,
            "alic_time": float(a.get("wall_time") or 0),
            "prerna_time": float(p.get("wall_time") or 0),
            "alic_sp": _calls(a),
            "prerna_sp": _calls(p),
            "sha_match": bool(a.get("sha256") and a.get("sha256") == p.get("sha256")),
        })
    return out


def write_pair_csv(results_csv: str | Path, pair_csv: str | Path) -> None:
    pairs = build_pair_checks(read_csv(results_csv))
    pair_csv = Path(pair_csv)
    pair_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "instance_id", "both_present", "both_optimal", "objective_agreement",
        "decision_agreement", "outcome", "alic_status", "prerna_status",
        "alic_objective", "prerna_objective", "alic_time", "prerna_time",
        "alic_sp", "prerna_sp", "sha_match",
    ]
    with pair_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(pairs)


def write_excel(results_csv: str | Path, pair_csv: str | Path, xlsx_path: str | Path) -> None:
    import xlsxwriter

    results = read_csv(results_csv)
    pairs = read_csv(pair_csv)
    xlsx_path = Path(xlsx_path)
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    wb = xlsxwriter.Workbook(xlsx_path)
    fmt_head = wb.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1, "text_wrap": True})
    fmt_num = wb.add_format({"num_format": "0.0000"})
    fmt_int = wb.add_format({"num_format": "0"})
    fmt_bad = wb.add_format({"bg_color": "#F4CCCC"})
    fmt_good = wb.add_format({"bg_color": "#D9EAD3"})

    int_fields = {
        "p","n","m","replicate","seed","accepted_attempt","rejected_total","unit_feasible_count",
        "lp_calls","ilp_calls","milp_calls","master_calls","completion_calls",
        "completion_pass1_calls","completion_pass2_calls","efficiency_tests","rank_levels",
        "tie_solutions","preprocessing_calls","cuts","nodes","products","bits",
        "max_model_vars","max_model_cons","iterations","primary_pass_calls","secondary_pass_calls",
        "secondary_ilp_calls","secondary_milp_calls","uniqueness_checks","uniqueness_certificates",
        "affine_checks","affine_certificates","master_max_vars","master_max_cons",
        "primary_max_vars","primary_max_cons","secondary_max_vars","secondary_max_cons",
        "alic_sp","prerna_sp",
    }
    num_fields = {
        "wall_time","cpu_time","objective","dual_bound","gap","preprocessing_wall_time",
        "alic_time","prerna_time","alic_objective","prerna_objective",
    }

    def add_sheet(name: str, rows: list[dict[str, str]]) -> None:
        ws = wb.add_worksheet(name)
        if not rows:
            ws.write(0, 0, "No data")
            return
        headers = list(rows[0].keys())
        for c, h in enumerate(headers):
            ws.write(0, c, h, fmt_head)
        for ri, row in enumerate(rows, start=1):
            for c, h in enumerate(headers):
                v = row.get(h, "")
                if v in ("", None):
                    ws.write_blank(ri, c, None)
                    continue
                try:
                    if h in num_fields:
                        ws.write_number(ri, c, float(v), fmt_num)
                    elif h in int_fields:
                        ws.write_number(ri, c, int(float(v)), fmt_int)
                    else:
                        ws.write(ri, c, v)
                except Exception:
                    ws.write(ri, c, v)
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, len(rows), len(headers)-1)
        ws.set_column(0, min(len(headers)-1, 4), 18)
        ws.set_column(5, len(headers)-1, 14)
        if name == "Pair checks" and "objective_agreement" in headers:
            c = headers.index("objective_agreement")
            ws.conditional_format(1,c,len(rows),c,{"type":"text","criteria":"containing","value":"True","format":fmt_good})
            ws.conditional_format(1,c,len(rows),c,{"type":"text","criteria":"containing","value":"False","format":fmt_bad})

    add_sheet("Results", results)
    add_sheet("Pair checks", pairs)
    ws = wb.add_worksheet("Summary")
    ws.write_row(0,0,["Metric","ALIC Revised","Prerna--Sharma reconstruction"],fmt_head)
    by_method=defaultdict(list)
    for r in results:
        by_method[r.get("method","")].append(r)
    metrics=[
        ("Executions", lambda rr: len(rr)),
        ("Optimal", lambda rr: sum(x.get("status")=="OPTIMAL" for x in rr)),
        ("Median wall time (s)", lambda rr: median([float(x["wall_time"]) for x in rr]) if rr else ""),
        ("Mean wall time (s)", lambda rr: mean([float(x["wall_time"]) for x in rr]) if rr else ""),
    ]
    for i,(label,fn) in enumerate(metrics,1):
        ws.write(i,0,label)
        ws.write(i,1,fn(by_method.get(ALIC_LABEL,[])))
        ws.write(i,2,fn(by_method.get(REF_LABEL,[])))
    ws.set_column(0,0,28)
    ws.set_column(1,2,24)
    wb.close()
