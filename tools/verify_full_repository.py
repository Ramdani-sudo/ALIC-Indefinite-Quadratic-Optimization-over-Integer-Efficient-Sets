from __future__ import annotations

import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oqpes.instance import Instance

RESULTS = ROOT / "results" / "extended_1440_300s" / "results.csv"
MANIFEST = ROOT / "instances" / "extended_1440" / "benchmark_manifest.csv"
INSTANCE_DIR = MANIFEST.parent

with RESULTS.open(encoding="utf-8-sig", newline="") as f:
    result_rows = list(csv.DictReader(f))

expected: dict[str, str] = {}
for row in result_rows:
    iid = row["instance_id"]
    sha = row["sha256"]
    if iid in expected and expected[iid] != sha:
        raise SystemExit(f"Inconsistent result hashes for {iid}")
    expected[iid] = sha

with MANIFEST.open(encoding="utf-8", newline="") as f:
    manifest_rows = list(csv.DictReader(f))
manifest = {row["instance_id"]: row for row in manifest_rows}

missing = sorted(set(expected) - set(manifest))
extra = sorted(set(manifest) - set(expected))
manifest_mismatch = []
content_mismatch = []
missing_files = []

for iid, sha in expected.items():
    row = manifest.get(iid)
    if row is None:
        continue
    if row["sha256"] != sha:
        manifest_mismatch.append(iid)
    path = INSTANCE_DIR / row["instance_file"]
    if not path.exists():
        missing_files.append(iid)
        continue
    got = Instance.load_npz(path).canonical_hash()
    if got != sha:
        content_mismatch.append(iid)

print(f"Result rows: {len(result_rows)}")
print(f"Unique archived instances: {len(expected)}")
print(f"Manifest rows: {len(manifest_rows)}")
print(f"NPZ files: {len(list(INSTANCE_DIR.glob('*.npz')))}")
print(f"Missing manifest entries: {len(missing)}")
print(f"Extra manifest entries: {len(extra)}")
print(f"Manifest hash mismatches: {len(manifest_mismatch)}")
print(f"Missing NPZ files: {len(missing_files)}")
print(f"NPZ content hash mismatches: {len(content_mismatch)}")

if (
    len(result_rows) != 2880
    or len(expected) != 1440
    or len(manifest_rows) != 1440
    or missing
    or extra
    or manifest_mismatch
    or missing_files
    or content_mismatch
):
    raise SystemExit(1)

print("Full repository verification passed: all 1440 benchmark instances match the archived result hashes.")
